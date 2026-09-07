"""
model.py — DINCAE 2.0's network architecture (rewritten in PyTorch)
=================================================

Follows the reference implementation `reference/DINCAE.jl/src/model.jl`.
**Copies 2.0's architecture, not the 1.0 paper's Table 1** -- the two differ a
lot, and 2.0 sec.4 gives the reasons:

  * 1.0's **fully-connected bottleneck + dropout are removed**. Reason: an FC
    layer requires the training and inference input matrices to be exactly the
    same size, which does not work for a large domain (2.0 sec.4, citing Long et
    al. 2015 FCN). In the code, `dropout_rate_train = 0.3` is commented out.
  * The skip connection changes from **concat to addition** (SumSkip, 2.0 Eq.2).
    2.0 sec.5.1: on the same Ligurian benchmark as 1.0, 0.3835 -> **0.3604**. The
    mechanism is the residual-network one (He et al. 2016), easing vanishing
    gradients.
  * Pooling uses **MeanPool** (hardcoded at `model.jl:268`). Note 2.0's Table 1
    text says max pooling -- **the paper and code disagree**. This follows the
    code (1.0's ablation also supports avg: 0.3835 vs 0.3900), but this was never
    re-verified in 2.0; `--pool max` is available for an A/B test.
  * **The refinement step** (2.0 sec.2.2 Eq.4): a second, identically-structured
    network consumes `cat(the first level's raw output, the input)`, with
    unshared weights; the loss is computed and weighted for **every level's
    output** (default alpha=0.3, alpha'=0.7). 2.0 Table 2: 0.60 -> **0.55**, a
    much bigger gain than adding physical auxiliary fields (worth only 0.03).

Output parameterisation (2.0 Eq.6-7; `model.jl:16-29`): the network's first slice
of output is **not the mean, but m/sigma^2** (information form), in the same
coordinate frame as the input/target.

    inv_sigma^2 = exp(min(x2, gamma));   sigma^2 = 1/max(inv_sigma^2, mu);   m = x1*sigma^2

gamma = log(min_std_err^-2) = 10, mu = 1e-3 (corresponding to a sigma range of
0.0067-31.6). The min/max only matter during the first few epochs, while the
weights are still close to random.

Size: the ATC corridor grid is 36x12, 3 pooling levels -> 18x6 -> 9x3 -> 5x2.
Follows 2.0's altimetry benchmark (177x69 -> 3 levels -> 23x9, filters 32/64/96).
Odd sizes are handled by `ceil_mode` pooling + cropping after upsampling,
equivalent to the reference implementation's `sz_small = sz÷2 + odd` and
`croppadding`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MIN_STD_ERR = 0.006737946999085467          # exp(-5), the reference implementation's default
GAMMA = float(torch.log(torch.tensor(MIN_STD_ERR ** -2)))       # = 10.0
MU = 1e-3


def transform_msigma2(x, gamma: float = GAMMA, mu: float = MU):
    """(B, 2*nvar, H, W) raw output -> (mean, sigma^2), nvar channels each.

    Follows `model.jl:16-41`: each output variable takes **two slices**, the
    first is m/sigma^2, the second is log(1/sigma^2).
    """
    assert x.shape[1] % 2 == 0, "the number of output channels must be 2x the number of variables"
    x1 = x[:, 0::2]                                     # m/sigma^2
    x2 = x[:, 1::2]                                     # log(1/sigma^2)
    inv_s2 = torch.exp(torch.clamp(x2, max=gamma))
    s2 = 1.0 / torch.clamp(inv_s2, min=mu)
    return x1 * s2, s2


class _Level(nn.Module):
    """One encoder-decoder block, equivalent to the `inner` chain in `recmodel4`
    (with an optional SumSkip).

        x -> conv(enc_in->enc_out) -> ReLU -> pool2 -> inner -> up2 -> crop
          -> conv(dec_in->dec_out) -> [ReLU]
        if skip: returns x + the above (requires enc_in == dec_out)
    """

    def __init__(self, enc_in, enc_out, dec_in, dec_out, inner,
                 out_relu=True, skip=False, pool="mean"):
        super().__init__()
        self.down = nn.Conv2d(enc_in, enc_out, 3, padding=1)
        self.inner = inner
        self.up = nn.Conv2d(dec_in, dec_out, 3, padding=1)
        self.out_relu = out_relu
        self.pool = pool
        self.skip = skip
        if skip and enc_in != dec_out:
            raise ValueError(f"SumSkip requires enc_in==dec_out, got {enc_in} vs {dec_out}")

    def forward(self, x):
        h = F.relu(self.down(x))
        hw = h.shape[-2:]                                     # remember the pre-pooling size, for cropping later
        if self.pool == "mean":
            h = F.avg_pool2d(h, 2, ceil_mode=True, count_include_pad=False)
        else:
            h = F.max_pool2d(h, 2, ceil_mode=True)
        h = self.inner(h)
        h = F.interpolate(h, scale_factor=2, mode="nearest")   # the reference implementation's default is :nearest
        h = h[..., : hw[0], : hw[1]]                           # = croppadding(x, odd)
        h = self.up(h)
        if self.out_relu:
            h = F.relu(h)
        return x + h if self.skip else h


def build_unet(n_in, n_out_raw, enc_internal=(32, 64, 96), skip_levels=None, pool="mean"):
    """Build a U-Net following `recmodel4`'s recursive structure.

    enc = [n_in, *enc_internal], dec = [n_out_raw, *enc_internal]. At level l:
    conv enc[l]->enc[l+1], pool, recurse, upsample, conv dec[l+1]->dec[l]. The
    innermost level is identity. The outermost level (l=1) output layer has no
    activation (the reference implementation's `f = l==1 ? identity : relu`) and
    no skip (n_in != n_out_raw).
    """
    enc = [n_in, *enc_internal]
    dec = [n_out_raw, *enc_internal]
    L = len(enc)                                    # levels 1..L-1 are built, level L is identity
    if skip_levels is None:
        skip_levels = range(2, L + 1)               # the reference default is 2:(len(enc_internal)+1)

    net: nn.Module = nn.Identity()
    for l in range(L - 1, 0, -1):                   # build from the inside out
        net = _Level(enc_in=enc[l - 1], enc_out=enc[l],
                     dec_in=dec[l], dec_out=dec[l - 1],
                     inner=net, out_relu=(l != 1),
                     skip=(l in skip_levels and l != 1), pool=pool)
    return net


class DINCAE(nn.Module):
    """DINCAE 2.0: a U-Net (+ optional refinement step).

    forward returns (mean, sigma^2) for **every level**, because the loss is
    computed for every level (2.0 Eq.4). Inference uses the last level. A
    refinement level's input is `cat(the previous level's raw output, the raw
    input)` (`model.jl:182-189`).
    """

    def __init__(self, n_in, n_var, enc_internal=(32, 64, 96),
                 loss_weights=(0.3, 0.7), pool="mean", gamma=GAMMA, mu=MU):
        super().__init__()
        self.n_var = n_var
        self.n_out_raw = 2 * n_var
        self.loss_weights = tuple(loss_weights)
        self.gamma, self.mu = gamma, mu
        nets = [build_unet(n_in, self.n_out_raw, enc_internal, pool=pool)]
        for _ in range(len(self.loss_weights) - 1):            # refinement level: input has the previous level's output added
            nets.append(build_unet(n_in + self.n_out_raw, self.n_out_raw,
                                   enc_internal, pool=pool))
        self.nets = nn.ModuleList(nets)

    def forward(self, xin):
        outs = []
        raw = self.nets[0](xin)
        outs.append(transform_msigma2(raw, self.gamma, self.mu))
        for net in self.nets[1:]:
            raw = net(torch.cat([raw, xin], dim=1))
            outs.append(transform_msigma2(raw, self.gamma, self.mu))
        return outs

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
