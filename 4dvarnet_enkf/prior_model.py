"""
prior_model.py  —  Baseline Component 2: the dynamical prior  Φ  (GENN)
=======================================================================

Role in the framework
----------------------
In the variational reconstruction (4DVarNet-style), the cost has two terms:

    J(x) = ‖(x − y) ⊙ Ω‖²            (observation term — fit the robot measurements)
         + ‖x − Φ(x)‖²               (dynamical prior — is the sequence plausible?)

Φ is NOT a forecaster. It is a REGULARISER: given a candidate full-state sequence
x, Φ(x)(t) says "what frame t should look like, judged from its neighbouring
frames". If x is dynamically consistent, x ≈ Φ(x) and the prior term is small.

This file implements Φ as a **GENN** (Gibbs-Energy Neural Network), following the
paper §3.2, with its defining property:

    ZERO-CENTRE: ψ(x)(s) does not depend on x(s) — the paper zeros the single
    CENTRE TAP of the conv kernels, so the output at a space-time point s=(t,i,j)
    never reads the input at exactly s (it does use s's neighbours). With a
    pointwise φ, Φ(x)(s) ⊥ x(s), so ‖x − Φ(x)‖² cannot collapse to the identity.

Architecture (paper §3.2)
-------------------------
    single branch:  Φ_b(x) = φ( ψ(x) )
      ψ : conv whose single CENTRE TAP is forced to zero  ⇒ ψ(x)(s) ⊥ x(s)
      φ : pointwise (1×1×1) conv stack + nonlinearity      ⇒ nonlinear transform
          Kernel size 1 everywhere, exactly as §3.2 specifies.
    TWO-SCALE (Eq.10, the paper's GENN experiments — Table 1/2 "two-scale GENN"):
      Φ(x) = Up( Φ₁( Dw(x) ) ) + Φ₂(x)
      Dw = average pooling, Up = ConvTranspose (both per the paper). Φ₁ runs AT THE
      COARSE RESOLUTION — that is the point of the multi-scale form: on the half-size
      grid Φ₁'s 3×3 kernel spans 6×6 original cells, so the two branches genuinely see
      different scales. Φ₂ sees x itself, not a high-pass residual.
    Each branch's ψ is zero-centre, so Φ still excludes the trivial identity
    Φ(x)=x (the paper's only stated requirement). two_scale is ON by default to
    match the paper.

    One honest caveat on the two-scale form: strict pixel-level independence
    Φ(x)(s) ⊥ x(s) holds EXACTLY only for a single branch (measured: exactly 0.0). With
    two scales, Dw pools x(s) into a coarse cell and Up spreads that cell back over a
    neighbourhood, so x(s) re-enters Φ(x)(s) through the coarse branch.

    Measured on the trained models, d|Φ(x)(s)|/dx(s) is 2e-2 to 4e-2 — i.e. 2-4% of what
    the identity Φ(x)=x would give (b0_k1 2.1e-2, b0_k4 4.2e-2, a2_k1 1.8e-2, a4_k1
    3.7e-2). The same architecture UNTRAINED leaks ~8e-4, so training increases the leak
    by roughly 25x: the optimiser does exploit this channel. The identity is still far
    from reachable, but "effectively excluded" is the honest phrasing, not "excluded".
    This is a property of Eq.10 itself, not of this implementation — the paper's own
    two-scale form has it too, and the paper does not quantify it.

    Reproduce with checks/check_paper_conformance.py; do not trust this number from
    memory, an earlier version of this docstring quoted ~3e-5, which came from an
    untrained model and was wrong by three orders of magnitude for the trained one.

State tensor convention
-----------------------
    x : (B, C, T, H, W)
        B = batch, C = physical channels (density, vx, vy, var = 4),
        T = time-window length (dT, in seconds), H×W = 36×12 grid.
    Φ(x) has the same shape. (4DVarNet for SSH used time-as-channel [B,dT,H,W];
    here we keep C and T as separate axes so the zero-centre is cleanly along T
    and the 4 physical channels are mixed by the convolutions.)

This module is pure PyTorch and has no dependency on the data pipeline; its only
contract is the (B,C,T,H,W) → (B,C,T,H,W) mapping above. Verification (shape,
zero-centre property, differentiability) is in `check_prior_model.py`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroCentreConv3d(nn.Module):
    """ψ: a 3D conv over (time, H, W) whose single CENTRE TAP is masked to zero.

    Faithful to the paper (§3.2): "the central values of all convolution kernels is
    set to zero such that ψ(x)(s) at position s does not depend on variable x(s)".
    So the output at a space-time point (t, i, j) does NOT use the input at exactly
    (t, i, j) [any channel], but DOES use its space-time neighbours (t±, i±, j±).
    Combined with a POINTWISE φ, this makes Φ(x)(s) independent of x(s), so the prior
    term ‖x − Φ(x)‖² cannot collapse to the trivial identity. The mask is re-applied
    to the weight on every forward, so the constraint survives optimiser updates.
    """

    def __init__(self, in_ch, out_ch, kt=3, kh=3, kw=3):
        # in_ch  : input channels  (for the first conv this is C = 4 physical channels)
        # out_ch : output channels (= `hidden`, the width of the hidden feature maps)
        # kt, kh, kw : space-time kernel; all ODD so there is a well-defined centre tap.
        super().__init__()
        assert kt % 2 == 1 and kh % 2 == 1 and kw % 2 == 1, "kernel sizes must be odd (need a centre tap)"
        self.conv = nn.Conv3d(in_ch, out_ch, (kt, kh, kw),
                              padding=(kt // 2, kh // 2, kw // 2), bias=True)
        # Mask that zeroes ONLY the single centre space-time tap (kt//2, kh//2, kw//2),
        # for every (out_ch, in_ch) pair. Shape (1,1,kt,kh,kw) broadcasts over channels.
        # => output at (t,i,j) does not read input at exactly (t,i,j) (paper's ψ(x)(s) ⊥ x(s)).
        mask = torch.ones(1, 1, kt, kh, kw)
        mask[:, :, kt // 2, kh // 2, kw // 2] = 0.0
        # register_buffer: moved with the module (.to/state_dict) but NOT trained.
        self.register_buffer("centre_tap_mask", mask)

    def forward(self, x):
        # x: (B, in_ch, T, H, W). Re-apply the mask to the weight every forward so the
        # "ψ(x)(s) ⊥ x(s)" constraint holds even after the optimiser nudges the weights.
        w = self.conv.weight * self.centre_tap_mask      # (out_ch, in_ch, kt, kh, kw)
        return F.conv3d(x, w, self.conv.bias, padding=self.conv.padding)  # -> (B, out_ch, T, H, W)


class _GENNBranch(nn.Module):
    """Single-scale GENN branch: ψ (zero-centre) -> φ (pointwise 1×1×1 convs + nonlinearity)."""

    def __init__(self, n_channels=4, hidden=32, kt=3, kh=3, kw=3, n_phi_layers=2):
        super().__init__()
        # ψ: lift the C=4 physical channels to `hidden` feature channels while mixing
        # neighbouring frames (zero-centre) and 3x3 spatial neighbours.
        self.psi = ZeroCentreConv3d(n_channels, hidden, kt, kh, kw)
        # φ: a stack of 1x1x1 convs. Kernel size 1 means POINTWISE — it mixes the `hidden`
        # channels at each (t,h,w) location independently, adding nonlinearity WITHOUT
        # touching time or space again (so it cannot re-introduce a dependence on frame t).
        phi = []
        for _ in range(n_phi_layers - 1):                 # (n_phi_layers - 1) hidden->hidden layers
            phi += [nn.Conv3d(hidden, hidden, 1), nn.ReLU(inplace=True)]
        phi += [nn.Conv3d(hidden, n_channels, 1)]         # final layer: -> C physical channels
        self.phi = nn.Sequential(*phi)

    def forward(self, x):
        # x (B,C,T,H,W) --ψ--> (B,hidden,T,H,W) --ReLU--> --φ--> (B,C,T,H,W)
        return self.phi(F.relu(self.psi(x)))


class GENN(nn.Module):
    """The dynamical prior Φ.  Φ(x): (B,C,T,H,W) -> (B,C,T,H,W).

    Args:
        n_channels : number of physical channels. ATC = 4 (density, vx, vy, vel_var).
                     Input and output both carry these C channels.
        hidden     : width of the hidden feature maps between ψ and φ (default 32).
                     Bigger = more capacity + more parameters; not the physical channels.
        kt         : TIME kernel of ψ — how many frames it spans (3 -> uses t-1, t+1;
                     must be ODD). Larger kt lets Φ use frames further away in time.
        kh, kw     : SPATIAL kernel of ψ (3x3 neighbourhood in H, W).
        n_phi_layers : number of pointwise (1x1x1) conv layers in φ (default 2).
                     More layers = a deeper per-location nonlinear transform.
        two_scale  : if True, use the two-scale form (paper Eq.10): a COARSE branch for
                     global/large-scale dynamics + a FINE branch for local detail
                     (their outputs are summed). If False, a single branch is used.
        scale      : spatial downsampling factor of the coarse branch (default 2 ->
                     pool 36x12 down by 2x before the coarse conv, then upsample back).
    """

    def __init__(self, n_channels=4, hidden=32, kt=3, kh=3, kw=3,
                 n_phi_layers=2, two_scale=True, scale=2):
        super().__init__()
        self.two_scale = two_scale
        self.scale = scale
        self.branch_fine = _GENNBranch(n_channels, hidden, kt, kh, kw, n_phi_layers)
        if two_scale:
            self.branch_coarse = _GENNBranch(n_channels, hidden, kt, kh, kw, n_phi_layers)
            # Up in Eq.10 is a ConvTranspose layer — a LEARNED upsampling, not fixed
            # interpolation, so the coarse branch can shape how its output is spread back
            # over the fine grid. stride=kernel=scale gives an exact 1->scale expansion.
            self.up = nn.ConvTranspose2d(n_channels, n_channels, kernel_size=scale, stride=scale)

    def forward(self, x):
        # x: (B, C, T, H, W) -> Φ(x): same shape
        if not self.two_scale:
            return self.branch_fine(x)                    # single-branch case
        # Paper Eq.10:  Φ(x) = Up( Φ₁( Dw(x) ) ) + Φ₂(x)
        # Φ₁ runs at the COARSE resolution (that is what makes this multi-scale: on the
        # half-size grid its 3×3 kernel covers 6×6 original cells), and Φ₂ sees x itself.
        B, C, T, H, W = x.shape
        # Dw: average pooling over H,W only. avg_pool2d takes 4D, so fold T into the batch
        # axis — the T axis is never touched, keeping the zero-centre property along time.
        xr = x.reshape(B * T, C, H, W)
        coarse = F.avg_pool2d(xr, kernel_size=self.scale, ceil_mode=True)   # (B*T, C, Hc, Wc)
        Hc, Wc = coarse.shape[-2:]
        coarse = self.branch_coarse(coarse.reshape(B, C, T, Hc, Wc))        # Φ₁ at coarse res
        # Up: ConvTranspose back to the fine grid; crop in case ceil_mode padded the pooling.
        up = self.up(coarse.reshape(B * T, C, Hc, Wc))[..., :H, :W]
        return up.reshape(B, C, T, H, W) + self.branch_fine(x)              # + Φ₂(x)


def prior_residual(x, phi):
    """Prior residual r(t) = x(t) − Φ(x)(t), i.e. the summand of ‖x − Φ(x)‖² in the variational cost."""
    return x - phi(x)
