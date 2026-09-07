"""
losses.py — DINCAE's Gaussian negative log-likelihood (including the
truth_uncertain KL branch)
=====================================================================

Follows the reference implementation `reference/DINCAE.jl/src/model.jl:54-148`.
Paper 2.0 Eq.3:

    J = 1/(2N) * sum [ ((y - y_hat)/sigma_hat)^2 + log sigma_hat^2 + 2 log sqrt(2 pi) ]

The code drops the 1/2 and the constant term (they don't affect the gradient), and
adds two things the paper does not spell out:

  1. **The log term at unobserved cells is zeroed**:
     `sigma^2_masked = sigma^2_rec*mask + (1 - mask)`, so log(1)=0. This step is
     important -- otherwise the network could push sigma-hat down to the floor on
     cells with no target, just to shrink the log term.
  2. **The normalising N does not participate in the gradient**
     (`ChainRulesCore.ignore_derivatives`). N differs every minibatch (1.0
     sec.3.1 specifically stresses this).

**Multiple variables** (`model.jl:132-148`): each output variable is **computed
independently, normalised by its own N, then summed directly**. So the relative
weight between channels is entirely decided by the learned 1/sigma-hat^2, never
set by hand -- this is exactly what sets it apart from "hand-tuned weighting"
(which failed 8 times in this project, see the 4dvarnet_enkf record).

**The truth_uncertain branch** (`model.jl:68-86`, in neither paper): when the
"ground truth" itself carries an uncertainty sigma^2_true, the loss becomes the KL
divergence between two Gaussians:

    2*KL(p_true || q_rec) = log(sigma^2_rec/sigma^2_true) + (sigma^2_true + (m_rec - m_true)^2)/sigma^2_rec - 1

Useful to us because the gridded ground truth has its own error: velocity's
sigma^2_true = `vel_var / density` (in-cell velocity variance / headcount = the
standard error of the cell mean), and `vel_var` is exactly state's 4th channel.
"""
from __future__ import annotations

import torch


def dincae_cost_single(m_rec, s2_rec, m_true, s2_true, mask, truth_uncertain=False):
    """Cost for a single output variable. All tensors (B,1,H,W) or (B,H,W), mask is 0/1 float."""
    n = mask.sum().detach().clamp(min=1.0)          # N does not participate in the gradient
    if truth_uncertain:
        ratio = (s2_rec / s2_true) * mask + (1.0 - mask)        # unobserved = 1 -> log = 0
        d2 = ((m_rec - m_true) ** 2 + s2_true) * mask
        return (torch.log(ratio).sum() + (d2 / s2_rec).sum()) / n
    s2n = s2_rec * mask + (1.0 - mask)
    d2 = ((m_rec - m_true) ** 2) * mask
    return (torch.log(s2n).sum() + (d2 / s2_rec).sum()) / n


def decode_target(target, s2_floor=1e-6):
    """Decode the information-form target back into (m_true, sigma^2_true, mask).

    target (B, 2*nvar, H, W): even slices = a/sigma^2_true, odd slices = 1/sigma^2_true.
    mask = (1/sigma^2_true != 0) -- missing/undefined = zero precision (`model.jl:109-114`).
    """
    inv = target[:, 1::2]
    mask = (inv != 0).to(target.dtype)
    s2 = 1.0 / torch.clamp(inv, min=s2_floor)                   # this value is meaningless where masked out
    m = target[:, 0::2] * s2
    return m * mask, s2, mask


def dincae_loss(outs, target, loss_weights, truth_uncertain=False):
    """Total loss over every output level x every variable (2.0 Eq.4).

    outs : [(mean, sigma^2), ...] one per level, from DINCAE.forward
    returns (total, per_level) -- per_level is useful in logs to see whether the
    refinement step is doing anything
    """
    m_true, s2_true, mask = decode_target(target)
    per_level = []
    total = target.new_zeros(())
    for w, (m_rec, s2_rec) in zip(loss_weights, outs):
        lvl = target.new_zeros(())
        for c in range(m_rec.shape[1]):                          # independently normalised per variable, then summed
            lvl = lvl + dincae_cost_single(
                m_rec[:, c], s2_rec[:, c], m_true[:, c], s2_true[:, c], mask[:, c],
                truth_uncertain=truth_uncertain)
        per_level.append(lvl)
        total = total + w * lvl
    return total, per_level


@torch.no_grad()
def residual_mse(outs, target, per_channel=True):
    """Diagnostic: the last level's residual MSE on valid cells (same mask
    convention as the loss, but without the sigma-hat term).

    This is the human-facing metric -- the training objective is NLL, but MSE is
    what lines up in scale with 4dvarnet_enkf's numbers.
    """
    m_true, _, mask = decode_target(target)
    m_rec, _ = outs[-1]
    se = ((m_rec - m_true) ** 2) * mask
    if per_channel:
        n = mask.sum(dim=(0, 2, 3)).clamp(min=1.0)
        return (se.sum(dim=(0, 2, 3)) / n)
    return se.sum() / mask.sum().clamp(min=1.0)
