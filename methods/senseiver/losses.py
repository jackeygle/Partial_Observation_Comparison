"""
losses.py — training loss and diagnostic breakdown
================================

**The training loss is copied from the reference implementation as-is**:
`network_light.py:68`'s `F.mse_loss(pred_values, field_values, reduction='sum')`,
unweighted 4-channel squared error on the raw field. No per-channel/per-cell
weighting is added -- that would no longer be the paper's method.

`reduction='sum'` follows the reference implementation. It differs from `'mean'`
only by a constant factor, which under Adam is essentially absorbed by the
per-parameter second-moment normalisation, so the two are equivalent; `'sum'` is
kept to align verbatim with the reference implementation. What's logged is the
value divided by the element count (the reference implementation logs it the
same way).

The breakdown functions below are for **diagnostics only** and never enter the gradient.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def senseiver_loss(pred, target):
    """The paper's/reference implementation's training loss. pred/target: (B, Nq, C) or (B, C, H, W)."""
    return F.mse_loss(pred, target, reduction="sum")


@torch.no_grad()
def diagnostics(pred, target, obs_mask, channels):
    """MSE breakdown: per channel / observed region vs. blind region.

    pred/target (B,C,H,W), obs_mask (B,H,W) bool (True = that cell was observed).
    Also reports density MSE on "occupied" cells (density>0): a single MSE gets
    diluted by a large number of empty cells, hiding whether the model is just
    outputting a smooth field near zero.
    """
    se = (pred - target) ** 2
    obs = obs_mask[:, None].expand_as(se)
    out = {"mse": float(se.mean()),
           "mse_blind": float(se[~obs].mean()) if (~obs).any() else float("nan"),
           "mse_obs": float(se[obs].mean()) if obs.any() else float("nan")}
    for i, c in enumerate(channels):
        out[f"mse_{c}"] = float(se[:, i].mean())
    occ = target[:, 0] > 0
    out["mse_density_occ"] = float(se[:, 0][occ].mean()) if occ.any() else float("nan")
    out["occ_frac"] = float(occ.float().mean())
    return out
