"""
state.py — state definition: the four channels, validity rules, per-channel
transforms, per-cell statistics
=====================================================

Four things live here: `CHANNELS`/`NCH` (the four channels), `channel_valid`
(per-channel validity rule), `CHANNEL_TRANSFORM`/`fwd_channel`/`inv_channel`
(per-channel transforms), `StateStats` (per-cell statistics).

**The per-cell mean field -- in one sentence: each cell's average over the 32
training days.** (The paper calls this term climatology / `remove_mean`, ocean/
meteorology jargon; this directory always calls it "the per-cell mean field", see
the README's "Terminology" section for the correspondence.) Measured: "typical
foot traffic" differs by dozens of times between different corridor locations
(the busiest cell averages 0.26 people/cell, corners are near 0), and this
difference is stable across all 92 days.

DINCAE subtracts it before reconstruction, working in the residual space (1.0
sec.3's `remove_mean`; 2.0 sec.2.3 does the same for the auxiliary variables too).
The reference implementation, `reference/DINCAE.jl/src/data.jl:266-276`:

    meandata = sum(non-NaN values, dims=4) ./ sum(non-NaN, dims=4)

Note this is **one mean taken over the entire time axis**, one number per pixel
per variable -- **not binned by season/time-of-day**. Periodicity is handled by
the `cos/sin` input channels, not by this mean table. We follow the same
approach: one mean per cell, the diurnal cycle handled by the
`cos/sin(time-of-day)` input channels.

**Why it is worth subtracting**: the network then only has to judge "more or less
than usual right now," rather than also memorising "how many people are usually
here" (the latter can be read exactly from a lookup table, no need to learn it).
More importantly, it gives blind cells a fallback -- 36.9% of cells are never
observed at all within {t-1,t,t+1}, and after subtracting the mean, an output of 0
means "same as usual," not "nobody at all." Measured: using this mean table alone
as the prediction already beats carry-forward filling by 30-39%.

**Per-channel validity** (`channel_valid`) -- the mean is computed only over
(frame, cell) pairs where that channel **is defined**, otherwise the placeholder
0 would pollute the mean. This is not a deviation from the paper -- it is the
paper's own principle, "missing = zero precision," applied to our data:

  density : defined everywhere (density=0 is a genuine measurement: "nobody is here")
  vx, vy  : `density > 0` -- velocity on an empty cell is a placeholder
            (`h5_to_grid.py`: `vel = where(density>0, vel/density, 0)`)
  var     : `vel_var > 0` <=> at least 2 people in the cell
            (`h5_to_grid.py:128`: `vel_var[nnz <= 1] = 0`, variance of 1 point is undefined)

Output: artifacts/state_stats.npz -- mean[NCH,H,W], std[NCH], count, valid_mask

Usage (pure numpy/scipy, the login node is fine):
    python3 state.py                # all 32 training days
    python3 state.py --days 3       # smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np

# append rather than insert(0): 4dvarnet_enkf also has a losses.py, inserting at
# the front would shadow this directory's same-named module
from crowdcore import navigation as nav                                         # noqa: E402
from crowdcore import observation_model as om                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CHANNELS = ("density", "vx", "vy", "var")
NCH = len(CHANNELS)


def channel_valid(X):
    """(NCH, T, H, W) bool -- where each channel **is defined** (frame, cell).
    See the module docstring.

    Judged on **raw physical values** (independent of CHANNEL_TRANSFORM). Note
    log1p is monotonic, so `var > 0` and `log1p(var) > 0` are equivalent
    conditions; raw values are used here only for readability.
    """
    occ = X[:, 0] > 0
    return np.stack([np.ones_like(occ), occ, occ, X[:, 3] > 0])


# --------------------------------------------------------------------------- #
# Per-channel transforms
# --------------------------------------------------------------------------- #
#: `var` goes through log1p. 1.0's conclusion section: the method "can easily be
#: extended to parameterised probability distributions, in particular a
#: log-normal for concentration-like variables." `var` (in-cell velocity
#: variance) is exactly that kind -- non-negative and heavy-tailed: its residual
#: std is only 0.23, yet crowded cells can have `vel_var` exceeding 2, so after
#: normalising there are still plenty of 10-sigma samples, with a handful of
#: extreme points dominating the gradient. Measured: without the transform,
#: `var`'s normalised dev MSE oscillates between 8 and 62 (1.0 = the level of
#: "just output the per-cell mean field"), i.e. the reconstruction is an order
#: of magnitude worse than doing nothing.
CHANNEL_TRANSFORM = (None, None, None, "log1p")


def fwd_channel(v, c):
    """Raw physical value -> the space the model works in (per channel)."""
    if CHANNEL_TRANSFORM[c] == "log1p":
        return np.log1p(np.maximum(v, 0.0))       # var is a variance; negative values can only come from observation noise
    return v


#: Cap on the exponent for `exp(mu + sigma^2/2)`. See inv_channel's note.
_INV_EXP_CAP = 20.0


def inv_channel(v, c, var_in_space=None):
    """Model space -> raw physical value (the inverse of `fwd_channel`).

    By default returns the **median**, `expm1(mu)`.

    When `var_in_space` is given, switches to the **log-normal mean**,
    `exp(mu + sigma^2/2) - 1` (if x ~ N(mu,sigma^2) and y = expm1(x), this is
    E[y]). **But do not use it as the default point estimate**: sigma-hat^2 is
    clamped by Eq.6 at 1/mu = 1000, and multiplying back by std^2 can push
    `0.5*sigma^2` up to 11, blowing the exponential up -- measured: during one
    smoke test, `var`'s physical-space MSE reached 10^22 because of this. It is
    mathematically correct, but a handful of "I don't know" cells would then
    dominate the MSE with astronomical numbers, so for a channel that is
    **heavy-tailed with an uncalibrated sigma-hat**, the median is the more
    usable point estimate, and `var`'s error should mainly be judged in the
    transformed (log) space. An extra `_INV_EXP_CAP` floor is applied to the
    exponent here to keep it from overflowing to inf.
    """
    if CHANNEL_TRANSFORM[c] == "log1p":
        m = v if var_in_space is None else v + 0.5 * var_in_space
        return np.expm1(np.minimum(m, _INV_EXP_CAP))
    return v


class StateStats:
    """Reads state_stats.npz.

    `mean[c,H,W]` is the per-cell time mean; `std[c]` is that channel's residual
    standard deviation (a scalar, computed over walkable ∩ defined (frame, cell)
    pairs).

    **Why std is needed**: the paper uses `obs_err_std = 1` for every variable
    (the code default), which is fine on SST -- its residual scale is naturally
    O(1) degC. Our four channels' residual variances differ by three orders of
    magnitude (density 0.036 / vx 0.53 / vy 0.13 / var 0.037), and treating them
    all as sigma^2=1 unbalances the Gaussian NLL: the `log sigma-hat^2` term has
    no lower bound, so a channel whose residual scale is much smaller than 1 can
    push sigma-hat^2 straight to Eq.6's floor and earn a chunk of negative loss
    for free, without the reconstruction actually improving (measured: train NLL
    drops by 5 units while density's dev MSE actually rises from 0.0254 to
    0.0419). Normalising each channel's residual to unit variance makes
    sigma^2_obs=1 match the data's scale, exactly what the reference code's
    `normalize2` (`data.jl:111-120`) does.
    """

    def __init__(self, path=os.path.join(HERE, "artifacts", "state_stats.npz")):
        z = np.load(path, allow_pickle=False)
        self.mean = z["mean"].astype(np.float32)          # (NCH,H,W)
        self.std = z["std"].astype(np.float32)            # (NCH,)  residual standard deviation
        self.count = z["count"]
        self.valid = z["valid_mask"].astype(bool)         # (H,W) walkable
        self.n_train_days = int(z["n_train_days"])
        assert self.mean.shape[0] == NCH and self.std.shape == (NCH,)
        assert (self.std > 0).all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0, help="use only the first N training days (smoke test)")
    ap.add_argument("--out", default=os.path.join(HERE, "artifacts", "state_stats"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    train = om.split_files("train")
    if args.days:
        train = train[: args.days]
    valid = nav.build_valid_mask_from_config()        # consistent across days (union of the training set's visited cells)
    H, W = valid.shape
    print(f"train days {len(train)}, walkable {valid.sum()}/{valid.size}, channels {CHANNELS}")

    s = np.zeros((NCH, H, W))       # sum x
    sq = np.zeros((NCH, H, W))      # sum x^2   (for the residual variance)
    n = np.zeros((NCH, H, W))       # number of defined samples
    for k, fp in enumerate(train):
        with h5py.File(fp, "r") as f:
            X = f["grid"][:]
        cv = channel_valid(X)
        for c in range(NCH):
            # Both the per-cell mean field and the residual std are computed in
            # the **transformed** space, matching the encoding/loss convention
            v = np.where(cv[c], fwd_channel(X[:, c].astype(np.float64), c), 0.0)
            s[c] += v.sum(0)
            sq[c] += (v ** 2).sum(0)
            n[c] += cv[c].sum(0)
        print(f"  {k + 1}/{len(train)} {os.path.basename(fp)}", flush=True)

    mean = np.where(n > 0, s / np.maximum(n, 1), 0.0)
    # Residual variance (aggregated over walkable cells): sum_frames (x - mean_cell)^2 = sum(x^2) - n*mean^2
    # so one pass over the data gives it exactly, no need for a second scan.
    ss = np.where(valid[None], sq - n * mean ** 2, 0.0).sum(axis=(1, 2))
    nn = np.where(valid[None], n, 0.0).sum(axis=(1, 2))
    var_resid = ss / np.maximum(nn, 1)
    std = np.sqrt(np.maximum(var_resid, 1e-12))

    np.savez_compressed(args.out + ".npz", mean=mean, std=std, count=n,
                        valid_mask=valid, n_train_days=len(train),
                        channels=np.asarray(CHANNELS))
    with open(args.out + ".json", "w") as f:
        json.dump({"n_train_days": len(train), "channels": list(CHANNELS),
                   "walkable_cells": int(valid.sum()),
                   "mean_over_walkable": {c: float(mean[i][valid].mean())
                                          for i, c in enumerate(CHANNELS)},
                   "resid_std": {c: float(std[i]) for i, c in enumerate(CHANNELS)},
                   "resid_var": {c: float(var_resid[i]) for i, c in enumerate(CHANNELS)},
                   "samples_per_cell_median": {c: float(np.median(n[i][valid]))
                                               for i, c in enumerate(CHANNELS)}}, f, indent=2)

    print("\nPer-cell mean + residual std (for normalisation):")
    for i, c in enumerate(CHANNELS):
        print("  %-8s mean %9.5f   resid_std %8.5f   median samples/cell %.0f"
              % (c, mean[i][valid].mean(), std[i], np.median(n[i][valid])))
    print(f"\nwrote {args.out}.npz / .json")


if __name__ == "__main__":
    main()
