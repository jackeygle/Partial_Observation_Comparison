"""
calibrate_decay.py — calibrate the tables needed for an age-dependent sigma^2
===================================================

DINCAE's input is `y/sigma^2` and `1/sigma^2`, missing = both are 0 (1.0 sec.3 /
2.0 sec.2.3). Our observations have one more dimension: **how old they are**.
Diagnostics show the median age of "most recent observation" on unobserved cells
is 9 s, while the field decorrelates in 2 s (see README), so naive carry-forward
would systematically feed in stale data.

The correct approach is to shrink an observation from `age` seconds ago toward
the per-cell mean field, weighted by its correlation (BLUE):

    y_hat(age)      = stats + AC(age) * (y_old - stats)
    sigma^2_eff(age) = AC(age)^2 * sigma^2_obs + (1 - AC(age)^2) * Var_resid

At age=0 this degenerates to sigma^2_obs; as age -> infinity, AC -> 0 and
sigma^2_eff -> Var_resid, so the input automatically becomes "uninformative".
This script produces everything needed to calibrate this formula:

  1. **The per-cell mean field** stats[c, cell, bin] -- per-cell x
     time-of-day-bin mean (computed on training days). The bin width is chosen
     by **held-out-day MSE**, not guessed.
  2. **AC_c(lag)** -- the temporal autocorrelation of the residual (after
     subtracting the per-cell mean field), per channel
  3. **Var_resid[c, cell]** -- per-cell residual variance (= the ceiling of
     sigma^2_eff as AC -> 0)
  4. **P_occ(lag)** -- P(occupied at t+lag | occupied at t). Velocity
     information decays for two reasons: the flow changed, and the cell emptied
     out. AC only accounts for the first; this table accounts for the second.

**Per-channel validity** (`channel_valid`) -- where each of the 4 channels "is
defined" differs, and every statistic is computed only where defined, otherwise
the placeholder 0 would pollute the statistics:

  density : defined everywhere (density=0 is a genuine measurement: "nobody is here")
  vx, vy  : `density > 0` -- velocity on an empty cell is a placeholder
            (`h5_to_grid.py`: `vel = where(density>0, vel/density, 0)`)
  var     : `vel_var > 0` <=> at least 2 people in the cell
            (`h5_to_grid.py:128`: `vel_var[nnz <= 1] = 0`, variance of 1 point is undefined)

Output: decay_tables.npz + decay_tables.json (a human-readable summary)

Usage (pure numpy/scipy, the login node is fine):
    python3 calibrate_decay.py                  # all 32 training days
    python3 calibrate_decay.py --days 3         # smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np

# Scripts in checks/ import source modules from the project root; the root must
# be inserted at the front (so this directory's losses.py takes priority),
# 4dvarnet_enkf can only be appended (it also has a losses.py, and inserting it
# at the front would shadow this directory's).
from crowdcore import config                                                    # noqa: E402
from crowdcore import navigation as nav                                         # noqa: E402
from crowdcore import observation_model as om                                    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ATC is in Osaka (UTC+9). time is unix seconds; time-of-day must use local time,
# otherwise binning would cut across day boundaries.
TZ_OFFSET = 9 * 3600
BASE_BIN = 300                       # accumulation resolution (s); candidate bin widths must be integer multiples of it
#   86400 = a single bin = degenerates to "per-cell all-time mean" (no
#   time-of-day binning), used as a reference lower bound.
#   Candidates must leave headroom on both ends, otherwise a "data-driven"
#   choice is just hitting the edge of the search range.
BIN_CANDIDATES = (300, 900, 1800, 3600, 7200, 14400, 86400)
CHANNELS = ("density", "vx", "vy", "var")
NCH = len(CHANNELS)

# Dense then geometrically increasing: p90(age)=35 s, tail extends to 512 s
LAGS = (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)


def tod_bin(t: np.ndarray) -> np.ndarray:
    """unix seconds -> time-of-day bin index at BASE_BIN resolution."""
    return (((t.astype(np.int64) + TZ_OFFSET) % 86400) // BASE_BIN).astype(np.int64)


def channel_valid(X):
    """(NCH, T, H, W) bool -- where each channel **is defined** (frame, cell). See the module docstring."""
    occ = X[:, 0] > 0
    return np.stack([np.ones_like(occ), occ, occ, X[:, 3] > 0])


def load_day(fp):
    with h5py.File(fp, "r") as f:
        return f["grid"][:], f["time"][:]


# --------------------------------------------------------------------------- #
# Pass A: accumulate the per-cell mean field (at BASE_BIN resolution, later pooled into wider bins)
# --------------------------------------------------------------------------- #
def accumulate_mean_field(files, valid):
    nbin = 86400 // BASE_BIN
    H, W = valid.shape
    s = np.zeros((NCH, H, W, nbin))          # sum of values (only where defined)
    n = np.zeros((NCH, H, W, nbin))          # number of defined samples

    for k, fp in enumerate(files):
        X, t = load_day(fp)
        b = tod_bin(t)
        cv = channel_valid(X)                                    # (NCH,T,H,W)
        for bi in np.unique(b):
            sel = b == bi
            for c in range(NCH):
                v = cv[c][sel]
                s[c, :, :, bi] += np.where(v, X[sel, c], 0.0).sum(0)
                n[c, :, :, bi] += v.sum(0)
        print(f"  [stats] {k + 1}/{len(files)} {os.path.basename(fp)}", flush=True)
    return s, n


def pool_mean_field(s, n, bin_width):
    """Pools to bin_width. A bin with zero samples falls back to that cell's all-time mean; if there are no samples at all, 0."""
    assert bin_width % BASE_BIN == 0
    g = bin_width // BASE_BIN
    pool = lambda a: a.reshape(*a.shape[:-1], -1, g).sum(-1)
    sp, np_ = pool(s), pool(n)

    stats = np.zeros_like(sp)
    tot_s, tot_n = sp.sum(-1, keepdims=True), np_.sum(-1, keepdims=True)
    fb = np.where(tot_n > 0, tot_s / np.maximum(tot_n, 1), 0.0)     # that cell's all-time mean
    stats = np.where(np_ > 0, sp / np.maximum(np_, 1), fb)
    return stats


def score_mean_field(files, valid, clims):
    """Per-cell mean field MSE on held-out days, per channel, computed only on **defined** (frame, cell) pairs.

    `clims` = {bin_width: stats}. Each dev day is **read only once**, scoring all
    candidate bin widths at the same time (otherwise 7 candidates x 7 days = 49
    reads of 280 MB each).
    """
    se = {bw: np.zeros(NCH) for bw in clims}
    cnt = {bw: np.zeros(NCH) for bw in clims}
    for fp in files:
        X, t = load_day(fp)
        tb = tod_bin(t)
        cv = channel_valid(X)[:, :, valid]                        # (NCH,T,N)
        Xv = np.stack([X[:, c][:, valid] for c in range(NCH)])     # (NCH,T,N)
        for bw, stats in clims.items():
            b = tb // (bw // BASE_BIN)
            pred = np.moveaxis(stats[:, :, :, b], -1, 1)[:, :, valid]
            for c in range(NCH):
                d = (Xv[c] - pred[c])[cv[c]]
                se[bw][c] += (d ** 2).sum(); cnt[bw][c] += d.size
        print(f"  [score] {os.path.basename(fp)}", flush=True)
    return {bw: se[bw] / np.maximum(cnt[bw], 1) for bw in clims}


# --------------------------------------------------------------------------- #
# Pass B: residual temporal autocorrelation + occupancy persistence probability
# --------------------------------------------------------------------------- #
def accumulate_autocorr(files, valid, stats, bin_width):
    g = bin_width // BASE_BIN
    nL = len(LAGS)
    cross = np.zeros((NCH, nL))       # sum a_t * a_{t+L}
    e_lo = np.zeros((NCH, nL))        # sum a_t^2      (same pairing)
    e_hi = np.zeros((NCH, nL))        # sum a_{t+L}^2
    cnt = np.zeros((NCH, nL))
    occ_pair = np.zeros(nL)
    occ_lo = np.zeros(nL)
    N = int(valid.sum())
    var_cell = np.zeros((NCH, N))
    var_n = np.zeros((NCH, N))

    for k, fp in enumerate(files):
        X, t = load_day(fp)
        b = tod_bin(t) // g
        pred = np.moveaxis(stats[:, :, :, b], -1, 1)[:, :, valid]   # (NCH,T,N)
        cv = channel_valid(X)[:, :, valid]                         # (NCH,T,N)
        occ = cv[1]                                                # density>0
        a = np.stack([X[:, c][:, valid] - pred[c] for c in range(NCH)])

        for c in range(NCH):
            var_cell[c] += np.where(cv[c], a[c] ** 2, 0.0).sum(0)
            var_n[c] += cv[c].sum(0)

        for li, L in enumerate(LAGS):
            lo = slice(None, None) if L == 0 else slice(None, -L)
            hi = slice(None, None) if L == 0 else slice(L, None)
            occ_pair[li] += (occ[lo] & occ[hi]).sum(); occ_lo[li] += occ[lo].sum()
            for c in range(NCH):
                m = cv[c][lo] & cv[c][hi]                          # both ends defined
                x, y = a[c][lo], a[c][hi]
                cross[c, li] += (x * y * m).sum()
                e_lo[c, li] += (x * x * m).sum()
                e_hi[c, li] += (y * y * m).sum()
                cnt[c, li] += m.sum()
        print(f"  [ac] {k + 1}/{len(files)} {os.path.basename(fp)}", flush=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        ac = cross / np.sqrt(np.maximum(e_lo * e_hi, 1e-30))
        p_occ = occ_pair / np.maximum(occ_lo, 1)
        var_cell = var_cell / np.maximum(var_n, 1)
    return np.nan_to_num(ac), p_occ, var_cell, cnt


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0, help="use only the first N training days (smoke test)")
    ap.add_argument("--out", default=os.path.join(ROOT, "decay_tables"))
    args = ap.parse_args()

    train = om.split_files("train")
    dev_days = om.split_files("valid")
    if args.days:
        train, dev_days = train[: args.days], dev_days[:1]
    print(f"train days {len(train)}, dev days {len(dev_days)}, channels {CHANNELS}")

    valid = nav.build_valid_mask_from_config()               # consistent across days (union of the training set's visited cells)
    print(f"walkable cells {valid.sum()}/{valid.size}")

    print("Pass A: state_stats")
    s, n = accumulate_mean_field(train, valid)

    print("\nBin width selection (dev-day MSE, lower is better):")
    clims = {bw: pool_mean_field(s, n, bw) for bw in BIN_CANDIDATES}
    scores = score_mean_field(dev_days, valid, clims)
    for bw in BIN_CANDIDATES:
        tag = "  (= no time-of-day binning)" if bw == 86400 else ""
        print("  bin %6ds : " % bw
              + "  ".join("%s %.5f" % (c, scores[bw][i]) for i, c in enumerate(CHANNELS))
              + tag)
    best = min(BIN_CANDIDATES, key=lambda bw: scores[bw][0])
    if best in (BIN_CANDIDATES[0], BIN_CANDIDATES[-1]):
        print(f"  ! the optimum falls on the edge of the candidate range ({best}s) -- widen BIN_CANDIDATES and reselect")
    print(f"  -> selected bin_width = {best} s  (by density's dev MSE)")
    stats = clims[best]

    print("\nPass B: residual autocorrelation")
    ac, p_occ, var_cell, cnt = accumulate_autocorr(train, valid, stats, best)

    obs_std = np.asarray(config.get("observation", "obs_std"), dtype=float)

    np.savez_compressed(
        args.out + ".npz",
        lags=np.asarray(LAGS), ac=ac, p_occ=p_occ,
        var_cell=var_cell, stats=stats, bin_width=best,
        valid_mask=valid, obs_std=obs_std, channels=np.asarray(CHANNELS),
        n_train_days=len(train),
    )
    with open(args.out + ".json", "w") as f:
        json.dump({
            "n_train_days": len(train), "bin_width_s": int(best),
            "channels": list(CHANNELS),
            "bin_width_dev_mse": {str(k): v.tolist() for k, v in scores.items()},
            "walkable_cells": int(valid.sum()), "lags": list(LAGS),
            "ac": {c: ac[i].tolist() for i, c in enumerate(CHANNELS)},
            "p_occ_persist": p_occ.tolist(),
            "var_resid_mean": {c: float(var_cell[i].mean()) for i, c in enumerate(CHANNELS)},
            "obs_std": obs_std.tolist(), "obs_var": (obs_std ** 2).tolist(),
        }, f, indent=2)

    print("\n=== AC(lag) ===")
    print("lag(s)   " + " ".join("%6d" % L for L in LAGS))
    for i, c in enumerate(CHANNELS):
        print("%-8s " % c + " ".join("%6.3f" % v for v in ac[i]))
    print("P(occ|occ)" + " ".join("%6.3f" % v for v in p_occ))
    print("\nVar_resid (mean over cells): "
          + "  ".join("%s %.5f" % (c, var_cell[i].mean()) for i, c in enumerate(CHANNELS)))
    print("sigma^2_obs                : "
          + "  ".join("%s %.5f" % (c, (obs_std ** 2)[i]) for i, c in enumerate(CHANNELS)))
    print(f"\nwrote {args.out}.npz / .json")


if __name__ == "__main__":
    main()
