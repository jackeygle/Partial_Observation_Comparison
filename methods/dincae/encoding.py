"""
encoding.py — DINCAE's information-form input/target encoding (strictly per the paper)
====================================================================

DINCAE does not "fill in" missing values. The input is two slices: `y/sigma^2` and
`1/sigma^2`, missing = both are 0 = zero precision (1.0 sec.3; 2.0 sec.2.3;
reference implementation `reference/DINCAE.jl/src/data.jl:311-325`). Input and
target both live in the same "information space" coordinate frame, so **there is
no fill-in step**.

**sigma^2_obs is taken as the constant 1**, the paper/code's default
`obs_err_std = 1`. 1.0 sec.3's own words: the exact value of this constant does
not matter, because it gets absorbed by the first layer's weight matrix, which is
trained. Taking it as 1 has the benefit that both slices are naturally O(1),
needing no extra scale handling. So:

    observed  ->  scaled = (fwd(y) - mean)/std ,  invvar = 1
    missing   ->  scaled = 0                   ,  invvar = 0

**Only the 3 neighbouring frames are used in time** (`ntime_win = 3`, 1.0 sec.3's
"previous day / today / next day"). If a cell is not observed in any of
{t-1, t, t+1}, it is missing -- there is no "fall back to an older observation" in
the paper.

**Target**: the paper's target is the observation itself, with its missing values
(they have no ground truth). **We have complete ground truth, and this is the
only place that cannot be copied as-is** -- we use the ground truth as the target,
with the mask taken from "is that channel defined here" (`channel_valid`).
Treating "undefined" as missing, with zero precision, is exactly the paper's own
mechanism.
"""
from __future__ import annotations

import numpy as np

from methods.dincae.state import (CHANNELS, NCH, StateStats, channel_valid,  # noqa: F401
                         fwd_channel)

FRESH_OFFSETS = (-1, 0, 1)          # ntime_win = 3 (the paper's default, must be odd)
CYCLE_PERIODS = (86400.0, 604800.0)  # diurnal + weekly period (seconds). The paper
#                                     uses (365.25 d); the reference code's
#                                     `cycle_periods` is already a list -- ATC's
#                                     periodicity is within a day and a week, not
#                                     within a year.

N_STATIC = 2 + 2 * len(CYCLE_PERIODS)                    # row, col + cos/sin x number of periods
N_IN = N_STATIC + 2 * NCH * len(FRESH_OFFSETS)           # = 6 + 24 = 30
N_TGT = 2 * NCH                                          # one (a/sigma^2_true, 1/sigma^2_true) pair per channel


def observed_pair(Y, M, mean, std):
    """Observation -> the two information-form slices (sigma^2_obs = 1, in the
    **normalised residual space**).

    Input : Y (T,NCH,H,W) observed values (0 where unobserved); M (T,NCH,H,W) bool observation mask
            mean (NCH,H,W) per-cell time mean; std (NCH,) that channel's residual std
    Output: scaled (T,NCH,H,W) = ((y - mean)/std)*M ; invvar (T,NCH,H,W) = M

    Reason for dividing by std: see `state.StateStats`'s docstring -- the paper
    uses `obs_err_std = 1` for every variable, and our four channels' residual
    variances differ by three orders of magnitude; without normalising, the
    Gaussian NLL's `log sigma-hat^2` term would earn free negative loss on a
    small-scale channel. After normalising, sigma^2_obs=1 matches the data's scale.
    """
    m = M.astype(np.float32)
    scaled = np.empty(Y.shape, dtype=np.float32)
    for c in range(NCH):                              # per channel: var goes through log1p, see CHANNEL_TRANSFORM
        a = (fwd_channel(Y[:, c].astype(np.float64), c) - mean[c][None]) / std[c]
        scaled[:, c] = (a * m[:, c]).astype(np.float32)
    return scaled, m


def static_channels(t_unix, H, W):
    """Coordinates + cyclic time. The lat/long and cos/sin(day-of-year) items from
    the paper's input list (1.0 sec.3; code `data.jl:409-419`). Output: (T, N_STATIC, H, W)"""
    T = len(t_unix)
    out = np.empty((T, N_STATIC, H, W), dtype=np.float32)
    out[:, 0] = np.linspace(-1, 1, H, dtype=np.float32)[:, None]
    out[:, 1] = np.linspace(-1, 1, W, dtype=np.float32)[None, :]
    for k, period in enumerate(CYCLE_PERIODS):
        ph = 2 * np.pi * (np.asarray(t_unix, dtype=np.float64) / period)
        out[:, 2 + 2 * k] = np.cos(ph).astype(np.float32)[:, None, None]
        out[:, 3 + 2 * k] = np.sin(ph).astype(np.float32)[:, None, None]
    return out


def encode_target(X, mean, std, walkable, full_field=False):
    """The target also uses information form, with the mask decided by whether
    the second slice is 0 (`model.jl:109-114`).

    sigma^2_true = 1 (the paper has no `truth_uncertain` branch, it only exists in
    the code), so:
        slice 1 = normalised residual * mask, slice 2 = mask
    mask = `channel_valid` ∩ walkable, i.e. "that channel is defined here".
    The normalisation convention matches the input (same mean / std), see
    `observed_pair`.

    Output: (T, N_TGT, H, W)
    """
    T, _, H, W = X.shape
    out = np.zeros((T, N_TGT, H, W), dtype=np.float32)
    cv = channel_valid(X)                                        # (NCH,T,H,W)
    for c in range(NCH):
        m = (cv[c] & walkable[None]).astype(np.float32)
        a = (fwd_channel(X[:, c].astype(np.float64), c) - mean[c][None]) / std[c]
        if full_field:
            # ABLATION (not the paper): supervise every cell, so DINCAE is trained the way
            # the other three methods are and the comparison isolates the architecture from
            # the information form. `a` is already the right target everywhere -- it is built
            # from the ground truth, and on an undefined cell the truth is exactly 0
            # (measured 2026-09-05: of blind cells with density==0, 0.000% have non-zero
            # vx or vy). It is the `a * m` below that throws that target away.
            #
            # Do NOT instead flip the mask slice to 1 while keeping `a * m`: the target lives
            # in NORMALISED space, so a stored 0 decodes to the physical per-cell climatology
            # mean[c] (spatial average -0.067 for vy), and the net would be trained to predict
            # the climatology on empty cells rather than zero.
            #
            # The mask slice going to 1 everywhere also drops the walkable restriction, so the
            # 42 map-obstacle cells that still carry 5.55% of the crowd become supervised too.
            out[:, 2 * c] = a.astype(np.float32)
            out[:, 2 * c + 1] = 1.0
        else:
            out[:, 2 * c] = (a * m).astype(np.float32)
            out[:, 2 * c + 1] = m
    return out
