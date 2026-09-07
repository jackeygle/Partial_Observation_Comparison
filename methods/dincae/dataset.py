"""
dataset.py — encode one day of ATC into trainable samples (following the paper)
===================================================

**The observation configuration follows `4dvarnet_enkf/config.yaml` exactly** (3
robots / radius 7 / line-of-sight occlusion / the same `obs_std` / the same fixed
seed) -- the two technical routes face exactly the same observation scenario, and
any difference comes only from the method itself. No augmentation like DINCAE 2.0
sec.4's "random per-track dropout" is done -- that would change the observation
scenario.

One structural difference from vanilla DINCAE: **per-frame reconstruction** (the
paper's samples are also "one target instant + several neighbouring input
frames", so this is actually copied as-is, not a change). Adjacent frames are
highly redundant, so target frames are subsampled by `--stride` (default 4).

Channel layout (`N_IN = 6 + 24 = 30`, see `encoding.py`):

    [0:2]    row, col
    [2:6]    cos/sin diurnal cycle, cos/sin weekly cycle
    [6:30]   dt in {-1,0,+1} x (residual[4], mask[4])

Target (8,H,W): one (residual*mask, mask) pair per channel, see
`encoding.encode_target`.
"""
from __future__ import annotations

import os
import sys

import h5py
import numpy as np
import torch

# append rather than insert(0): 4dvarnet_enkf also has a losses.py, inserting at
# the front would shadow this directory's same-named module
from crowdcore import config as cfg4d                                            # noqa: E402
from crowdcore import observation_model as om                                     # noqa: E402

from methods.dincae.state import StateStats, NCH                           # noqa: E402
from methods.dincae.encoding import (FRESH_OFFSETS, N_IN, N_STATIC, encode_target,  # noqa: E402
                      observed_pair, static_channels)

CACHE_VER = 4   # 1=age/aggregation; 2=paper version; 3=+residual normalisation; 4=+var goes through log1p


def obs_config():
    """Observation parameters -- taken **verbatim from 4dvarnet_enkf/config.yaml**,
    no separate defaults set here."""
    return dict(
        sensing_range=cfg4d.get("observation", "sensing_range"),
        num_agents=cfg4d.get("observation", "num_agents"),
        obs_std=np.asarray(cfg4d.get("observation", "obs_std"), dtype=np.float32),
        add_noise=cfg4d.get("observation", "add_noise"),
        obs_every_k=cfg4d.get("observation", "obs_every_k", default=1),
        seed=0,                       # matches 4dvarnet_enkf's build_windows (fixed seed)
    )


def encode_day(fp, stats: StateStats, stride=4, max_frames=0, full_field=False):
    """One day -> (inputs, target); dim 0 is the subsampled target frames.

    inputs (n, N_IN, H, W) float32 ; target (n, 2*NCH, H, W) float32
    """
    oc = obs_config()
    with h5py.File(fp, "r") as f:
        T_all = f["grid"].shape[0]
        T = min(T_all, max_frames) if max_frames else T_all
        X = f["grid"][:T]
        t_unix = f["time"][:T]

    obs = om.generate_observations(
        X, sensing_range=oc["sensing_range"], num_agents=oc["num_agents"],
        add_noise=oc["add_noise"], seed=oc["seed"], valid_mask=stats.valid,
        obs_std=oc["obs_std"], obs_every_k=oc["obs_every_k"])
    Y, M = obs["Y"][:, :NCH], obs["Omega_c"][:, :NCH]

    scaled, invvar = observed_pair(Y, M, stats.mean, stats.std)

    # Target frames: subsample, leaving enough margin for the neighbouring window
    lo, hi = -min(FRESH_OFFSETS), T - max(FRESH_OFFSETS)
    idx = np.arange(lo, hi, stride, dtype=np.int64)
    H, W = X.shape[2], X.shape[3]

    out = np.empty((len(idx), N_IN, H, W), dtype=np.float32)
    out[:, :N_STATIC] = static_channels(t_unix[idx], H, W)
    o = N_STATIC
    for dt in FRESH_OFFSETS:
        j = idx + dt
        out[:, o:o + NCH] = scaled[j]
        out[:, o + NCH:o + 2 * NCH] = invvar[j]
        o += 2 * NCH
    assert o == N_IN, (o, N_IN)

    tgt = encode_target(X[idx], stats.mean, stats.std, stats.valid, full_field)
    return out, tgt


def cache_key(stats: StateStats, stride, full_field=False):
    """Cache key -- encodes everything that would change the encoded result, so
    changing any of them rebuilds automatically."""
    oc = obs_config()
    return (f"v{CACHE_VER}_s{stride}_nd{stats.n_train_days}_c{NCH}"
            f"_na{oc['num_agents']}_sr{oc['sensing_range']}"
            f"_k{oc['obs_every_k']}_sd{oc['seed']}"
            # full-field targets are a DIFFERENT encoding of the same days, so they need
            # their own key -- otherwise they would silently reuse (or overwrite) the 13 GB
            # of information-form cache already on disk.
            + ("_ff1" if full_field else ""))


def encode_day_cached(fp, stats, stride, max_frames, cache_dir, full_field=False):
    """`encode_day` with a disk cache. The observation seed is fixed => the
    encoding is deterministic => only needs computing once.

    Input is stored as float16 (both slices are O(1), no overflow risk), target
    as float32.
    """
    if not cache_dir or max_frames:                 # max_frames is a debug-only truncation, not cached
        return encode_day(fp, stats, stride, max_frames, full_field)
    os.makedirs(cache_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(fp))[0]
    key = cache_key(stats, stride, full_field)
    xp = os.path.join(cache_dir, f"{stem}_{key}_x.npy")
    yp = os.path.join(cache_dir, f"{stem}_{key}_y.npy")
    if not (os.path.exists(xp) and os.path.exists(yp)):
        x, y = encode_day(fp, stats, stride, max_frames, full_field)
        # write to a temp file then rename: an interruption/race never leaves a
        # half-written file that gets treated as a valid cache
        np.save(xp + ".tmp.npy", x.astype(np.float16)); os.replace(xp + ".tmp.npy", xp)
        np.save(yp + ".tmp.npy", y); os.replace(yp + ".tmp.npy", yp)
        print(f"  [cache] wrote {stem} x{x.shape}", flush=True)
    return np.load(xp, mmap_mode="r"), np.load(yp, mmap_mode="r")


class ChunkedDays:
    """Loads several days per chunk, shuffles across days within a chunk, yields minibatches.

    Encoding is per-day (the observation mask is continuous in time), so no
    whole-dataset shuffle is done. Each epoch: shuffle the day order -> read
    `days_per_chunk` days at a time -> shuffle frames within the chunk -> iterate
    minibatches.
    """

    def __init__(self, files, stats, batch_size=64, stride=4, days_per_chunk=4,
                 max_frames=0, shuffle=True, seed=0, cache_dir=None, full_field=False):
        self.files, self.stats = list(files), stats
        self.batch_size, self.stride = batch_size, stride
        self.days_per_chunk, self.max_frames = days_per_chunk, max_frames
        self.shuffle, self.cache_dir = shuffle, cache_dir
        self.full_field = full_field
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        order = list(range(len(self.files)))
        if self.shuffle:
            self.rng.shuffle(order)
        for s in range(0, len(order), self.days_per_chunk):
            xs, ts = [], []
            for di in order[s:s + self.days_per_chunk]:
                x, t = encode_day_cached(self.files[di], self.stats, self.stride,
                                         self.max_frames, self.cache_dir, self.full_field)
                # the cache is mmap'd float16 -> read the whole block into memory and upcast to float32
                xs.append(np.asarray(x, dtype=np.float32))
                ts.append(np.asarray(t, dtype=np.float32))
            X = np.concatenate(xs); Tg = np.concatenate(ts)
            del xs, ts
            perm = self.rng.permutation(len(X)) if self.shuffle else np.arange(len(X))
            for b in range(0, len(perm), self.batch_size):
                j = perm[b:b + self.batch_size]
                yield (torch.from_numpy(X[j]), torch.from_numpy(Tg[j]))
            del X, Tg
