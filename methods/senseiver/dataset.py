"""
dataset.py — one day of ATC -> Senseiver's training samples
================================================

**The observation scenario follows `4dvarnet_enkf/config.yaml` exactly** (3
robots / sensing radius 7 / line-of-sight occlusion / the same obs_std / the
same obs_every_k / the same walkable rule), calling its
`observation_model.generate_observations` and
`navigation.build_valid_mask_from_config` directly. All three methods face
exactly the same observations, and any difference comes only from the method
itself -- this is the premise the three-way comparison rests on.

This directory sets no defaults anywhere for the observation parameters; to
change them, edit `4dvarnet_enkf/config.yaml`.

Sample granularity
----------------
In the reference implementation, one training sample = (one frame's sensor
readings, that frame's query points). We follow the same convention: **one
sample per frame**, with query points being the entire 432-cell grid (see
`sensors.query_all`'s note).

Adjacent frames (1-second spacing) are highly redundant, subsampled by
`--stride`, default 4. This is not a deviation from the paper -- the reference
implementation likewise trains on `training_frames` frames randomly drawn from
all frames.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# append rather than insert(0): 4dvarnet_enkf also has a losses.py, inserting at
# the front would shadow this directory's.
from crowdcore import config as cfg4d                                          # noqa: E402
from crowdcore import navigation as nav                                        # noqa: E402
from crowdcore import observation_model as om                                  # noqa: E402


def obs_config():
    """Observation parameters -- taken verbatim from 4dvarnet_enkf/config.yaml."""
    return dict(
        sensing_range=cfg4d.get("observation", "sensing_range"),
        num_agents=cfg4d.get("observation", "num_agents"),
        add_noise=cfg4d.get("observation", "add_noise"),
        obs_every_k=cfg4d.get("observation", "obs_every_k"),
        init_method=cfg4d.get("observation", "init_method"),
    )


def state_shape():
    """(C, H, W) -- taken from config.yaml's grid.state_shape."""
    return tuple(cfg4d.get("grid", "state_shape"))


def channels():
    return list(cfg4d.get("grid", "channels"))


# --------------------------------------------------------------------------- #
def load_day(path, stride=4, seed=0, frames=0, obs_every_k=None, add_noise=None):
    """One day -> (X, Y, Omega), all subsampled and flattened to (N, C, HW) / (N, HW).

    When frames>0, only takes that day's first `frames` frames (used during
    evaluation to align with another method's frame count).
    """
    oc = obs_config()
    X, _ = om.load_state(path)
    X = np.asarray(X, dtype=np.float32)
    if frames > 0:
        X = X[:frames]
    valid = nav.build_valid_mask_from_config(X)
    out = om.generate_observations(
        X, sensing_range=oc["sensing_range"], num_agents=oc["num_agents"],
        add_noise=oc["add_noise"] if add_noise is None else add_noise,
        seed=seed, valid_mask=valid,
        obs_every_k=oc["obs_every_k"] if obs_every_k is None else obs_every_k)

    idx = np.arange(0, X.shape[0], stride)
    T, C, H, W = X.shape
    Xs = X[idx].reshape(len(idx), C, H * W)
    Ys = out["Y"][idx].reshape(len(idx), C, H * W).astype(np.float32)
    Om = out["Omega"][idx].reshape(len(idx), H * W)
    return Xs, Ys, Om


def day_observation_seed(base_seed, day_index, trajectory_mode):
    """Return a reproducible observation seed for one day.

    ``fixed`` preserves the original protocol: every day replays exactly the
    same robot trajectory.  ``per_day`` keeps the experiment deterministic but
    gives day ``i`` its own trajectory.
    """
    if trajectory_mode == "fixed":
        return int(base_seed)
    if trajectory_mode == "per_day":
        return int(base_seed) + int(day_index)
    raise ValueError(f"unknown trajectory_mode={trajectory_mode!r}")


class DayBank:
    """Concatenates several days into one pool that batches can be randomly
    drawn from (the reference implementation also holds everything resident in
    memory and subsamples frames randomly).

    Memory: about 2 x N x C x HW x 4B per day after subsampling. At stride=4,
    one day is about 138 MB, 32 days about 4.4 GB -- 64G requested in the sbatch
    scripts is plenty.
    """

    def __init__(self, files, stride=4, seed=0, max_days=0, frames=0,
                 obs_every_k=None, add_noise=None, verbose=True,
                 trajectory_mode="fixed"):
        files = files[:max_days] if max_days else files
        self.observation_seeds = []
        Xs, Ys, Os = [], [], []
        for i, f in enumerate(files):
            day_seed = day_observation_seed(seed, i, trajectory_mode)
            self.observation_seeds.append(day_seed)
            x, y, o = load_day(f, stride, day_seed, frames, obs_every_k, add_noise)
            Xs.append(x); Ys.append(y); Os.append(o)
            if verbose:
                print(f"  [{i+1}/{len(files)}] {os.path.basename(f).split('_')[0]}: "
                      f"{x.shape[0]} frames, observed cells/frame {o.sum(1).mean():.1f}, "
                      f"observation seed {day_seed}", flush=True)
        self.X = np.concatenate(Xs, 0)
        self.Y = np.concatenate(Ys, 0)
        self.Omega = np.concatenate(Os, 0)
        self.n = self.X.shape[0]

    def input_stats(self):
        """Per-channel standardisation statistics for the encoder's input:
        computed over **observed cells** only, since those are the only
        readings that reach the encoder."""
        m = self.Omega                                        # (N, HW)
        vals = [self.Y[:, c][m] for c in range(self.Y.shape[1])]
        mean = np.array([v.mean() for v in vals], np.float32)
        std = np.array([max(v.std(), 1e-6) for v in vals], np.float32)
        return mean, std

    def batches(self, batch, rng, drop_last=True):
        order = rng.permutation(self.n)
        stop = (self.n // batch) * batch if drop_last else self.n
        for i in range(0, stop, batch):
            yield order[i:i + batch]


class TemporalDayBank:
    """`DayBank` for the temporal extension (not in the reference implementation).

    A sample still targets one frame, chosen exactly as `DayBank` chooses it -- every
    `stride`-th frame of each day, observations generated over the whole day with the
    same seed -- so a k-frame run trains on the **same targets and the same current-frame
    observations** as its k=1 counterpart with that seed. What changes is that each
    day's observations are kept at full frame rate, so a sample can look back
    `window-1` frames. Windows never cross a day boundary: near the start of a day they
    are simply shorter.

    Memory: full-rate Y is about 276 MB per day, ~8.8 GB for the 32 training days,
    on top of the subsampled targets. The sbatch script requests 96G.
    """

    def __init__(self, files, stride=4, seed=0, max_days=0, frames=0,
                 obs_every_k=None, add_noise=None, verbose=True, *, window,
                 trajectory_mode="fixed"):
        if window < 2:
            raise ValueError(f"TemporalDayBank needs window >= 2, got {window}; use DayBank")
        files = files[:max_days] if max_days else files
        self.window = window
        self.observation_seeds = []
        self.Yd, self.Od, self.frames = [], [], []
        Xs, day_of, frame_of = [], [], []
        for d, f in enumerate(files):
            day_seed = day_observation_seed(seed, d, trajectory_mode)
            self.observation_seeds.append(day_seed)
            X, Y, Om = load_day(f, 1, day_seed, frames, obs_every_k, add_noise)  # full frame rate
            idx = np.arange(0, X.shape[0], stride)                            # == DayBank's targets
            Xs.append(X[idx]); self.Yd.append(Y); self.Od.append(Om); self.frames.append(idx)
            day_of.append(np.full(len(idx), d, dtype=np.int32))
            frame_of.append(idx.astype(np.int64))
            del X
            if verbose:
                print(f"  [{d+1}/{len(files)}] {os.path.basename(f).split('_')[0]}: "
                      f"{len(idx)} target frames of {Y.shape[0]}, observed cells/frame "
                      f"{Om.sum(1).mean():.1f}, window {window}, observation seed "
                      f"{day_seed}", flush=True)
        self.X = np.concatenate(Xs, 0)
        self.day_of = np.concatenate(day_of)
        self.frame_of = np.concatenate(frame_of)
        self.n = self.X.shape[0]

    def windows(self, idx):
        """[(Y_win (K,C,HW), Omega_win (K,HW)), ...] oldest first, last = the target frame."""
        out = []
        for i in idx:
            d, f = self.day_of[i], self.frame_of[i]
            lo = max(0, f - self.window + 1)
            out.append((self.Yd[d][lo:f + 1], self.Od[d][lo:f + 1]))
        return out

    def target_omega(self, idx):
        """(len(idx), HW) observation mask of each target frame."""
        return np.stack([self.Od[self.day_of[i]][self.frame_of[i]] for i in idx])

    def input_stats(self):
        """Same statistic as `DayBank.input_stats`, over the same cells: observed cells of
        the target frames only."""
        C = self.Yd[0].shape[1]
        vals = [[] for _ in range(C)]
        for Y, O, idx in zip(self.Yd, self.Od, self.frames):
            Ys, Os = Y[idx], O[idx]
            for c in range(C):
                vals[c].append(Ys[:, c][Os])
        vals = [np.concatenate(v) for v in vals]
        mean = np.array([v.mean() for v in vals], np.float32)
        std = np.array([max(v.std(), 1e-6) for v in vals], np.float32)
        return mean, std

    def batches(self, batch, rng, drop_last=True):
        order = rng.permutation(self.n)
        stop = (self.n // batch) * batch if drop_last else self.n
        for i in range(0, stop, batch):
            yield order[i:i + batch]
