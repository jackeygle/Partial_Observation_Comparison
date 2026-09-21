"""
data.py — causal history -> future-frame samples from grid_cache, held on the training device

grid_cache holds the same gridded states the EnKF runs on (4 channels, 36x12, 1 s), one file
per day. The original train.py built the same pairs with SeqDataset(nin=1, nout=1, step=1):
every consecutive frame pair within a day, none across a day boundary.

The defaults remain one input and one output frame. ``input_frames=5`` and
``output_frames=5`` produce ``(x_{t-4:t}, x_{t+1:t+5})`` without copying the
stored frame tensor. The whole split
lives on the device (train ~8.7 GB float32) and batches are gathered by index
there -- no DataLoader round trip per batch.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import h5py

from crowdcore import config, navigation
from crowdcore import observation_model as om


def _simulate_history(path, X, fill=None):
    """One day's partial observation, filled into the complete grid a CNN needs.

    Same simulator, same per-day trajectory seed and same fill rule the other
    methods use, so what LCSKF trains on is the observation DINCAE and 4DVarNet
    are given.  Roughly 21 s per full day on one CPU core.
    """
    obs = config.get("observation")
    sim = om.generate_observations(
        X, sensing_range=obs["sensing_range"], num_agents=obs["num_agents"],
        add_noise=obs["add_noise"], seed=om.day_seed(path, 0, obs["trajectory_mode"]),
        valid_mask=navigation.build_valid_mask_from_config(X),
        obs_std=np.asarray(obs["obs_std"], dtype=np.float32),
        obs_every_k=obs["obs_every_k"])
    filled = om.fill_missing_state(sim["Y"], sim["Omega_c"],
                                   method=fill or obs["init_method"])
    return filled.astype(np.float32), sim["Omega"].astype(bool)


class PairFrames:
    def __init__(self, split: str, device, days: int = 0, max_frames: int = 0,
                 input_frames: int = 1, output_frames: int = 1, history: str = "truth",
                 history_fill: str = ""):
        """``history`` chooses what the *inputs* are.  Targets are always the truth.

        ``"truth"``     the true past state.  Free, and what one-step scoring assumes,
                        but not what the filter ever sees once it is running.
        ``"observed"``  the robots' partial observation of the past, with unobserved
                        cells carried forward (``fill_missing_state``) so the grid a
                        convolution needs is complete.  This is the same X0 4DVarNet
                        starts its solver from, simulated here with the same
                        ``generate_observations`` and per-day seed every method uses.
        ``history_fill`` picks how the unobserved cells are filled: ``"prev"`` carries
        the last measured value forward, ``"zeros"`` leaves them at zero.  Zero is the
        better guess here -- 81.6% of unobserved walkable cells are genuinely empty --
        but it is only safe to use alongside the observation-mask channel, which is
        what tells a zero that means "nobody here" from one that means "no idea".

        a path          a directory of ``bg_<day>.npz`` written by
                        ``checks/export_backgrounds.py`` (the filter's own analyses).

        Either non-truth mode also carries the observation mask, so the Kalman
        correction can be applied where the robots actually were rather than at a
        freshly sampled layout.
        """
        if input_frames < 1:
            raise ValueError("input_frames must be positive")
        if output_frames < 1:
            raise ValueError("output_frames must be positive")
        files = om.split_files(split)
        if days:
            files = files[:days]
        chunks = []
        for fp in files:
            if max_frames:
                # Debug/smoke runs need only a short prefix.  Reading the HDF5
                # slice directly avoids loading and discarding ~40k frames per
                # day before taking the first few hundred.
                with h5py.File(fp, "r") as handle:
                    X = handle["grid"][:max_frames].astype(np.float32)
            else:
                X, _ = om.load_state(fp)
            chunks.append(X)
        lengths = [len(X) for X in chunks]
        frames = np.concatenate(chunks)
        del chunks
        # Store the first target-frame index. Neither history nor future may
        # cross a day boundary.
        starts = np.r_[0, np.cumsum(lengths)[:-1]]
        stops = np.cumsum(lengths)
        targets = [np.arange(start + input_frames, stop - output_frames + 1, dtype=np.int64)
                   for start, stop in zip(starts, stops)]
        self.split, self.n_days, self.n_frames = split, len(files), len(frames)
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.frames = torch.from_numpy(frames).to(device)
        self.index = torch.from_numpy(np.concatenate(targets)).to(device)
        self.history_mode = history
        self.background, self.observed = None, None
        if history != "truth":
            backgrounds, observed = [], []
            offset = 0
            for fp, n in zip(files, lengths):
                stem = os.path.basename(fp).split("_")[0]
                if history == "observed":
                    est, omega = _simulate_history(fp, frames[offset:offset + n], history_fill)
                else:
                    bg = os.path.join(history, f"bg_{stem}.npz")
                    with np.load(bg) as handle:
                        est, omega = handle["Est"], handle["Omega"]
                if len(est) < n:
                    raise ValueError(f"{stem}: history has {len(est)} frames, need {n}")
                backgrounds.append(est[:n]); observed.append(omega[:n])
                offset += n
            self.background = torch.from_numpy(np.concatenate(backgrounds)).to(device)
            self.observed = torch.from_numpy(np.concatenate(observed)).to(device)
            del backgrounds, observed

    @property
    def n_pairs(self) -> int:
        return len(self.index)

    def batch(self, rows, truth_history=False):
        """Return x ``(B,input_frames,4,H,W)`` and y ``(B,output_frames,4,H,W)``.

        ``truth_history`` forces the true past state even when a non-truth history
        was loaded, so one training run can mix the two regimes -- the standard
        answer to the robustness/clean trade-off that training on the degraded
        input alone produces.
        """
        target = self.index[rows]
        history_offsets = torch.arange(-self.input_frames, 0, device=target.device)
        future_offsets = torch.arange(self.output_frames, device=target.device)
        source = (self.frames if (truth_history or self.background is None)
                  else self.background)
        history = source[target[:, None] + history_offsets[None]]
        future = self.frames[target[:, None] + future_offsets[None]]
        return history, future

    def observed_at(self, rows, offset=0):
        """Robot mask (B,H,W) at the target frame plus ``offset``, or None on truth.

        ``offset=0`` is the frame being assimilated; ``offset=-1`` is the frame the
        covariance net is handed as ``x_t``.  ``input_frames >= 1`` keeps every
        offset in [-input_frames, 0] inside the same day.
        """
        if self.observed is None:
            return None
        if not -self.input_frames <= offset <= 0:
            raise ValueError(f"offset {offset} leaves the sample's own day")
        return self.observed[self.index[rows] + offset]
