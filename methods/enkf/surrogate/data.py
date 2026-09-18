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

import numpy as np
import torch
import h5py

from crowdcore import observation_model as om


class PairFrames:
    def __init__(self, split: str, device, days: int = 0, max_frames: int = 0,
                 input_frames: int = 1, output_frames: int = 1):
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

    @property
    def n_pairs(self) -> int:
        return len(self.index)

    def batch(self, rows):
        """Return x ``(B,input_frames,4,H,W)`` and y ``(B,output_frames,4,H,W)``."""
        target = self.index[rows]
        history_offsets = torch.arange(-self.input_frames, 0, device=target.device)
        future_offsets = torch.arange(self.output_frames, device=target.device)
        history = self.frames[target[:, None] + history_offsets[None]]
        future = self.frames[target[:, None] + future_offsets[None]]
        return history, future
