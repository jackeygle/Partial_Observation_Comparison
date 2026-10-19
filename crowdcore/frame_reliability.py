"""Which grid frames may be scored (route 2: the tracker-gap frames are kept in the data, but
not scored).

After a gap in the raw ATC records the 1-s windowing inherited from Partial_Observation fills
the gap with ~0.05-s snapshots under lagging labels (see
supervisor_evaluation/checks/frame_reliability.py, which writes one bool per grid frame to
FLAG_DIR/<grid stem>.npy). Scoring rule, the same for every method, for selection and for
testing: frames are grouped into aligned BLOCK-frame blocks (the 4DVarNet windows, starting at
frame 0); a block is excluded when it or the block before it holds an unreliable frame, so the
methods that carry state (EnKF) or context (4DVarNet, Senseiver-G, DINCAE t-1) across the gap
are not scored while they recover.

On by default under DATA_VARIANT=all92 and off otherwise (GAP_MASK=0/1 overrides). When off,
every frame is scorable and results are unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from crowdcore import config

BLOCK = 200
FLAG_DIR = Path("/scratch/work/zhangx29/data_all92/frame_reliability")
ENABLED = os.environ.get("GAP_MASK", "1" if config.DATA_VARIANT == "all92" else "0") == "1"


def day_stem(grid_file) -> str:
    """'.../atc-20130616_corridor_1.0s.h5' or '.../atc-20130616.npz' -> 'atc-20130616'."""
    return Path(grid_file).name.split("_")[0].split(".")[0]


def reliable_frames(grid_file, n: int) -> np.ndarray:
    """Per-frame reliability flags for the first n frames of a day (all True when disabled)."""
    if not ENABLED:
        return np.ones(n, bool)
    flags = np.load(FLAG_DIR / f"{day_stem(grid_file)}.npy")
    if len(flags) < n:
        raise ValueError(f"{grid_file}: {len(flags)} reliability flags for {n} frames")
    return flags[:n]


def scorable_frames(grid_file, n: int) -> np.ndarray:
    """Bool per frame: True where the frame may be scored (all True when disabled)."""
    ok = reliable_frames(grid_file, n)
    if ok.all():
        return ok
    nb = -(-n // BLOCK)
    bad = np.zeros(nb, bool)
    bad[np.flatnonzero(~ok) // BLOCK] = True
    bad[1:] |= bad[:-1]                                   # the block after a gap block too
    return np.repeat(~bad, BLOCK)[:n]


def scorable_after_warmup(day, warmup: int, n_scored: int) -> np.ndarray:
    """Mask for an export that holds frames warmup .. warmup+n_scored-1 of a day (EnKF)."""
    return scorable_frames(day, warmup + n_scored)[warmup:]


def describe() -> str:
    return (f"gap mask {'ON' if ENABLED else 'off'} (aligned {BLOCK}-frame blocks with an unreliable "
            f"frame + the next block; flags {FLAG_DIR})")
