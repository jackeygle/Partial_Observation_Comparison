"""
check_data_pipeline.py  —  verification of the raw-data pipeline (CSV -> H5 -> grid)
=====================================================================================

This is NOT part of the pipeline. It only *exercises* `data_pipeline/`:

  [1] synthetic micro-example (answers computed BY HAND — proves absolute
      correctness, independent of any reference data):
        - unit conversion (mm -> m, spd/ang -> vx,vy),
        - index grouping by timestamp, incl. groups split across chunk boundaries,
        - points2grid triangular-kernel weights vs hand-computed values,
        - points2grid vs an independent brute-force reimplementation.
  [2] stage-1 faithfulness on REAL data (stronger than the built-in 200-row check):
        convert() on a truncated real CSV, then compare ALL rows of
        position/velocity AND the index records against the reference trajectory H5.
  [3] end-to-end: our own H5 (no cached index_1.0s -> forces build_period_index)
        -> grid, compared channel-by-channel against the existing grid_cache.
        This is the full CSV -> H5 -> grid chain in one run.

Run:
    module load scicomp-pytorch-env/2026.1
    python3 checks/check_data_pipeline.py                  # all three layers
    python3 checks/check_data_pipeline.py --synthetic-only # no real data needed
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make parent-level modules importable

import h5py
import numpy as np

from data_pipeline import csv_to_h5
from data_pipeline.csv_to_h5 import _iter_csv_chunks, convert
from data_pipeline.h5_to_grid import (
    SUBSETS, build_grid_for_day, build_period_index, points2grid, rotation_matrix,
)

# Default real-data paths (Triton)
DEFAULT_CSV = "/scratch/work/zhangx29/ATC/atc-20121028.csv"
DEFAULT_REF_H5 = "/scratch/work/zhangx29/data/ATC/Sundays/atc-20121028.h5"
DEFAULT_REF_CACHE = "/scratch/work/zhangx29/data/grid_cache/atc-20121028_corridor_1.0s.h5"


# --------------------------------------------------------------------------- #
# [1] Synthetic micro-example — hand-computed answers
# --------------------------------------------------------------------------- #
def _write_synthetic_csv(path, rows):
    """rows = list of (time, pid, x_mm, y_mm, z_mm, spd_mm, ang, face)."""
    with open(path, "w") as f:
        for r in rows:
            f.write(",".join(f"{v:.6f}" for v in r) + "\n")


def test_stage1_synthetic(tmpdir):
    """Unit conversion + index grouping, incl. chunk-boundary handling."""
    # 4 timestamps with group sizes 3/2/1/2 — boundaries and row counts deliberately irregular
    rows = [
        (100.0, 1, 1000, -2000, 0, 500, 0.0, 0),          # t=100.0 (3 people)
        (100.0, 2, 2500, 3000, 0, 1000, np.pi / 2, 0),
        (100.0, 3, -4000, 500, 0, 2000, np.pi, 0),
        (100.5, 1, 1100, -1900, 0, 500, 0.25, 0),         # t=100.5 (2 people)
        (100.5, 2, 2500, 3100, 0, 800, -0.5, 0),
        (101.7, 3, -3800, 700, 0, 1500, 1.0, 0),          # t=101.7 (1 person)
        (103.2, 1, 1200, -1800, 0, 300, 2.0, 0),          # t=103.2 (2 people)
        (103.2, 4, 0, 0, 0, 0, 0.0, 0),
    ]
    csv_path = os.path.join(tmpdir, "synthetic.csv")
    _write_synthetic_csv(csv_path, rows)

    # Force tiny chunks (64 B ≈ 1 row per chunk) so timestamp groups span chunk
    # boundaries — tests convert()'s boundary logic
    orig_iter = csv_to_h5._iter_csv_chunks
    csv_to_h5._iter_csv_chunks = lambda p: orig_iter(p, chunk_bytes=64)
    try:
        h5_path = os.path.join(tmpdir, "synthetic.h5")
        convert(csv_path, h5_path)
    finally:
        csv_to_h5._iter_csv_chunks = orig_iter

    with h5py.File(h5_path, "r") as f:
        pos, vel, index = f["position"][:], f["velocity"][:], f["index"][:]

    # Hand-computed expectation: pos = mm/1000, vel = spd/1000 * (cos ang, sin ang)
    exp_pos = np.array([[r[2] / 1000, r[3] / 1000] for r in rows], dtype=np.float32)
    exp_vel = np.array([[r[5] / 1000 * np.cos(r[6]), r[5] / 1000 * np.sin(r[6])]
                        for r in rows], dtype=np.float32)
    assert np.allclose(pos, exp_pos, atol=1e-6), "position unit conversion wrong"
    assert np.allclose(vel, exp_vel, atol=1e-6), "velocity conversion wrong"

    # Hand-computed expected index: (time, start, stop, count=1)
    # Time-label convention (verified record-by-record against the original pipeline's
    # reference H5): record i's start:stop covers the rows of the i-th timestamp group,
    # but time = the (i+1)-th group's timestamp (last record = its own group's timestamp).
    # I.e. time is the group's END time — the same convention index_1.0s uses, labelling
    # windows by their end_time.
    exp_index = [(100.5, 0, 3, 1), (101.7, 3, 5, 1), (103.2, 5, 6, 1), (103.2, 6, 8, 1)]
    assert len(index) == 4, f"expected 4 timestamp groups, got {len(index)}"
    for rec, (t, s, e, c) in zip(index, exp_index):
        assert (abs(rec["time"] - t) < 1e-9 and rec["start"] == s
                and rec["stop"] == e and rec["count"] == c), f"index record wrong: {rec} != {(t, s, e, c)}"

    # Chunked reading itself must not drop/reorder rows: tiny chunks concatenated == whole-file read
    whole = np.loadtxt(csv_path, delimiter=",", usecols=(0, 2, 3, 5, 6))
    chunked = np.concatenate(list(_iter_csv_chunks(csv_path, chunk_bytes=48)))
    assert np.array_equal(whole, chunked), "chunked reader corrupts rows"

    # 1s windowing (build_period_index), traced by hand under the end-time label convention:
    #   First window starts at index[0].time=100.5, end_time=101.5: takes (100.5,0,3) -> (101.5, 0, 3, count=2)
    #     First-window count=2, not 1: initialisation already took index[0].count, then the
    #     loop accumulates it again from the start — the first record is double-counted.
    #     Verified on the reference H5: this exactly matches the index_1.0s cached by the
    #     original GridFromH5Dataset (first window count=26). It is inherent behaviour of the
    #     original pipeline; reproducing it is required for equivalence with the existing grid_cache.
    #   101.7>=101.5 opens a new window end=102.5: takes (101.7,3,5)  -> (102.5, 3, 5, 1)
    #   103.2>=102.5 opens a new window end=103.5: takes the two 103.2 records but the file
    #     ends — incomplete windows are not emitted
    pidx = build_period_index(index, 1.0)
    exp_pidx = [(101.5, 0, 3, 2), (102.5, 3, 5, 1)]
    assert len(pidx) >= 2, f"expected >=2 period windows, got {len(pidx)}"
    for rec, (t, s, e, c) in zip(pidx, exp_pidx):
        assert (abs(rec["time"] - t) < 1e-9 and rec["start"] == s
                and rec["stop"] == e and rec["count"] == c), f"period window wrong: {rec} != {(t, s, e, c)}"
    print("[1a] stage-1 synthetic: unit conversion, index grouping (cross-chunk), 1s windowing OK")


def points2grid_bruteforce(pos, vel, origin, theta, shape, resolution, count):
    """Independent reference: same math as grid.py, written as plain loops."""
    H, W = shape
    R = rotation_matrix(theta)
    pos_l = (pos.astype(np.float64) - np.asarray(origin)) @ R / resolution
    vel_l = vel.astype(np.float64) @ R / resolution
    out = np.zeros((4, H, W))
    for i in range(H):
        for j in range(W):
            c = np.array([i + 0.5, j + 0.5])
            w = np.zeros(len(pos_l))
            for p in range(len(pos_l)):
                d = np.abs(pos_l[p] - c)
                if (d < 1.0).all():
                    w[p] = (1 - d[0]) * (1 - d[1])
            w = w / max(count, 1)
            dens = w.sum()
            out[0, i, j] = dens
            if dens > 0:
                vm = (w[:, None] * vel_l).sum(axis=0) / dens
                out[1, i, j], out[2, i, j] = vm
                nnz = int((w != 0).sum())
                if nnz > 1:
                    dev2 = ((vel_l - vm) ** 2).sum(axis=1)
                    out[3, i, j] = (nnz / (nnz - 1.0)) * (w * dev2).sum() / dens
    return out.astype(np.float32)


def test_stage2_synthetic(seed=0):
    """points2grid: hand-computed single-point weights + brute-force cross-check."""
    s = SUBSETS["corridor"]
    origin, theta, shape, res = np.array(s["origin"]), s["theta"], s["shape"], 1.0
    R = rotation_matrix(theta)

    # --- Hand-computed: 1 pedestrian at local coords (10.75, 5.25), i.e. cell (10,5)
    #     centre offset by (+0.25,-0.25)
    local = np.array([[10.75, 5.25]])
    world = local * res @ R.T + origin                    # inverse transform back to world coords
    # Round-trip through the inverse transform must recover the original local coords
    assert np.allclose((world - origin) @ R / res, local, atol=1e-9)
    v_world = np.array([[1.0, 0.5]])
    g = points2grid(world, v_world, origin, theta, shape, res, count=1)

    # Triangular-kernel weights (hand-computed): (1-|di|)(1-|dj|) over the 4 neighbouring cells
    exp = {(10, 5): 0.75 * 0.75, (11, 5): 0.25 * 0.75, (10, 4): 0.75 * 0.25, (11, 4): 0.25 * 0.25}
    for (i, j), w in exp.items():
        assert abs(g[0, i, j] - w) < 1e-6, f"density weight at {(i, j)}: {g[0, i, j]} != {w}"
    assert abs(g[0].sum() - 1.0) < 1e-6, "single-point density must sum to 1 (mass conservation)"
    # Velocity channels = rotated velocity (same for all covered cells); single-point variance = 0
    v_local = (v_world @ R / res)[0]
    for (i, j) in exp:
        assert np.allclose(g[1:3, i, j], v_local, atol=1e-6), f"velocity at {(i, j)} wrong"
        assert g[3, i, j] == 0.0, "single-point variance must be 0"

    # --- count normalisation: with count=4 the density must be exactly 1/4
    g4 = points2grid(world, v_world, origin, theta, shape, res, count=4)
    assert abs(g4[0].sum() - 0.25) < 1e-6, "count normalisation wrong"

    # --- Empty frame -> all zeros
    empty = points2grid(np.zeros((0, 2)), np.zeros((0, 2)), origin, theta, shape, res, 1)
    assert (empty == 0).all()

    # --- Random multi-point vs independent brute-force implementation (incl. variance channel)
    rng = np.random.default_rng(seed)
    for _ in range(5):
        P = int(rng.integers(2, 30))
        local_pts = rng.uniform([0, 0], shape, size=(P, 2))
        world_pts = local_pts * res @ R.T + origin
        vels = rng.normal(0, 1.5, size=(P, 2))
        cnt = int(rng.integers(1, 30))
        a = points2grid(world_pts, vels, origin, theta, shape, res, cnt)
        b = points2grid_bruteforce(world_pts, vels, origin, theta, shape, res, cnt)
        assert np.allclose(a, b, atol=1e-5), f"points2grid != brute force (max|Δ|={np.abs(a - b).max():.2e})"
        # Mass conservation: each point fully inside the grid contributes 1/count
        fully_inside = ((local_pts > 1) & (local_pts < np.array(shape) - 1)).all(axis=1).sum()
        assert a[0].sum() * cnt >= fully_inside - 1e-6, "mass conservation violated"
    print("[1b] stage-2 synthetic: hand-computed kernel weights, count norm, brute-force x5 OK")


# --------------------------------------------------------------------------- #
# [2] Stage 1 on real data — full comparison on a truncated CSV
# --------------------------------------------------------------------------- #
def _truncate_csv(csv_path, out_path, n_lines):
    with open(csv_path, "rb") as src, open(out_path, "wb") as dst:
        for _ in range(n_lines):
            line = src.readline()
            if not line:
                break
            dst.write(line)


def test_stage1_real(csv_path, ref_h5_path, tmpdir, n_lines):
    trunc_csv = os.path.join(tmpdir, "trunc.csv")
    _truncate_csv(csv_path, trunc_csv, n_lines)
    our_h5 = os.path.join(tmpdir, "ours.h5")
    convert(trunc_csv, our_h5)

    with h5py.File(our_h5, "r") as f:
        pos, vel, index = f["position"][:], f["velocity"][:], f["index"][:]
    N = len(pos)
    with h5py.File(ref_h5_path, "r") as f:
        ref_pos, ref_vel = f["position"][:N], f["velocity"][:N]
        ref_index = f["index"][:len(index)]

    ep = float(np.max(np.abs(pos - ref_pos)))
    ev = float(np.max(np.abs(vel - ref_vel)))
    assert ep == 0.0 and ev == 0.0, f"stage-1 mismatch: pos max|Δ|={ep:.2e} vel max|Δ|={ev:.2e}"

    # index invariants: groups tile all rows contiguously, timestamps strictly increasing
    # (under the end-time label convention, the last record's time = its own group's
    #  timestamp = the second-to-last record's label, so only check [:-1])
    assert index["start"][0] == 0 and index["stop"][-1] == N
    assert (index["start"][1:] == index["stop"][:-1]).all(), "index groups must tile the rows"
    assert (np.diff(index["time"][:-1]) > 0).all(), "timestamps must be strictly increasing"
    assert index["time"][-1] == index["time"][-2], "tail record must carry its own group's timestamp"

    # Record-by-record comparison against the reference H5 index (last group may be
    # incomplete due to truncation — excluded)
    m = len(index) - 1
    for name in ("time", "start", "stop", "count"):
        assert np.array_equal(index[name][:m], ref_index[name][:m]), \
            f"index field '{name}' differs from reference H5"
    print(f"[2] stage-1 real: {N} rows pos/vel exact match, {m} index records match reference")
    return our_h5


# --------------------------------------------------------------------------- #
# [3] End-to-end: our H5 (fresh windowing) -> grid vs existing grid_cache
# --------------------------------------------------------------------------- #
def test_end_to_end(our_h5, ref_h5_path, ref_cache_path, frames, atol=1e-4):
    # Our H5 only has the raw index -> build_grid_for_day will window on the fly via build_period_index
    with h5py.File(our_h5, "r") as f:
        assert "index_1.0s" not in f, "test premise broken: fresh H5 must not carry cached windows"
        our_windows = build_period_index(f["index"][:], 1.0)

    # 3a) On-the-fly windowing vs the index_1.0s cached in the reference H5
    #     (last window may be affected by truncation — excluded)
    with h5py.File(ref_h5_path, "r") as f:
        assert "index_1.0s" in f, f"reference H5 has no index_1.0s (found: {list(f.keys())})"
        ref_windows = f["index_1.0s"][:len(our_windows)]
    m = len(our_windows) - 1
    for name in ("time", "start", "stop", "count"):
        assert np.array_equal(our_windows[name][:m], ref_windows[name][:m]), \
            f"build_period_index field '{name}' differs from cached index_1.0s"

    # 3b) Full-chain grid vs the existing grid_cache, channel by channel
    n = min(frames, m)
    grid, times = build_grid_for_day(our_h5, "corridor", 1.0, max_frames=n)
    with h5py.File(ref_cache_path, "r") as f:
        ref_grid, ref_times = f["grid"][:n], f["time"][:n]
    assert np.allclose(times, ref_times), "frame times differ from grid_cache"
    names = ("density", "vx", "vy", "var")
    errs = []
    for c in range(4):
        err = float(np.max(np.abs(grid[:, c] - ref_grid[:, c])))
        errs.append(f"{names[c]}={err:.2e}")
        assert err <= atol, f"channel {names[c]}: max|Δ|={err:.3e} > atol={atol}"
    print(f"[3] end-to-end: {m} windows match index_1.0s; {n} frames vs grid_cache "
          f"max|Δ| {' '.join(errs)} (atol={atol})")


def main():
    ap = argparse.ArgumentParser(description="Verify the raw-data pipeline (CSV -> H5 -> grid)")
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--ref-h5", default=DEFAULT_REF_H5, help="reference trajectory H5 (same day)")
    ap.add_argument("--ref-cache", default=DEFAULT_REF_CACHE, help="reference grid_cache (same day)")
    ap.add_argument("--lines", type=int, default=300_000, help="CSV lines to truncate for the real-data test")
    ap.add_argument("--frames", type=int, default=60, help="frames to compare against grid_cache")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic-only", action="store_true", help="skip the real-data sections")
    args = ap.parse_args()

    tmpdir = tempfile.mkdtemp(prefix="check_pipeline_")
    try:
        test_stage1_synthetic(tmpdir)
        test_stage2_synthetic(seed=args.seed)
        if not args.synthetic_only:
            our_h5 = test_stage1_real(args.csv, args.ref_h5, tmpdir, args.lines)
            test_end_to_end(our_h5, args.ref_h5, args.ref_cache, args.frames)
        print("\nall data-pipeline checks passed")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
