"""
h5_to_grid.py  —  trajectory H5  ->  4-channel macroscopic grid (grid_cache)
============================================================================

This is the SECOND stage of the raw-data pipeline (the first is csv_to_h5.py):

    raw ATC CSV  --[csv_to_h5]-->  trajectory H5  --[THIS FILE]-->  grid_cache (N,4,H,W)

It is a clean, self-contained re-implementation (pure numpy) of the gridding that
the original project did via `pedpred.grid.Grid.points2grid` + `GridFromH5Dataset`.
Re-implementing it here means the clean project fully owns the raw->grid pipeline,
so anyone with the raw data can reproduce `grid_cache/atc-*_corridor_1.0s.h5`.

Faithfulness is checked by `validate_against_cache()`, which compares this output
to the existing grid_cache frame by frame (see `--validate`).

----------------------------------------------------------------------------
Trajectory-H5 schema (input)  /  what each field means
----------------------------------------------------------------------------
  position : (P, 2) float32   world coordinates (x, y) of each pedestrian
                              observation point, in metres
  velocity : (P, 2) float32   corresponding velocity (vx, vy), in m/s
  index    : (M,) struct (time, start, stop, count)
             raw per-timestamp index: position[start:stop] are all pedestrian
             points at that timestamp
  index_1.0s : (M',) struct (time, start, stop, count)
             index re-windowed into 1-second periods (cached by
             GridFromH5Dataset). One frame = all points in the window
             position[start:stop]; count = how many raw timestamps were
             accumulated within that 1 second (used to normalise density by
             the number of time steps).

----------------------------------------------------------------------------
grid_cache schema (output)
----------------------------------------------------------------------------
  grid : (N, 4, H, W) float32   channel order [density, vx, vy, vel_var]
  time : (N,) float64           time in seconds for each frame
  attrs: subset, period, resolution, kernel, origin, theta, shape

  - density : per-cell pedestrian density (kernel density estimate,
              normalised by the count of time steps)
  - vx, vy  : per-cell mean velocity (rotated into the grid coordinate frame)
  - vel_var : per-cell velocity variance (weighted mean of squared vx,vy
              deviations, with unbiased correction)

The corridor subset (matches the existing cache):
  origin=(38.2789, -15.8076), theta=2.5647 rad, shape=(36,12), resolution=1.0 m/cell.
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np

# Subset definitions (identical to SUBSETS in the original build_grid_cache.py)
SUBSETS = {
    "corridor": dict(origin=(38.2789, -15.8076), theta=2.5647, shape=(36, 12)),
    "no_walls": dict(origin=(-3.0431, 4.3197), theta=2.4128, shape=(16, 12)),
    "all":      dict(origin=(55.3890, -8.2735), theta=2.6970, shape=(88, 36)),
}


def rotation_matrix(theta):
    """2x2 rotation matrix R = [[c,-s],[s,c]] (same as the original grid.py)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def points2grid(pos, vel, origin, theta, shape, resolution, count):
    """Grid one frame of points (pos, vel) into (4,H,W) = [density, vx, vy, vel_var].

    Clean reproduction of grid.py's points2grid(kernel='tri', scale=1,
    normalise_count=count, normalise_resolution=resolution, normalise_period=1.0).

    Input
        pos   : (P, 2) world coordinates (metres)
        vel   : (P, 2) velocity (m/s)
        origin, theta, resolution : grid coordinate-frame parameters
        shape : (H, W)
        count : number of time steps accumulated in this window (for density
                normalisation)
    Output
        (4, H, W) float32
    """
    H, W = int(shape[0]), int(shape[1])
    origin = np.asarray(origin, dtype=np.float64)
    R = rotation_matrix(theta)

    # World coords -> local (grid) coords:  local = (global - origin) @ R / resolution
    pos_l = (pos.astype(np.float64) - origin) @ R / resolution      # (P,2)
    vel_l = vel.astype(np.float64) @ R / resolution                 # (P,2) velocity rotated into grid frame too

    # Local coordinates of each cell centre (H,W,2); centres at (i+0.5, j+0.5)
    ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    centers = np.stack([ii, jj], axis=-1) + 0.5                     # (H,W,2)

    P = pos_l.shape[0]
    if P == 0:                                                      # empty frame: all zeros
        return np.zeros((4, H, W), dtype=np.float32)

    # Triangular-kernel weights weights[p,h,w] = prod_d (1-|d_d|) when |d_d|<1, else 0 (scale=1)
    # d = pos_l[p] - center[h,w]
    diff = pos_l[:, None, None, :] - centers[None, :, :, :]         # (P,H,W,2)
    ad = np.abs(diff)
    inside = (ad < 1.0).all(axis=-1)                                # (P,H,W)
    tri = np.prod(1.0 - ad, axis=-1)                                # (P,H,W)
    weights = np.where(inside, tri, 0.0)                            # (P,H,W)
    # Normalisation: *(normalise_resolution/resolution)^2 / count = 1/count (resolution self-cancels)
    weights = weights / max(count, 1)

    # Density = sum of weights over all points
    density = weights.sum(axis=0)                                   # (H,W)

    # Mean velocity (weighted), per component
    vel_num = np.einsum("phw,pd->dhw", weights, vel_l)              # (2,H,W)
    with np.errstate(invalid="ignore", divide="ignore"):
        vel_mean = np.where(density[None] > 0, vel_num / density[None], 0.0)  # (2,H,W)

    # Velocity variance (weighted), with unbiased correction nnz/(nnz-1)
    dev2 = ((vel_l[:, :, None, None] - vel_mean[None]) ** 2).sum(axis=1)      # (P,H,W)
    nnz = (weights != 0).sum(axis=0)                                # (H,W) number of contributing points
    var_num = (weights * dev2).sum(axis=0)                          # (H,W)
    with np.errstate(invalid="ignore", divide="ignore"):
        bias = np.where(nnz > 1, nnz / (nnz - 1.0), 0.0)
        vel_var = np.where(density > 0, bias * var_num / density, 0.0)        # (H,W)
    vel_var[nnz <= 1] = 0.0                                         # variance undefined for 0/1 points -> 0

    out = np.stack([density, vel_mean[0], vel_mean[1], vel_var], axis=0)
    return out.astype(np.float32)


INDEX_DTYPE = np.dtype([("time", "f8"), ("start", "i8"), ("stop", "i8"), ("count", "i8")])


def build_period_index(raw_index, period):
    """Re-window the raw per-timestamp index into `period`-second windows -> index_{period}s.

    Faithful reproduction of the original GridFromH5Dataset windowing: each output
    window is (end_time, start, stop, count), where start..stop covers all raw
    frames inside the window and count = sum of those frames' counts. The final
    incomplete window is not emitted (same as the original implementation). This
    lets an H5 produced by csv_to_h5 (which only has the raw index) be gridded.

    ! Verified inherent behaviour (do NOT "fix"): the first window's count counts
    raw_index[0] twice — the initialisation unpack below already takes its count,
    and the loop accumulates it again from the start. This matches the original
    cached index_1.0s field by field (first window count=26 while the window
    actually contains only 25 timestamps), so the first frame's density is
    slightly low; "fixing" it would break equivalence with the existing
    grid_cache. Verified by: check_data_pipeline.test_end_to_end.
    """
    start_time, start, stop, count = raw_index[0]
    end_time = start_time + period
    out = []
    for frame_time, frame_start, frame_stop, frame_count in raw_index:
        if frame_time >= end_time:
            out.append((end_time, start, stop, count))
            end_time += period
            start = frame_start
            count = 0
        stop = frame_stop
        count += frame_count
    return np.array(out, dtype=INDEX_DTYPE)


def build_grid_for_day(traj_h5_path, subset="corridor", period=1.0, resolution=1.0,
                       max_frames=None):
    """Grid one day's trajectory H5 into (N,4,H,W).

    Prefers the cached `index_{period}s`; if absent (e.g. an H5 freshly produced
    by csv_to_h5), re-windows the raw `index` on the fly (build_period_index).
    """
    s = SUBSETS[subset]
    origin, theta, shape = s["origin"], s["theta"], s["shape"]
    index_name = f"index_{period}s"

    with h5py.File(traj_h5_path, "r") as f:
        if index_name in f:
            index = f[index_name][:]
        else:
            index = build_period_index(f["index"][:], period)   # re-window on the fly
        N = len(index) if max_frames is None else min(max_frames, len(index))
        H, W = shape
        grid = np.zeros((N, 4, H, W), dtype=np.float32)
        times = np.zeros((N,), dtype=np.float64)
        for i in range(N):
            end_time, start, stop, count = index[i]
            pos = f["position"][start:stop]
            vel = f["velocity"][start:stop]
            grid[i] = points2grid(pos, vel, origin, theta, shape, resolution, count)
            times[i] = float(end_time)
    return grid, times


def validate_against_cache(traj_h5_path, cache_h5_path, subset="corridor",
                           period=1.0, n=30, atol=1e-4):
    """Cross-check: this implementation vs the existing grid_cache, comparing per-channel max abs error over the first n frames."""
    grid, _ = build_grid_for_day(traj_h5_path, subset, period, max_frames=n)
    with h5py.File(cache_h5_path, "r") as f:
        ref = f["grid"][:n]
    names = ("density", "vx", "vy", "var")
    print(f"[validate] first {n} frames, our impl vs grid_cache, per-channel max absolute error:")
    ok = True
    for c in range(4):
        err = float(np.max(np.abs(grid[:, c] - ref[:, c])))
        flag = "OK" if err <= atol else "DIFF"
        if err > atol:
            ok = False
        print(f"    {names[c]:8s} max|Δ| = {err:.3e}   [{flag}]")
    print(f"[validate] {'ALL PASS ✓' if ok else 'MISMATCH FOUND ✗'} (atol={atol})")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Trajectory H5 -> 4-channel grid_cache (clean reimpl)")
    ap.add_argument("--traj-h5", required=True, help="input trajectory H5 (position/velocity/index_1.0s)")
    ap.add_argument("--out", default=None, help="output grid_cache H5 path (if omitted, only validation runs)")
    ap.add_argument("--subset", default="corridor", choices=list(SUBSETS))
    ap.add_argument("--period", type=float, default=1.0)
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--validate", default=None, help="path to an existing grid_cache to cross-check against")
    ap.add_argument("--n", type=int, default=30, help="use the first n frames for validation")
    args = ap.parse_args()

    if args.validate:
        validate_against_cache(args.traj_h5, args.validate, args.subset, args.period, args.n)
        return

    grid, times = build_grid_for_day(args.traj_h5, args.subset, args.period, args.resolution)
    if args.out:
        s = SUBSETS[args.subset]
        with h5py.File(args.out, "w") as h:
            h.create_dataset("grid", data=grid, chunks=(1, 4, *s["shape"]), compression="lzf")
            h.create_dataset("time", data=times)
            h.attrs["subset"] = args.subset
            h.attrs["period"] = args.period
            h.attrs["resolution"] = args.resolution
            h.attrs["kernel"] = "tri"
            h.attrs["origin"] = np.asarray(s["origin"], dtype="f8")
            h.attrs["theta"] = float(s["theta"])
            h.attrs["shape"] = np.asarray(s["shape"], dtype="i4")
        print(f"wrote {grid.shape} -> {args.out}")


if __name__ == "__main__":
    main()
