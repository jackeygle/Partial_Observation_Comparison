"""
csv_to_h5.py  —  raw ATC CSV  ->  trajectory H5
================================================

FIRST stage of the raw-data pipeline:

    raw ATC CSV  --[THIS FILE]-->  trajectory H5  --[h5_to_grid]-->  grid_cache

Clean, self-contained port of the original `scripts/csv2h5_batch.py` (pure numpy +
h5py, no pandas). Brought into the clean project so the raw->grid pipeline is fully
reproducible: anyone with the raw ATC CSVs can regenerate everything.

----------------------------------------------------------------------------
Raw ATC CSV schema (input)  —  8 comma-separated columns per row, one row per
                                pedestrian observation at one timestamp
----------------------------------------------------------------------------
  col 0  time      (s)    timestamp in seconds
  col 1  pid              pedestrian ID (same person recurs across timestamps;
                          not stored here — we only group rows by time)
  col 2  pos_x_mm  (mm)   x position in millimetres
  col 3  pos_y_mm  (mm)   y position in millimetres
  col 4  pos_z_mm  (mm)   z position (mm, height, unused)
  col 5  spd_mm    (mm/s) speed in mm/s
  col 6  ang       (rad)  motion direction angle
  col 7  face      (rad)  facing angle (unused)

We only take columns (0,2,3,5,6) = time, x, y, spd, ang.

----------------------------------------------------------------------------
Trajectory H5 schema (output)
----------------------------------------------------------------------------
  position : (N, 2) float32   (x, y) in metres  = pos_x_mm/1000, pos_y_mm/1000
  velocity : (N, 2) float32   (vx, vy) m/s      = spd/1000 * (cos ang, sin ang)
  index    : (M,) struct (time, start, stop, count)
             Groups rows by timestamp: position[start:stop] are all pedestrians
             at the same timestamp; count=1 (raw per-timestamp; accumulation
             into 1-second periods happens later in h5_to_grid).

             ! `time` label convention (verified row-by-row against the original
             pipeline's reference H5, see check_data_pipeline):
             record i's start:stop covers the rows of timestamp group i, but its
             time = the timestamp of group i+1 (last record = its own group's
             timestamp). I.e. `time` is the group's "end time" — the same
             convention as index_1.0s labelling windows by their end_time.
             It must be reproduced faithfully to stay equivalent to the
             existing data.

Note on IDs: the raw CSV has a pedestrian pid, but the macroscopic grid
representation models per-cell statistics and does not track individuals, so
pid is intentionally not written to the H5 (worth explaining to the supervisor:
the missing ID field in the H5 is deliberate).
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np

CHUNK_BYTES = 64 * 1024 * 1024                    # read 64 MiB of text per chunk
INDEX_DTYPE = np.dtype([("time", "f8"), ("start", "i8"), ("stop", "i8"), ("count", "i8")])
_USE_COLS = (0, 2, 3, 5, 6)                       # time, pos_x_mm, pos_y_mm, spd_mm, ang


def _row_transform(chunk):
    """(N,5) [time,x_mm,y_mm,spd_mm,ang] -> (time, pos(N,2) m, vel(N,2) m/s)."""
    t = chunk[:, 0]
    px = (chunk[:, 1] / 1000.0).astype(np.float32)
    py = (chunk[:, 2] / 1000.0).astype(np.float32)
    spd = (chunk[:, 3] / 1000.0).astype(np.float32)
    ang = chunk[:, 4].astype(np.float32)
    vx = (spd * np.cos(ang)).astype(np.float32)
    vy = (spd * np.sin(ang)).astype(np.float32)
    pos = np.stack([px, py], axis=1)
    vel = np.stack([vx, vy], axis=1)
    return t, pos, vel


def _count_rows(csv_path):
    n = 0
    with open(csv_path, "rb") as f:
        while True:
            buf = f.read(1 << 20)
            if not buf:
                break
            n += buf.count(b"\n")
    return n


def _iter_csv_chunks(csv_path, chunk_bytes=CHUNK_BYTES):
    """Yield (N,5) float64 chunks; chunk boundaries are cut at newlines (rows never split)."""
    leftover = b""
    with open(csv_path, "rb") as f:
        while True:
            block = f.read(chunk_bytes)
            if not block:
                break
            data = leftover + block
            cut = data.rfind(b"\n")
            if cut < 0:
                leftover = data
                continue
            payload, leftover = data[: cut + 1], data[cut + 1:]
            arr = np.loadtxt(BytesIO(payload), delimiter=",", usecols=_USE_COLS, dtype=np.float64)
            yield arr.reshape(1, -1) if arr.ndim == 1 else arr
    if leftover.strip():
        arr = np.loadtxt(BytesIO(leftover), delimiter=",", usecols=_USE_COLS, dtype=np.float64)
        yield arr.reshape(1, -1) if arr.ndim == 1 else arr


def convert(csv_path, h5_path):
    """Whole-file conversion: CSV -> trajectory H5 (position, velocity, index)."""
    csv_path, h5_path = Path(csv_path), Path(h5_path)
    n_rows = _count_rows(csv_path)
    with h5py.File(h5_path, "w") as h:
        position = h.create_dataset("position", shape=(n_rows, 2), dtype="f4", chunks=True)
        velocity = h.create_dataset("velocity", shape=(n_rows, 2), dtype="f4", chunks=True)
        index = h.create_dataset("index", shape=(n_rows,), dtype=INDEX_DTYPE,
                                 chunks=True, maxshape=(None,))
        write_pos, idx_count, last_time, run_start = 0, 0, np.nan, 0
        for chunk in _iter_csv_chunks(csv_path):
            t, pos, vel = _row_transform(chunk)
            n = len(chunk)
            position[write_pos:write_pos + n] = pos
            velocity[write_pos:write_pos + n] = vel
            if n:
                if np.isnan(last_time):
                    last_time = float(t[0])
                # boundary[i] = whether row i STARTS a new timestamp (time greater
                # than the previous row). Row 0 is compared against the previous
                # chunk's last time (last_time), so grouping is seamless across
                # chunks — verified by check_data_pipeline forcing tiny 64-byte
                # chunk splits.
                boundary = np.empty(n, dtype=bool)
                boundary[0] = t[0] > last_time
                if n > 1:
                    boundary[1:] = t[1:] > t[:-1]
                bidx = np.flatnonzero(boundary)
                if bidx.size:
                    # Each boundary row finalises the PREVIOUS group: row range
                    # [starts, stops) holds the previous group's rows, while the
                    # time label is t[bidx] = the NEW group's timestamp — the
                    # "end time" convention (see the schema notes in the module
                    # docstring; matches the original pipeline's reference H5
                    # record by record).
                    stops = write_pos + bidx
                    starts = np.empty_like(stops)
                    starts[0] = run_start                     # first group resumes from the previous chunk's break point
                    starts[1:] = stops[:-1]
                    need = idx_count + bidx.size
                    if need > index.shape[0]:
                        index.resize(max(need, index.shape[0] * 2), 0)
                    rec = np.empty(bidx.size, dtype=INDEX_DTYPE)
                    rec["time"], rec["start"], rec["stop"], rec["count"] = (
                        t[bidx].astype(np.float64), starts, stops, 1)
                    index[idx_count:idx_count + bidx.size] = rec
                    idx_count += bidx.size
                    last_time, run_start = float(t[bidx[-1]]), int(stops[-1])
            write_pos += n
        if write_pos > run_start:                              # finalise the last timestamp group
            if idx_count + 1 > index.shape[0]:
                index.resize(idx_count + 1, 0)
            rec = np.empty(1, dtype=INDEX_DTYPE)
            rec["time"], rec["start"], rec["stop"], rec["count"] = last_time, run_start, write_pos, 1
            index[idx_count] = rec[0]
            idx_count += 1
        index.resize(idx_count, 0)
    print(f"wrote {n_rows} rows -> {idx_count} timestamps : {h5_path}")


def validate_transform(csv_path, ref_h5_path, n=200, atol=1e-5):
    """Cross-check: position/velocity of the first n rows vs an existing trajectory H5."""
    chunk = next(_iter_csv_chunks(csv_path))[:n]
    _, pos, vel = _row_transform(chunk)
    with h5py.File(ref_h5_path, "r") as f:
        ref_pos, ref_vel = f["position"][:n], f["velocity"][:n]
    ep = float(np.max(np.abs(pos - ref_pos)))
    ev = float(np.max(np.abs(vel - ref_vel)))
    print(f"[validate] 前 {n} 行: position max|Δ|={ep:.3e}  velocity max|Δ|={ev:.3e}")
    print(f"[validate] {'通过 ✓' if ep <= atol and ev <= atol else '偏差 ✗'} (atol={atol})")
    return ep <= atol and ev <= atol


def main():
    ap = argparse.ArgumentParser(description="raw ATC CSV -> trajectory H5 (clean port)")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=None, help="输出轨迹 H5（不给则只验证）")
    ap.add_argument("--validate", default=None, help="给现有轨迹 H5 路径则对拍验证转换")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    if args.validate:
        validate_transform(args.csv, args.validate, args.n)
    elif args.out:
        convert(args.csv, args.out)
    else:
        ap.error("需要 --out 或 --validate")


if __name__ == "__main__":
    main()
