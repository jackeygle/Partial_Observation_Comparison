# Raw-data pipeline (CSV → grid_cache)

> Goal: reproduce `grid_cache` from the **raw ATC CSV**, with no reliance on
> pre-made intermediates. Both stages are clean numpy/h5py and **validated**
> against the existing data.

```
raw CSV ──[csv_to_h5.py]──► trajectory H5 (position/velocity/index) ──[h5_to_grid.py]──► grid_cache (N,4,H,W)
```

---

## Stage 1: `csv_to_h5.py` — CSV → trajectory H5

**Raw CSV (8 columns, each row = one pedestrian's observation at one instant)**

| Col | Field | Unit | Note |
|---|---|---|---|
| 0 | time | s | timestamp |
| 1 | **pid** | — | pedestrian ID (**not written to H5**, see note below) |
| 2 | pos_x_mm | mm | x position |
| 3 | pos_y_mm | mm | y position |
| 4 | pos_z_mm | mm | z height (unused) |
| 5 | spd_mm | mm/s | speed |
| 6 | ang | rad | direction of motion |
| 7 | face | rad | facing direction (unused) |

**Output trajectory H5**

| Field | Shape | Meaning |
|---|---|---|
| `position` | (N,2) f32 | (x,y) in metres = mm/1000 |
| `velocity` | (N,2) f32 | (vx,vy) in m/s = spd/1000 * (cos ang, sin ang) |
| `index` | (M,) struct (time,start,stop,count) | grouped by timestamp: `position[start:stop]` are all pedestrians at that instant; count=1 |

> **About the ID (the supervisor will ask)**: the CSV has `pid`, but our target is
> the **macroscopic field** (per-cell density/velocity statistics), not individual
> trajectory tracking, so the H5 **deliberately does not store pid**. Individual
> identity is aggregated away during gridding.

---

## Stage 2: `h5_to_grid.py` — trajectory H5 → grid_cache

Kernel-density-estimates the trajectory points into a 4-channel grid (triangular
kernel, corridor-subset rotated rectangle 36x12, 1m/cell, 1s/frame).

**Output grid_cache**

| Field | Shape | Meaning |
|---|---|---|
| `grid` | (N,4,H,W) f32 | channels [**density, vx, vy, vel_var**] |
| `time` | (N,) f64 | seconds of each frame |
| attrs | — | subset, period, resolution, kernel, origin, theta, shape |

- **density**: pedestrian density per cell (kernel estimate, normalised by the
  number of time steps accumulated that second, `count`)
- **vx, vy**: average velocity per cell (already rotated into the grid's
  coordinate frame)
- **vel_var**: weighted average of squared velocity deviation per cell (with the
  nnz/(nnz-1) unbiased correction)

> There is also an intermediate "windowing by period" step: the raw `index`
> (per-timestamp) -> `index_1.0s` (one frame per second, count = the number of
> time steps accumulated that second). The existing trajectory H5 already caches
> `index_1.0s`, and it is read directly.

---

## How to run

```bash
module load scicomp-pytorch-env/2026.1

# Stage 1: CSV -> trajectory H5
python3 csv_to_h5.py --csv /scratch/work/zhangx29/ATC/atc-20121028.csv \
                     --out /tmp/atc-20121028.h5

# Stage 2: trajectory H5 -> grid_cache
python3 h5_to_grid.py --traj-h5 /tmp/atc-20121028.h5 \
                      --out /tmp/atc-20121028_corridor_1.0s.h5 --subset corridor
```

## Validation (proves the clean reimplementation is faithful)

```bash
# Stage 1: conversion result vs. the existing trajectory H5
python3 csv_to_h5.py --csv .../atc-20121028.csv --validate .../Sundays/atc-20121028.h5
#   position max|Δ| = 0.000e+00   velocity max|Δ| = 0.000e+00     ✓

# Stage 2: gridded result vs. the existing grid_cache
python3 h5_to_grid.py --traj-h5 .../Sundays/atc-20121028.h5 \
                      --validate .../grid_cache/atc-20121028_corridor_1.0s.h5
#   density/vx/vy/var max|Δ| ~ 6e-6  (float32 rounding level)      ✓
```

-> Both stages reproduce to float rounding precision, proving this clean pipeline
is equivalent to the original one.

**More thorough validation** (hand-computed synthetic micro-examples + full
row/index comparison + end-to-end CSV->H5->grid chaining):

```bash
python3 checks/check_data_pipeline.py                  # runs all three layers (defaults to day 20121028)
python3 checks/check_data_pipeline.py --synthetic-only # does not need real data
```

> Two **inherent conventions** confirmed during validation (the original pipeline
> works this way, and reproducing it is what makes the result equivalent to the
> existing data):
> (1) the raw `index`'s recorded `time` is that group's "end time" (= the next
> group's timestamp; the last group uses its own time);
> (2) `index_1.0s`'s first window's `count` double-counts the very first raw
> record (matches the original cache field for field).
