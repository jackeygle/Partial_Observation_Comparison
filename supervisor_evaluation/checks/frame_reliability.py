"""Per-frame reliability of the gridded ATC days (route 2, option 1: flag, do not regrid).

    python3 -m supervisor_evaluation.checks.frame_reliability [--out DIR]

The 1-s windowing inherited from Partial_Observation (dataset.GridFromH5Dataset, ported as
crowdcore.data.h5_to_grid.build_period_index) advances its clock by one period per raw tracker
timestamp. After a gap in the raw records it therefore emits one window per raw timestamp until
its clock catches up: the gap is filled with ~0.05-s snapshots taken after it, under labels that
lag the real time. A grid frame is RELIABLE when the raw timestamps it was built from all lie
inside its own labelled second, [label - 1 s - TOL, label + TOL) (the label is the window's end
time; a window holds the raw timestamps before it).

Frames with only a few raw timestamps (sparse tracker output, e.g. 2013-08-11) are correctly
timed and the grid divides by the timestamp count, so they are not flagged; their count is
reported only.

For Sundays the windows are the stored `index_1.0s` (written by Partial_Observation and used to
build data/grid_cache); for Wednesdays they are rebuilt with build_period_index, exactly as
h5_to_grid did. Each day's window count and labels are checked against its grid file.
Writes <out>/<day>.npy (bool per frame) and <out>/summary.json.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os

import h5py
import numpy as np

from crowdcore.data.h5_to_grid import build_period_index

TOL = 0.1
SUN = "/scratch/work/zhangx29/data/ATC/Sundays"
WED = "/scratch/work/zhangx29/data_all92/ATC/Wednesdays"
GRID = "/scratch/work/zhangx29/data_all92/grid_cache"


def windows_for(traj):
    with h5py.File(traj, "r") as h:
        raw = h["index"][:]
        win = h["index_1.0s"][:] if "index_1.0s" in h else build_period_index(raw, 1.0)
    return raw, win


def reliability(raw, win):
    rt = raw["time"].astype(np.float64)
    rs = raw["start"].astype(np.int64)
    if not (np.diff(rs) > 0).all():                      # one raw entry per start -> searchsorted is exact
        raise SystemExit("raw index starts are not strictly increasing")
    ws = win["start"].astype(np.int64); we = win["stop"].astype(np.int64)
    lab = win["time"].astype(np.float64)
    lo = np.searchsorted(rs, ws, side="left")            # raw entries with start in [ws, we)
    hi = np.searchsorted(rs, we, side="left")
    n_raw = hi - lo
    cnt = n_raw.copy(); cnt[0] += raw["count"][0]          # the windowing counts raw entry 0 twice
    if not (cnt == win["count"]).all():
        raise SystemExit("raw entries per window do not match the window counts")
    # raw times are sorted, so a window's first / last raw entry give its min / max time
    tmin = np.full(len(win), np.nan); tmax = np.full(len(win), np.nan)
    nz = n_raw > 0
    tmin[nz] = rt[lo[nz]]
    tmax[nz] = rt[hi[nz] - 1]
    inside = nz & (tmin >= lab - 1.0 - TOL) & (tmax < lab + TOL)
    return inside, n_raw, tmax - lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/work/zhangx29/data_all92/frame_reliability")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    summary = []
    trajs = sorted(glob.glob(f"{SUN}/atc-*.h5") + glob.glob(f"{WED}/atc-*.h5"),
                   key=lambda p: os.path.basename(p))
    for traj in trajs:
        day = os.path.basename(traj)[4:12]
        wd = datetime.date(int(day[:4]), int(day[4:6]), int(day[6:])).strftime("%a")
        raw, win = windows_for(traj)
        with h5py.File(f"{GRID}/atc-{day}_corridor_1.0s.h5", "r") as g:
            gt = g["time"][:]
        if len(gt) != len(win) or not np.allclose(gt, win["time"].astype(np.float64), atol=1e-3):
            raise SystemExit(f"{day}: grid frames ({len(gt)}) do not match the windows ({len(win)})")
        ok, n_raw, lag = reliability(raw, win)
        np.save(f"{a.out}/atc-{day}.npy", ok)
        bad = ~ok
        runs = np.flatnonzero(np.diff(np.r_[0, bad.astype(np.int8), 0]) != 0).reshape(-1, 2)
        summary.append({"day": day, "weekday": wd, "frames": int(len(ok)), "unreliable": int(bad.sum()),
                        "unreliable_share": float(bad.mean()), "unreliable_runs": int(len(runs)),
                        "longest_run_s": int((runs[:, 1] - runs[:, 0]).max()) if len(runs) else 0,
                        "frames_under_5_raw": int((n_raw < 5).sum()),
                        "max_data_after_label_s": float(np.nanmax(lag))})
        s = summary[-1]
        print(f"{day} {wd} frames {s['frames']:6d} unreliable {s['unreliable']:5d} ({s['unreliable_share']*100:5.2f}%) "
              f"runs {s['unreliable_runs']:3d} longest {s['longest_run_s']:5d} s  max data-after-label {s['max_data_after_label_s']:7.1f} s"
              f"  frames<5 raw {s['frames_under_5_raw']}", flush=True)
    json.dump(summary, open(f"{a.out}/summary.json", "w"), indent=1)
    tot = sum(s["frames"] for s in summary); bad = sum(s["unreliable"] for s in summary)
    print(f"[total] {bad} of {tot} frames unreliable ({bad/tot*100:.3f}%) on "
          f"{sum(s['unreliable'] > 0 for s in summary)} of {len(summary)} days", flush=True)


if __name__ == "__main__":
    main()
