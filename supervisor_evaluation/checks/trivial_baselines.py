"""Two reference predictions that need no model, on the blind cells of the common frames.

  all zero       : every blind cell is predicted 0. The velocity of an empty cell is
                   defined as 0 and 79% of the scored blind cells are empty, so this is
                   a strong reference on the pooled metric, not a straw man.
  carry-forward  : every blind cell keeps the value it was last observed with. This is
                   `observation_model.fill_missing_state(..., 'prev')`, i.e. the state the
                   4DVarNet solver starts from, so it also measures what the solver adds.

Support: the frames of the unified uncertainty protocol, lo=500 and
hi=min(n-1, floor(n/200)*200), blind walkable cells, all four channels --
34,853,660 cells per channel over the seven test days. No GPU and no training.

  python3 -m supervisor_evaluation.checks.trivial_baselines
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

RAW = "supervisor_evaluation/outputs/full/raw"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW, help="directory holding enkf_observations/ and enkf_exports/")
    ap.add_argument("--out", default="supervisor_evaluation/outputs/full/trivial_baselines.json")
    a = ap.parse_args()

    se = {"zero": np.zeros(4), "carry_forward": np.zeros(4)}
    n_cells = 0.0
    per_day = []
    for f in sorted(glob.glob(os.path.join(a.raw, "enkf_observations", "obs_atc-*.npz"))):
        day = os.path.basename(f)[len("obs_"):-len(".npz")]
        z = np.load(f)
        x_true, omega, x0 = z["X_true"], z["Omega"], z["X0"]
        n = x_true.shape[0]
        lo, hi = 500, min(n - 1, (n // 200) * 200)
        truth = x_true[lo:hi].astype(np.float64)
        carried = x0[lo:hi].astype(np.float64)
        walkable = np.load(os.path.join(a.raw, "enkf_exports", f"{day}.npz"))["walkable"]
        blind = ((~omega[lo:hi]) & walkable[None]).astype(np.float64)

        se["zero"] += np.einsum("tchw,thw->c", truth ** 2, blind)
        se["carry_forward"] += np.einsum("tchw,thw->c", (carried - truth) ** 2, blind)
        n_cells += blind.sum()
        per_day.append({"day": day, "frames": [lo, hi], "blind_cells": int(blind.sum())})
        print(f"[{day}] frames [{lo}, {hi}), {int(blind.sum())} blind cells/channel", flush=True)

    doc = {
        "support": "blind walkable cells, lo=500 hi=min(n-1, floor(n/200)*200), as the unified uncertainty protocol",
        "scored_cells_per_channel": int(n_cells),
        "per_day": per_day,
        "results": {},
    }
    for name, total in se.items():
        channel = np.sqrt(total / n_cells)
        doc["results"][name] = {
            "pooled_rmse": float(np.sqrt(total.sum() / (n_cells * 4))),
            "channel_rmse": channel.tolist(),
        }
        print(f"{name:14s} pooled {doc['results'][name]['pooled_rmse']:.5f}  "
              f"per channel {np.round(channel, 5).tolist()}")

    with open(a.out, "w") as fh:
        json.dump(doc, fh, indent=1)
    print(f"[out] {a.out}")


if __name__ == "__main__":
    main()
