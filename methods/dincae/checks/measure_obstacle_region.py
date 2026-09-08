"""
measure_obstacle_region.py — does full-field supervision teach DINCAE "obstacle = 0"?

2026-09-07 meeting: the advisor pointed out that within the physically
non-walkable region (walls/obstacles, ~`~StateStats().valid`), density and
velocity are physically always 0, and a model trained on the **full** grid
(EnKF, 4DVarNet, Senseiver) should — and does — learn to output 0 there for
free. DINCAE's information form only ever supervises `channel_valid & walkable`
cells (`methods.dincae.encoding.encode_target`, `full_field=False`), so its
obstacle-region behaviour was never directly checked.

This script isolates exactly that region and compares two already-trained
checkpoints:

  runs/dincae_full   baseline, information-form supervision (obstacle cells
                     excluded from the loss entirely)
  runs/dincae_ff     `--full-field-loss` ablation (obstacle cells supervised
                     against physical 0, same as the other three methods)

It reuses `evaluate.py`'s `load_models`/`predict_day`/`clip_bounds` verbatim
so the reconstruction pipeline is identical to the headline numbers — only the
region being scored differs (`~stats.valid` here, vs `channel_valid & valid`
there).

Usage (GPU node):
    srun -p gpu-debug --gres=gpu:1 -t 00:14:00 bash -c \
      'source sbatch/_env.sh && cd methods/dincae && \
       python3 -u -m methods.dincae.checks.measure_obstacle_region --days 2'
    # full 7-day test split:
    sbatch methods/dincae/sbatch/submit_obstacle_region.sbatch
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from crowdcore import observation_model as om                                   # noqa: E402
from methods.dincae.state import CHANNELS, NCH, StateStats                       # noqa: E402
from methods.dincae.checks.evaluate import load_models, predict_day, clip_bounds  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def score(run_dir, ckpt, stats, files, dev, frames, batch):
    # A specific checkpoint, never the whole ckpt_*.pt glob: the glob would
    # average every checkpoint's outputs, which is a different configuration
    # from the one behind every reported DINCAE number (see
    # evaluate.PUBLISHED_CKPT) and would make this ablation incomparable with
    # the main table.
    models, epochs, paths = load_models(run_dir, os.path.join(run_dir, ckpt), dev)
    obstacle = ~stats.valid                       # (H,W) bool, fixed physical mask
    n_obstacle = int(obstacle.sum())

    se = np.zeros(NCH); n = np.zeros(NCH, dtype=np.int64)
    se_walk = np.zeros(NCH); n_walk = np.zeros(NCH, dtype=np.int64)
    nonzero_truth = np.zeros(NCH); n_truth = np.zeros(NCH, dtype=np.int64)

    for fp in files:
        Xt, rec, _, _, _ = predict_day(models, stats, fp, dev, frames, batch)
        rec_c = clip_bounds(rec)
        obs_b = np.broadcast_to(obstacle, Xt.shape[2:])
        walk_b = ~obs_b
        for c in range(NCH):
            d2 = (rec_c[:, c] - Xt[:, c]) ** 2
            se[c] += float(d2[:, obs_b].sum()); n[c] += int(obs_b.sum()) * Xt.shape[0]
            se_walk[c] += float(d2[:, walk_b].sum()); n_walk[c] += int(walk_b.sum()) * Xt.shape[0]
            nonzero_truth[c] += float((Xt[:, c][:, obs_b] != 0).sum())
            n_truth[c] += int(obs_b.sum()) * Xt.shape[0]
        del Xt, rec, rec_c

    return {
        "run_dir": run_dir, "checkpoints": [os.path.basename(p) for p in paths], "epochs": epochs,
        "n_obstacle_cells": n_obstacle,
        "obstacle_mse": {CHANNELS[c]: se[c] / n[c] for c in range(NCH)},
        "walkable_mse": {CHANNELS[c]: se_walk[c] / n_walk[c] for c in range(NCH)},
        "obstacle_truth_nonzero_frac": {CHANNELS[c]: nonzero_truth[c] / n_truth[c] for c in range(NCH)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", default=os.path.join(ROOT, "runs", "dincae_full"))
    ap.add_argument("--fullfield-dir", default=os.path.join(ROOT, "runs", "dincae_ff"))
    # Defaults are the checkpoints the reported numbers use, so this ablation is
    # comparable with the main table: epoch 70 for the baseline (see
    # evaluate.PUBLISHED_CKPT / check_outputs/eval_single_00070), epoch 140 for
    # the full-field run (check_outputs/eval_ff_00140).
    ap.add_argument("--baseline-ckpt", default="ckpt_00070.pt")
    ap.add_argument("--fullfield-ckpt", default="ckpt_00140.pt")
    ap.add_argument("--split", default="test", choices=["test", "valid"])
    ap.add_argument("--days", type=int, default=0, help="use only the first N days (debug)")
    ap.add_argument("--frames", type=int, default=0, help="use only the first N frames per day (debug)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default=os.path.join(ROOT, "check_outputs", "eval", "obstacle_region.json"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cpu":
        print("!! no GPU -- do not run torch on the login node (see README)", flush=True)

    stats = StateStats()
    files = om.split_files(args.split)
    if args.days:
        files = files[: args.days]
    print(f"{args.split} split: {len(files)} days, "
          f"{int((~stats.valid).sum())}/{stats.valid.size} cells are physically non-walkable "
          f"({100 * (~stats.valid).mean():.1f}%)")

    result = {}
    for tag, run_dir, ckpt in (("baseline_info_form", args.baseline_dir, args.baseline_ckpt),
                               ("full_field_ablation", args.fullfield_dir, args.fullfield_ckpt)):
        print(f"\n=== {tag}  ({run_dir}  {ckpt}) ===")
        r = score(run_dir, ckpt, stats, files, dev, args.frames, args.batch)
        result[tag] = r
        print(f"  checkpoint(s): {r['checkpoints']}  epochs {r['epochs']}")
        print(f"  {'channel':<10}{'obstacle MSE':>14}{'walkable MSE':>14}{'truth nonzero%':>16}")
        for c in CHANNELS:
            print(f"  {c:<10}{r['obstacle_mse'][c]:>14.5f}{r['walkable_mse'][c]:>14.5f}"
                  f"{100 * r['obstacle_truth_nonzero_frac'][c]:>15.2f}%")

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {args.out}")

    print("\n=== summary: obstacle-region MSE, baseline vs full-field ===")
    for c in CHANNELS:
        a = result["baseline_info_form"]["obstacle_mse"][c]
        b = result["full_field_ablation"]["obstacle_mse"][c]
        ratio = a / b if b > 0 else float("inf")
        print(f"  {c:<10} baseline {a:.5f}   full-field {b:.5f}   baseline/full-field = {ratio:.2f}x")


if __name__ == "__main__":
    main()
