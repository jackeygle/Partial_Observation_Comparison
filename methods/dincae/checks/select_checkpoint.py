"""
select_checkpoint.py — pick DINCAE's checkpoint on the VALIDATION split

Why this exists: the reported DINCAE checkpoint (`ckpt_00070.pt`) was described
as "best on its own validation set", but the only validation evaluation on disk
covered epochs 3-5, and the epoch sweep that does exist was run on the **test**
split -- where epoch 120 actually scores lower than epoch 70. Choosing a
checkpoint by test performance is test-set leakage, so this script re-does the
selection properly: every checkpoint of a run, scored on `--split valid`, under
the scope the comparison actually reports.

Scoring scope is `walkable` -- blind cells inside the walkable region, all four
channels, no `channel_valid` restriction. That matches how the models are now
compared: trained on the full field, scored on the physical domain, with the
map-obstacle region excluded because its truth is a fixed known zero.

Usage (GPU node):
    python3 -m methods.dincae.checks.select_checkpoint --run-dir runs/dincae_ff
    python3 -m methods.dincae.checks.select_checkpoint --run-dir runs/dincae_full
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

from crowdcore import observation_model as om
from methods.dincae.state import CHANNELS, NCH, StateStats
from methods.dincae.checks.evaluate import load_models, predict_day, clip_bounds

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def score_ckpt(path, stats, files, dev, frames, batch):
    """Blind MSE over walkable cells, pooled across days and channels."""
    models, epochs, _ = load_models(os.path.dirname(path), path, dev)
    walk = stats.valid
    se = n = 0.0
    for fp in files:
        Xt, rec, _, _, M = predict_day(models, stats, fp, dev, frames, batch)
        rec = clip_bounds(rec)
        blind = ~M                                        # (T,NCH,H,W)
        sel = blind & walk[None, None]
        d2 = (rec - Xt) ** 2
        se += float(d2[sel].sum()); n += int(sel.sum())
        del Xt, rec, M, d2, sel
    return se / n, epochs[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.path.join(ROOT, "runs", "dincae_ff"))
    ap.add_argument("--split", default="valid", choices=["valid", "test"])
    ap.add_argument("--days", type=int, default=0, help="0 = the whole split")
    ap.add_argument("--frames", type=int, default=0, help="0 = the whole day")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cpu":
        print("!! no GPU -- do not run this on the login node", flush=True)
    stats = StateStats()
    files = om.split_files(args.split)
    if args.days:
        files = files[: args.days]
    ckpts = sorted(glob.glob(os.path.join(args.run_dir, "ckpt_*.pt")))
    print(f"[select] {args.run_dir}: {len(ckpts)} checkpoints, "
          f"{len(files)} {args.split} days, scope=walkable blind\n", flush=True)

    rows = []
    for p in ckpts:
        mse, ep = score_ckpt(p, stats, files, dev, args.frames, args.batch)
        rows.append({"epoch": ep, "ckpt": os.path.basename(p), "walkable_blind_mse": mse})
        print(f"  epoch {ep:>4}  {os.path.basename(p):<16} {mse:.6f}", flush=True)

    best = min(rows, key=lambda r: r["walkable_blind_mse"])
    print(f"\n[best] epoch {best['epoch']} ({best['ckpt']}) "
          f"walkable blind MSE {best['walkable_blind_mse']:.6f}", flush=True)

    out = args.out or os.path.join(ROOT, "check_outputs", "eval",
                                   f"select_{os.path.basename(args.run_dir)}_{args.split}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"run_dir": args.run_dir, "split": args.split, "scope": "walkable blind",
                   "n_days": len(files), "rows": rows, "best": best}, f, indent=2)
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
