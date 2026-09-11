"""
select_checkpoint.py — pick 4DVarNet's checkpoint on the VALIDATION split

Why this exists. `varnet_best.pt` cannot be used to choose which epoch to report:
train.py selects it on `Xe`, the first --n-eval windows of --split, and --split
defaults to **train**. That is model selection on training data, and on the
emptiest part of it, since the first windows of a recording day are near-empty
(the density there is about 1/6.9 of the full-day mean).

It also lands different runs at different points of the paper's §3.4 curriculum.
Measured across the hidden=32 arms before the 96/5 retrain:

    arm     best epoch    solver iterations at that checkpoint
    mse5      78-143                  20
    vsb0       62-93                  15-20
    aug0       16-40                  5-10

Those checkpoints are not a loss-controlled comparison: aug0's reported number
came from a model evaluated at a quarter of the solver depth and a fifth of the
training of mse5's. This script removes both problems at once —

  * candidates are restricted to --min-epoch and later, where the curriculum has
    reached its final stage, so every reported model runs at the SAME number of
    solver iterations (the paper's §3.3/3.4 "typically, from 5 to 20" endpoint).
    The count is asserted, not assumed.
  * they are scored on --split valid, never on train and never on test.

This also puts 4DVarNet on the same footing as the other two methods, which was
the real inconsistency: Senseiver selects best.pt on `split_files("valid")`
(train.py:171-187) and DINCAE has checks/select_checkpoint.py doing the same.
4DVarNet was the only one of the three selecting on the training split.

Scoring scope is `walkable` — blind cells inside the walkable region, all four
channels, clipped to the physical bounds. That is compare5's main convention, so
the epoch chosen here is the epoch that convention would have chosen.

Usage (GPU node):
    python3 -m methods.varnet.checks.select_checkpoint --run-dir runs/varnet_mse5_h96_s0
    python3 -m methods.varnet.checks.select_checkpoint --run-dir runs/varnet_aug0_h96_s3 --days 3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import torch

from crowdcore import navigation as nav
from crowdcore import observation_model as om
from methods.varnet.train import build_windows
from methods.varnet.checks.eval_comprehensive import clip_bounds
from methods.varnet.checks.model_io import load_solver

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The curriculum's final stage. Every reported checkpoint must solve with this many
# iterations, or the arms are not comparable -- see the module docstring.
FINAL_N_ITER = 20


def candidates(run_dir, min_epoch):
    """The ckpt_<epoch>.pt snapshots at or after min_epoch, sorted by epoch."""
    out = []
    for p in sorted(glob.glob(os.path.join(run_dir, "ckpt_*.pt"))):
        m = re.search(r"ckpt_(\d+)\.pt$", os.path.basename(p))
        if m and int(m.group(1)) >= min_epoch:
            out.append((int(m.group(1)), p))
    return sorted(out)


def score_day(solvers, fp, args_by_ckpt, walk, dev, batch):
    """Squared error and cell count on blind-walkable cells, for every checkpoint.

    The day is loaded ONCE and every checkpoint is scored on it, so all of them see
    byte-identical observations -- with a per-checkpoint reload the robot routes would
    be regenerated each time and any difference between checkpoints would be partly
    that regeneration.
    """
    a0 = next(iter(args_by_ckpt.values()))
    # Rebuild the observations exactly as train.py:262-272 did, or the model is scored on a
    # different realisation of the robot routes and sensor noise than it was trained against.
    data_seed = a0["seed"] if a0.get("data_seed") is None else a0["data_seed"]
    X, Y, M, X0 = build_windows([fp], a0["dT"], a0["sensing_range"], a0["num_agents"],
                                data_seed, 1,
                                add_noise=False if a0.get("no_noise") else None,
                                obs_every_k=a0.get("obs_every_k"))
    blind = (M < 0.5).numpy()                      # (N,C,dT,H,W); the mask is 1.0/0.0 float
    # walk is (H,W): broadcast over windows, channels and frames
    sel = blind & walk[None, None, None]
    Xn = X.numpy()
    res = {}
    for ep, sol in solvers.items():
        sol.eval()
        with torch.enable_grad():                  # the solver differentiates J internally
            recs = [sol(*(t[i:i + batch].to(dev) for t in (X0, Y, M))).detach().cpu().numpy()
                    for i in range(0, X.shape[0], batch)]
        # clip_bounds indexes axis 1 as the channel axis, which is already the channel axis
        # for these (N,C,dT,H,W) windows, so it applies unchanged.
        xr = clip_bounds(np.concatenate(recs, 0))
        d2 = (xr - Xn) ** 2
        res[ep] = (float(d2[sel].sum()), int(sel.sum()))
        del recs, xr, d2
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="e.g. runs/varnet_mse5_h96_s0")
    ap.add_argument("--min-epoch", type=int, default=75,
                    help="ignore snapshots before this epoch. Default 75 = the epoch the "
                         "§3.4 curriculum reaches its final 20 iterations, so every "
                         "candidate solves at the same depth.")
    ap.add_argument("--split", default="valid", choices=["valid", "train", "test"],
                    help="ALWAYS valid for a reported selection; the others are for diagnostics.")
    ap.add_argument("--days", type=int, default=0, help="0 = every day of the split")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.split == "test":
        print("[warn] --split test is TEST-SET SELECTION. Never report a checkpoint chosen "
              "this way; this option exists only to measure how much the choice matters.",
              flush=True)

    run_dir = args.run_dir if os.path.isabs(args.run_dir) else os.path.join(ROOT, args.run_dir)
    if not os.path.isdir(run_dir):
        # Without this the next check reports "no snapshots at or after epoch N" and names a
        # path that does not exist, which reads as "training has not got that far" rather than
        # "you pointed me at nothing". Relative paths resolve against methods/varnet/, so a
        # repo-root-relative path like methods/varnet/runs/X silently doubles the prefix.
        raise SystemExit(f"no such directory: {run_dir}\n"
                         f"--run-dir is relative to {ROOT} (e.g. runs/varnet_mse5_h96_s0), "
                         f"or give an absolute path.")
    cands = candidates(run_dir, args.min_epoch)
    if not cands:
        raise SystemExit(
            f"no ckpt_*.pt at or after epoch {args.min_epoch} in {run_dir}.\n"
            "Runs trained before --ckpt-every was added (2026-09-09) only have "
            "varnet_best.pt / varnet_last.pt and cannot be selected on; they have to be "
            "reported from varnet_last.pt, which at least is the same epoch for every run.")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    solvers, args_by_ckpt = {}, {}
    for ep, p in cands:
        sol, a, ck = load_solver(p, dev)
        n_it = int(ck.get("n_iter_eff", -1))
        if n_it != FINAL_N_ITER:
            raise SystemExit(
                f"{os.path.basename(p)} solves with {n_it} iterations, not {FINAL_N_ITER}.\n"
                "Comparing arms across different solver depths is the exact problem this "
                f"script exists to remove. Raise --min-epoch past the curriculum stage that "
                "ends at this epoch, or fix --iter-schedule.")
        solvers[ep] = sol
        args_by_ckpt[p] = a

    walk = nav.build_valid_mask_from_config()
    files = om.split_files(args.split)
    if args.days:
        files = files[:args.days]
    print(f"[select] {os.path.basename(run_dir)}: {len(cands)} candidates "
          f"(epochs {cands[0][0]}..{cands[-1][0]}, all at {FINAL_N_ITER} iterations) | "
          f"{len(files)} {args.split} days | walkable {int(walk.sum())}/{walk.size} | {dev}",
          flush=True)

    se = {ep: 0.0 for ep, _ in cands}
    n = {ep: 0 for ep, _ in cands}
    for fp in files:
        stem = os.path.basename(fp).split("_")[0]
        day = score_day(solvers, fp, args_by_ckpt, walk, dev, args.batch)
        for ep, (s, c) in day.items():
            se[ep] += s
            n[ep] += c
        best_here = min(day, key=lambda e: day[e][0] / max(day[e][1], 1))
        print(f"  {stem}: best epoch {best_here} "
              f"({day[best_here][0] / max(day[best_here][1], 1):.6f})", flush=True)

    mse = {ep: se[ep] / max(n[ep], 1) for ep in se}
    order = sorted(mse, key=lambda e: mse[e])
    win = order[0]
    print(f"\n{'epoch':>7}  {'walkable blind MSE':>19}  {'RMSE':>8}")
    for ep in sorted(mse):
        mark = "  <-- selected" if ep == win else ""
        print(f"{ep:>7}  {mse[ep]:>19.6f}  {np.sqrt(mse[ep]):>8.4f}{mark}")
    spread = max(mse.values()) - min(mse.values())
    print(f"\n[select] epoch {win}  MSE {mse[win]:.6f}  RMSE {np.sqrt(mse[win]):.4f}")
    print(f"[select] spread across the {len(mse)} candidates: {spread:.6f} "
          f"({100 * spread / mse[win]:.1f}% of the winner) -- if this is small, the choice "
          f"of epoch barely matters and any of them would do.", flush=True)

    out = args.out or os.path.join(run_dir, "select_valid.json")
    with open(out, "w") as f:
        json.dump({"run_dir": run_dir, "split": args.split, "min_epoch": args.min_epoch,
                   "n_iter": FINAL_N_ITER, "days": [os.path.basename(x) for x in files],
                   "scope": "walkable (blind cells inside the walkable region, clipped)",
                   "mse_by_epoch": {str(k): v for k, v in mse.items()},
                   "selected_epoch": win, "selected_mse": mse[win],
                   "selected_ckpt": f"ckpt_{win:05d}.pt"}, f, indent=2)
    print(f"[select] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
