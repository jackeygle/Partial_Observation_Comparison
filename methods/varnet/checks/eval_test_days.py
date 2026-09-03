"""
eval_test_days.py  —  cross-date evaluation of the trained 4DVarNet
====================================================================

Loads a trained checkpoint and evaluates it on the HELD-OUT test split (7 days
that never appeared in training). For each test day it regenerates the same
multi-robot partial observations (same config: robots, range, line-of-sight,
temporal sparsity, noise), runs the solver over every dT-window, and reports:

    - blind-zone MSE : reconstruction error on the UNOBSERVED cells (the real task)
    - full-state MSE : reconstruction error over the WHOLE state (paper R-score)


Aggregated across the 7 days (mean +/- std) and printed per-day. Writes
check_outputs/eval/test_metrics.json for provenance (every number here is a
real solver output on real held-out data).

Run on a GPU node (see sbatch/submit_eval.sbatch).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import numpy as np
import torch

import config
import navigation as nav
import observation_model as om
from model_io import load_solver


def windows(arr, dT):
    return om.to_windows(arr, dT)


def clip_bounds(x):
    """Same physical bounds the EnKF baseline applies in _clip_bounds:
    density [0,5], vx/vy [-5,5], var [0,2]. Applied to the reconstruction so both
    methods live under identical bounds (fair comparison; enabled with --clip)."""
    x = x.clone()
    x[:, 0].clamp_(0, 5)          # density
    x[:, 1].clamp_(-5, 5)         # vx
    x[:, 2].clamp_(-5, 5)         # vy
    x[:, 3].clamp_(0, 2)          # var  (EnKF's bound; true var can exceed 2)
    return x


def eval_day(solver, day, a, dev, batch=32, frames=0, clip=False):
    """Return (blind_mse, full_mse, base_mse, n_windows) for one day.

    frames>0 restricts to the first `frames` frames of the day (to score 4DVarNet on
    exactly the same frames the EnKF baseline was run on, for a matched comparison)."""
    X, _ = om.load_state(day)
    if frames > 0:
        X = np.asarray(X)[:frames]
    valid = nav.build_valid_mask_from_config(X)
    # obs_every_k must match what the checkpoint was TRAINED with, otherwise the model is
    # scored on an observation pattern it never saw. It is read from the checkpoint's own
    # args (recorded by train_varnet.py's --obs-every-k), falling back to config for older
    # checkpoints that predate the flag.
    k_eff = a.get("obs_every_k") or config.get("observation", "obs_every_k")
    out = om.generate_observations(X, add_noise=not a.get("no_noise", False), valid_mask=valid,
                                   obs_every_k=k_eff)
    X0 = om.fill_missing_state(out["Y"], out["Omega_c"], method=config.get("observation", "init_method"))
    Xw = torch.from_numpy(windows(np.asarray(X), a["dT"])).float()
    Yw = torch.from_numpy(windows(out["Y"], a["dT"])).float()
    Mw = torch.from_numpy(windows(out["Omega_c"].astype(np.float32), a["dT"])).float()
    X0w = torch.from_numpy(windows(X0, a["dT"])).float()

    se_blind = n_blind = se_full = n_full = 0.0
    # Time ONLY the solver's forward pass, the same convention the EnKF driver uses (its timer
    # covers the filter loop, not the gridding or the robot simulation). Data preparation is
    # shared by both methods, so charging it to either would only add the same constant.
    solve_s = 0.0
    with torch.enable_grad():                                  # solver uses autograd internally
        for i in range(0, Xw.shape[0], batch):
            xb = Xw[i:i+batch].to(dev); yb = Yw[i:i+batch].to(dev)
            mb = Mw[i:i+batch].to(dev); x0b = X0w[i:i+batch].to(dev)
            if dev.type == "cuda":
                torch.cuda.synchronize()             # CUDA is async: without this we would
            t0 = time.perf_counter()                 # time the launch, not the computation
            xr = solver(x0b, yb, mb).detach()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            solve_s += time.perf_counter() - t0
            if clip:
                xr = clip_bounds(xr)                 # same physical bounds as EnKF
            unobs = mb < 0.5
            se_blind += float(((xr - xb)[unobs] ** 2).sum()); n_blind += int(unobs.sum())
            se_full += float(((xr - xb) ** 2).sum()); n_full += xr.numel()
    return se_blind / n_blind, se_full / n_full, Xw.shape[0], solve_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/varnet_b0_k1/varnet_best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--frames", type=int, default=0,
                    help=">0 restricts each day to its first N frames (to match the EnKF subset)")
    ap.add_argument("--tag", default="", help="suffix for the output json (e.g. _matched)")
    ap.add_argument("--no-clip", dest="clip", action="store_false",
                    help="disable the EnKF-consistent clip (default: clip ON for a fair comparison)")
    ap.set_defaults(clip=True)                           # clip to EnKF's bounds by default (fairness)
    ap.add_argument("--outdir", default="check_outputs/eval")
    ap.add_argument("--n-iter", type=int, default=0,
                    help="override the solver's iteration count (0 = use the trained one). "
                         "Only for sweeping how the reconstruction depends on it — the "
                         "reported numbers always use the trained count.")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    solver, a, ck = load_solver(args.ckpt, dev)
    if args.n_iter:
        print(f"[override] n_iter {solver.n_iter} -> {args.n_iter}", flush=True)
        solver.n_iter = args.n_iter
    # n_iter from the SOLVER, not from args: with --iter-schedule the trained count differs.
    print(f"[model] {args.ckpt}  epoch={ck.get('epoch')}  dT={a['dT']}  "
          f"n_iter={solver.n_iter}  device={dev}", flush=True)

    days = om.split_files(args.split)
    print(f"[data] {len(days)} {args.split} days (held out)\n", flush=True)
    print(f"{'day':16s} {'blindMSE':>10} {'blindRMSE':>11} {'fullMSE':>10} {'windows':>8} "
          f"{'ms/frame':>10}", flush=True)
    print("-" * 72, flush=True)
    per_day = []
    for d in days:
        stem = os.path.basename(d).split("_")[0]
        bl, fu, nw, sec = eval_day(solver, d, a, dev, args.batch, args.frames, args.clip)
        ms = sec / (nw * a["dT"]) * 1000
        print(f"{stem:16s} {bl:>10.4f} {bl**0.5:>11.4f} {fu:>10.4f} {nw:>8} {ms:>10.3f}", flush=True)
        per_day.append({"day": stem, "blind_mse": bl, "full_mse": fu, "n_windows": nw,
                        "solve_s": sec, "per_frame_ms": ms})

    bl = np.array([p["blind_mse"] for p in per_day])
    fu = np.array([p["full_mse"] for p in per_day])
    # RMSE per day first, then mean/std over days — sqrt of a mean is not the mean of sqrts
    rm = np.sqrt(bl)
    print("-" * 60, flush=True)
    tot_s = sum(p["solve_s"] for p in per_day)
    tot_fr = sum(p["n_windows"] for p in per_day) * a["dT"]
    print(f"{'MEAN':16s} {bl.mean():>10.4f} {rm.mean():>11.4f} {fu.mean():>10.4f} "
          f"{'':>8} {tot_s / tot_fr * 1000:>10.3f}", flush=True)
    print(f"{'STD':16s} {bl.std():>10.4f} {rm.std():>11.4f} {fu.std():>10.4f}", flush=True)
    print(f"[time] solver forward: {tot_s:.1f}s for {tot_fr:,} frames "
          f"= {tot_s / tot_fr * 1000:.3f} ms/frame  on {dev}", flush=True)

    summary = {"ckpt": args.ckpt, "epoch": ck.get("epoch"), "split": args.split,
               # record the settings the numbers depend on, so a json can never be read
               # against the wrong protocol later
               "obs_every_k": a.get("obs_every_k") or config.get("observation", "obs_every_k"),
               "n_iter": int(solver.n_iter), "dT": a["dT"], "frames_per_day": args.frames,
               "device": str(dev),
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "node": os.environ.get("SLURMD_NODENAME", ""),
               "solve_total_s": float(tot_s), "n_frames": int(tot_fr),
               "per_frame_ms": float(tot_s / tot_fr * 1000),
               "blind_mse_mean": float(bl.mean()), "blind_mse_std": float(bl.std()),
               "blind_rmse_mean": float(rm.mean()), "blind_rmse_std": float(rm.std()),
               "full_mse_mean": float(fu.mean()), "full_mse_std": float(fu.std()),
               "per_day": per_day}
    outp = os.path.join(args.outdir, f"test_metrics{args.tag}.json")
    with open(outp, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[out] {outp}", flush=True)


if __name__ == "__main__":
    main()
