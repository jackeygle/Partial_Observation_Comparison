"""Export the filter's own analyses as a training background, one file per day.

Training currently feeds the forecast and covariance nets the *true* past state,
while deployment feeds them the filter's own analyses.  On the test split that
background carries 0.247 normalised RMSE over walkable blind cells against 0 in
training, and the gap sits exactly where the headline metric is computed.

This script closes it the cheap way -- one round of DAgger: run the current
filter over the training days and keep what it produced, so the next covariance
model can be fitted on the states it will actually see.  Observations are
simulated in memory with the same ``generate_observations`` the other methods
use, so nothing but the analyses and their masks is written.

    sbatch methods/enkf/lcskf/sbatch/submit_backgrounds.sbatch --split train --only atc-20121028
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from crowdcore import config, navigation
from crowdcore import observation_model as om
from methods.enkf.lcskf.filter import load_models, run_arrays

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # methods/enkf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",
                   default=os.path.join(HERE, "runs/joint_j4_5f_r32_nofcst_s0/best.pt"))
    p.add_argument("--alpha", type=float, default=2.0,
                   help="covariance scale the background was calibrated at")
    p.add_argument("--split", default="train", choices=("train", "valid", "test"))
    p.add_argument("--only", default="", help="only days whose stem contains this")
    p.add_argument("--outdir", default="check_outputs/backgrounds_j4nofcst_a2")
    p.add_argument("--seed", type=int, default=0, help="base seed for day_seed")
    p.add_argument("--frames", type=int, default=0, help="debug: first N frames per day")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.outdir, exist_ok=True)
    mean_net, cov_net, train_args, mean_path = load_models(args.checkpoint, device)
    input_frames = int(train_args.get("input_frames", 1))
    print(f"[bg] {args.checkpoint} alpha={args.alpha:g} history={input_frames} mean={mean_path}",
          flush=True)

    sensing = config.get("observation", "sensing_range")
    agents = config.get("observation", "num_agents")
    every_k = config.get("observation", "obs_every_k")
    mode = config.get("observation", "trajectory_mode")
    init = config.get("observation", "init_method")
    obs_std = np.asarray(config.get("observation", "obs_std"), dtype=np.float32)

    days = [d for d in om.split_files(args.split) if args.only in os.path.basename(d)]
    print(f"[bg] {len(days)} {args.split} day(s); robots={agents} r={sensing} "
          f"k={every_k} traj={mode} init={init}", flush=True)

    for path in days:
        stem = os.path.basename(path).split("_")[0]
        out = os.path.join(args.outdir, f"bg_{stem}.npz")
        if os.path.exists(out):
            print(f"[skip] {stem} exists", flush=True); continue
        tic = time.time()
        X, _ = om.load_state(path)
        if args.frames:
            X = X[:args.frames]
        valid = navigation.build_valid_mask_from_config(X)
        sim = om.generate_observations(X, sensing_range=sensing, num_agents=agents,
                                       add_noise=True, seed=om.day_seed(path, args.seed, mode),
                                       valid_mask=valid, obs_std=obs_std, obs_every_k=every_k)
        X0 = om.fill_missing_state(sim["Y"], sim["Omega_c"], method=init)
        static_mask = torch.from_numpy(
            navigation.build_valid_mask_from_config().astype(bool)).to(device)
        estimates, _, seconds = run_arrays(
            sim["Y"], sim["Omega"], X0, obs_std, mean_net, cov_net, static_mask, device,
            0, "full", args.alpha, args.alpha, input_frames)
        np.savez(out, Est=estimates.astype(np.float32), Omega=sim["Omega"].astype(bool))
        with open(out.replace(".npz", ".source.json"), "w") as f:
            json.dump({"day": stem, "split": args.split, "frames": int(len(X)),
                       "checkpoint": args.checkpoint, "alpha": args.alpha,
                       "input_frames": input_frames, "mean_checkpoint": mean_path,
                       "day_seed": int(om.day_seed(path, args.seed, mode)),
                       "sensing_range": sensing, "num_agents": agents,
                       "obs_every_k": every_k, "trajectory_mode": mode,
                       "init_method": init, "filter_seconds": round(seconds, 1)}, f, indent=2)
        print(f"[done] {stem}: {len(X)} frames, filter {seconds:.0f}s, "
              f"total {time.time()-tic:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
