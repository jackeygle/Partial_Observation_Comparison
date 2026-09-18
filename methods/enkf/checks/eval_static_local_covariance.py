"""One-step Kalman gate for raw/centered climatological covariance teachers."""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from crowdcore import config, navigation
from methods.enkf.lowrank.kalman import analysis_update
from methods.enkf.lowrank.model import STATE_SCALE
from methods.enkf.lowrank.train import random_sensor_mask, sensor_library
from methods.enkf.surrogate.data import PairFrames
from methods.enkf.surrogate.model import load_pedpred3, surrogate_mean
from methods.enkf.checks.run_lowrank_unetkf import project_state

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHANNELS = ("density", "vx", "vy", "var")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mean-ckpt", default=os.path.join(HERE, "runs/surrogate_mean_s0/best.pt"))
    p.add_argument("--covariance", default=os.path.join(
        HERE, "runs/local_unetkf_static/static_covariance.npz"))
    p.add_argument("--bias-file", default=os.path.join(
        HERE, "runs/local_unetkf_bias/bias_maps.npz"))
    p.add_argument("--pairs", type=int, default=4096)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--ranks", default="32,128,256")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--days", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--out", default=os.path.join(
        HERE, "check_outputs/eval/static_local_covariance.json"))
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def load_modes(path, ranks, device):
    modes, metadata = {}, {}
    with np.load(path) as z:
        for kind in ("raw", "centered"):
            all_factor = torch.from_numpy(z[f"{kind}_factor"]).to(device)
            saved_diagonal = torch.from_numpy(z[f"{kind}_diagonal"]).to(device)
            psd_diagonal = saved_diagonal + all_factor.square().sum(0)
            prefix = "raw" if kind == "raw" else "centered_bc"
            modes[f"{prefix}_diagonal"] = (torch.zeros_like(all_factor[:1]), psd_diagonal)
            metadata[kind] = {"stored_rank": len(all_factor)}
            for rank in ranks:
                if rank > len(all_factor):
                    raise ValueError(f"requested rank {rank}, but {kind} stores {len(all_factor)}")
                factor = all_factor[-rank:]
                diagonal = (psd_diagonal - factor.square().sum(0)).clamp_min(1e-10)
                modes[f"{prefix}_r{rank}"] = (factor, diagonal)
    return modes, metadata


def empty_sums(modes):
    regions = ("all", "observed", "blind", "walkable", "walkable_blind")
    return ({mode: {region: torch.zeros(4, dtype=torch.float64) for region in regions}
             for mode in ("mean", "mean_bias_corrected", *modes)},
            {region: torch.zeros(4, dtype=torch.float64) for region in regions})


def main():
    args = parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("no GPU; submit a gpu-debug job or add --allow-cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ranks = tuple(sorted({int(value) for value in args.ranks.split(",") if value}))
    modes, covariance_meta = load_modes(args.covariance, ranks, device)
    with np.load(args.bias_file) as z:
        bias = torch.from_numpy(z["train_bias"]).to(device)
    data = PairFrames("valid", device, args.days, args.max_frames)
    n = min(args.pairs, data.n_pairs)
    rows = torch.linspace(0, data.n_pairs - 1, n, device=device).round().long().unique()
    n = len(rows)
    mean_net = load_pedpred3(args.mean_ckpt, device).eval()
    obs_std = torch.tensor(config.get("observation", "obs_std"), device=device)
    sensor_masks, walkable = sensor_library(device, 7.0)
    generator = torch.Generator(device=device).manual_seed(args.seed + 10_000)
    scale = torch.tensor(STATE_SCALE, device=device).view(1, 4, 1, 1)
    sums, counts = empty_sums(modes)
    tic = time.time()
    with torch.inference_mode():
        for begin in range(0, n, args.batch):
            batch_rows = rows[begin:begin + args.batch]
            x, target = data.batch(batch_rows)
            truth = target[:, 0]
            mean = surrogate_mean(mean_net, x)[:, 0]
            corrected_mean = project_state(mean + bias[None])
            mask = random_sensor_mask(len(x), 3, sensor_masks, generator)
            noise = torch.randn(truth.shape, device=device, generator=generator)
            observation = truth + noise * obs_std.view(1, 4, 1, 1)
            estimates = {"mean": mean, "mean_bias_corrected": corrected_mean}
            for name, (factor, diagonal) in modes.items():
                forecast = corrected_mean if name.startswith("centered_bc") else mean
                estimates[name] = analysis_update(
                    forecast,
                    factor[None].expand(len(x), -1, -1, -1, -1),
                    diagonal[None].expand_as(mean),
                    observation, mask, obs_std.square())
            expanded = mask[:, None].expand_as(truth)
            walking = walkable[None, None].expand_as(truth)
            selections = {
                "all": torch.ones_like(expanded),
                "observed": expanded,
                "blind": ~expanded,
                "walkable": walking,
                "walkable_blind": walking & ~expanded,
            }
            for region, selection in selections.items():
                counts[region] += selection.sum(dim=(0, 2, 3)).double().cpu()
                for name, estimate in estimates.items():
                    square = ((estimate - truth) / scale).square()
                    sums[name][region] += (square * selection).sum(
                        dim=(0, 2, 3), dtype=torch.float64).cpu()
            done = min(begin + args.batch, n)
            if done == n or done % (args.batch * 25) == 0:
                print(f"[eval] {done:,}/{n:,} pairs, {time.time()-tic:.1f}s", flush=True)

    normalized_mse, normalized_rmse = {}, {}
    for mode in sums:
        normalized_mse[mode], normalized_rmse[mode] = {}, {}
        for region in counts:
            channel_mse = sums[mode][region] / counts[region].clamp_min(1)
            normalized_mse[mode][region] = {
                "mean": float((sums[mode][region].sum() / counts[region].sum()).item()),
                "by_channel": dict(zip(CHANNELS, channel_mse.tolist())),
            }
            normalized_rmse[mode][region] = {
                "mean": float((sums[mode][region].sum() / counts[region].sum()).sqrt().item()),
                "by_channel": dict(zip(CHANNELS, channel_mse.sqrt().tolist())),
            }
    result = {
        "config": vars(args), "device": str(device), "pairs": n,
        "seconds": round(time.time() - tic, 3), "covariance": covariance_meta,
        "normalized_mse": normalized_mse, "normalized_rmse": normalized_rmse,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    compact = {mode: {region: values[region]["mean"]
                      for region in ("all", "observed", "blind", "walkable_blind")}
               for mode, values in normalized_rmse.items()}
    print(json.dumps(compact, indent=2), flush=True)
    print(f"[done] {args.out}", flush=True)


if __name__ == "__main__":
    main()
