"""Hard gate: is the old Gaussian-NLL sigma useful as Kalman diagonal variance?"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from crowdcore import config, navigation
from methods.enkf.checks.run_lowrank_unetkf import project_state
from methods.enkf.lowrank.kalman import analysis_update
from methods.enkf.lowrank.model import STATE_SCALE
from methods.enkf.lowrank.train import random_sensor_mask, sensor_library
from methods.enkf.surrogate.data import PairFrames
from methods.enkf.surrogate.model import load_pedpred3, raw_forecast, surrogate_mean
from methods.enkf.surrogate.train import LOGVAR_MAX, LOGVAR_MIN

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mean-ckpt", default=os.path.join(HERE, "runs/surrogate_mean_s0/best.pt"))
    p.add_argument("--sigma-ckpt", default=os.path.join(HERE, "runs/surrogate_sigma_s0/best.pt"))
    p.add_argument("--bias-file", default=os.path.join(HERE, "runs/local_unetkf_bias/bias_maps.npz"))
    p.add_argument("--covariance", default=os.path.join(
        HERE, "runs/local_unetkf_static/static_covariance.npz"))
    p.add_argument("--std-scales", default="0.3,0.5,1.0")
    p.add_argument("--pairs", type=int, default=4096)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--days", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("no GPU; submit a gpu-debug job or add --allow-cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    std_scales = tuple(float(value) for value in args.std_scales.split(","))
    mean_net = load_pedpred3(args.mean_ckpt, device).eval()
    sigma_net = load_pedpred3(args.sigma_ckpt, device).eval()
    with np.load(args.bias_file) as z:
        bias = torch.from_numpy(z["train_bias"]).to(device)
    static_diagonal = {}
    with np.load(args.covariance) as z:
        for kind in ("raw", "centered"):
            factor = torch.from_numpy(z[f"{kind}_factor"]).to(device)
            remainder = torch.from_numpy(z[f"{kind}_diagonal"]).to(device)
            static_diagonal[kind] = remainder + factor.square().sum(0)
    data = PairFrames("valid", device, args.days, args.max_frames)
    count = min(args.pairs, data.n_pairs)
    rows = torch.linspace(0, data.n_pairs - 1, count, device=device).round().long().unique()
    sensor_masks, walkable = sensor_library(device, 7.0)
    obs_std = torch.tensor(config.get("observation", "obs_std"), device=device)
    generator = torch.Generator(device=device).manual_seed(args.seed + 10_000)
    state_scale = torch.tensor(STATE_SCALE, device=device).view(1, 4, 1, 1)
    base_modes = ("mean", "mean_bias_corrected", "static_raw", "static_centered")
    sigma_modes = tuple(f"sigma_raw_s{value:g}" for value in std_scales) + tuple(
        f"sigma_centered_s{value:g}" for value in std_scales)
    modes = base_modes + sigma_modes
    regions = ("all", "observed", "blind", "occupied", "walkable_blind")
    sums = {mode: {region: 0.0 for region in regions} for mode in modes}
    counts = {region: 0 for region in regions}

    with torch.inference_mode():
        for begin in range(0, len(rows), args.batch):
            x, y = data.batch(rows[begin:begin + args.batch])
            truth = y[:, 0]
            mean = surrogate_mean(mean_net, x)[:, 0]
            corrected = project_state(mean + bias[None])
            variance = torch.exp(raw_forecast(sigma_net, x).clamp(
                LOGVAR_MIN, LOGVAR_MAX))[:, 0]
            mask = random_sensor_mask(len(x), 3, sensor_masks, generator)
            observation = truth + torch.randn(
                truth.shape, device=device, generator=generator) * obs_std.view(1, 4, 1, 1)
            zero_factor = mean.new_zeros((len(x), 1, 4, 36, 12))

            estimates = {"mean": mean, "mean_bias_corrected": corrected}
            estimates["static_raw"] = analysis_update(
                mean, zero_factor, static_diagonal["raw"][None].expand_as(mean),
                observation, mask, obs_std.square())
            estimates["static_centered"] = analysis_update(
                corrected, zero_factor, static_diagonal["centered"][None].expand_as(mean),
                observation, mask, obs_std.square())
            for value in std_scales:
                scaled_variance = variance * value ** 2
                estimates[f"sigma_raw_s{value:g}"] = analysis_update(
                    mean, zero_factor, scaled_variance, observation, mask, obs_std.square())
                estimates[f"sigma_centered_s{value:g}"] = analysis_update(
                    corrected, zero_factor, scaled_variance, observation, mask, obs_std.square())

            expanded = mask[:, None].expand_as(truth)
            walking = walkable[None, None].expand_as(truth)
            occupied = (truth[:, 0] > 0)[:, None].expand_as(truth)
            selections = {"all": torch.ones_like(expanded), "observed": expanded,
                          "blind": ~expanded, "occupied": occupied,
                          "walkable_blind": walking & ~expanded}
            for region, selection in selections.items():
                counts[region] += int(selection.sum())
                for mode, estimate in estimates.items():
                    square = ((estimate - truth) / state_scale).square()
                    sums[mode][region] += float(square[selection].double().sum())
            done = min(begin + args.batch, len(rows))
            if done == len(rows) or done % (args.batch * 8) == 0:
                print(f"[eval] {done:,}/{len(rows):,}", flush=True)

    rmse = {mode: {region: (sums[mode][region] / counts[region]) ** 0.5
                   for region in regions} for mode in modes}
    raw_baseline, centered_baseline = rmse["static_raw"], rmse["static_centered"]
    verdicts = {}
    for family, baseline in (("raw", raw_baseline), ("centered", centered_baseline)):
        candidates = [mode for mode in modes if mode.startswith(f"sigma_{family}")]
        best = min(candidates, key=lambda mode: rmse[mode]["all"])
        improvement = (baseline["all"] - rmse[best]["all"]) / baseline["all"]
        verdicts[family] = {"best": best, "all_improvement_vs_static": improvement,
                            "passes_one_percent_gate": improvement >= 0.01}
    result = {"config": vars(args), "pairs": len(rows), "normalized_rmse": rmse,
              "verdict": verdicts}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
