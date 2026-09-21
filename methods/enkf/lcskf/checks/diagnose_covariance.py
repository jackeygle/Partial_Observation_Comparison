"""Held-out ablations and covariance diagnostics for the low-rank U-Net.

This uses the same one-step validation construction as training, but never updates
parameters.  It separates the contribution of the learned diagonal from the
low-rank factor and reports whether the factor is using its nominal rank.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from crowdcore import config, navigation
from methods.enkf.lcskf.kalman import analysis_update
from methods.enkf.lcskf.covariance import STATE_SCALE
from methods.enkf.lcskf.train import random_sensor_mask, sensor_library
from methods.enkf.lcskf.dynamics.data import PairFrames
from methods.enkf.lcskf.dynamics.model import surrogate_mean
from methods.enkf.lcskf.filter import load_models


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pairs", type=int, default=4096)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    mean_net, cov_net, train_args, mean_path = load_models(args.checkpoint, device)
    input_frames = int(train_args.get("input_frames", 1))
    valid = PairFrames("valid", device, input_frames=input_frames)
    n = min(args.pairs, valid.n_pairs)
    obs_std = torch.tensor(config.get("observation", "obs_std"), device=device)
    masks, static_mask = sensor_library(device, float(train_args.get("sensing_radius", 7.0)))
    generator = torch.Generator(device=device).manual_seed(args.seed + 10_000)
    scale = torch.tensor(STATE_SCALE, device=device).view(1, 4, 1, 1)
    modes = ("mean", "diagonal", "full", "fixed-diagonal")
    sums = {m: {"all": 0.0, "blind": 0.0, "observed": 0.0} for m in modes}
    counts = {"all": 0, "blind": 0, "observed": 0}
    lowrank_trace = diagonal_trace = effective_rank = numerical_rank = 0.0
    factor_channel = torch.zeros(4, device=device, dtype=torch.float64)
    diagonal_channel = torch.zeros_like(factor_channel)
    samples = 0

    with torch.inference_mode():
        for begin in range(0, n, args.batch):
            rows = torch.arange(begin, min(begin + args.batch, n), device=device)
            x, target = valid.batch(rows)
            x_grid, target_grid = x[:, -1], target[:, 0]
            mean = surrogate_mean(mean_net, x)[:, 0]
            factor, diagonal = cov_net(x_grid, mean, static_mask)
            mask = random_sensor_mask(len(x), int(train_args.get("num_agents", 3)),
                                      masks, generator)
            noise = torch.randn(target_grid.shape, device=device, generator=generator)
            observation = target_grid + noise * obs_std.view(1, -1, 1, 1)

            outputs = {"mean": mean}
            outputs["diagonal"] = analysis_update(
                mean, torch.zeros_like(factor), diagonal, observation, mask, obs_std.square())
            outputs["full"] = analysis_update(
                mean, factor, diagonal, observation, mask, obs_std.square())
            outputs["fixed-diagonal"] = analysis_update(
                mean, factor, scale.square().expand_as(diagonal), observation, mask,
                obs_std.square())

            expanded = mask[:, None].expand_as(target_grid)
            selections = {"all": torch.ones_like(expanded, dtype=torch.bool),
                          "blind": ~expanded, "observed": expanded}
            for region, selection in selections.items():
                counts[region] += int(selection.sum())
                for mode, estimate in outputs.items():
                    sq = ((estimate - target_grid) / scale).square()
                    sums[mode][region] += float(sq[selection].double().sum())

            factor_var = factor.square().sum(dim=1)
            lowrank_trace += float(factor_var.double().sum())
            diagonal_trace += float(diagonal.double().sum())
            factor_channel += factor_var.double().sum(dim=(0, 2, 3))
            diagonal_channel += diagonal.double().sum(dim=(0, 2, 3))
            u = factor.permute(0, 2, 3, 4, 1).reshape(len(x), -1, factor.shape[1]).double()
            eigenvalues = torch.linalg.eigvalsh(u.transpose(1, 2) @ u).clamp_min(0)
            probability = eigenvalues / eigenvalues.sum(dim=1, keepdim=True).clamp_min(1e-30)
            effective_rank += float(torch.exp(
                -(probability * probability.clamp_min(1e-30).log()).sum(dim=1)).sum())
            numerical_rank += float((eigenvalues > eigenvalues[:, -1:] * 1e-3).sum())
            samples += len(x)

    result = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "mean_checkpoint": mean_path,
        "rank": int(train_args["rank"]),
        "pairs": n,
        "normalized_mse": {
            mode: {region: sums[mode][region] / counts[region] for region in counts}
            for mode in modes
        },
        "covariance": {
            "lowrank_trace_fraction": lowrank_trace / (lowrank_trace + diagonal_trace),
            "effective_rank_mean": effective_rank / samples,
            "numerical_rank_mean_threshold_1e-3": numerical_rank / samples,
            "lowrank_variance_mean_by_channel":
                (factor_channel / (samples * 36 * 12)).cpu().tolist(),
            "diagonal_variance_mean_by_channel":
                (diagonal_channel / (samples * 36 * 12)).cpu().tolist(),
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
