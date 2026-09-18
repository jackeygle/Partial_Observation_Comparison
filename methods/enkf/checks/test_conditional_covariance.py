"""Does the covariance U-Net learn sample-conditional covariance or a static map?

The shuffle control preserves the complete marginal distribution and spatial
structure of predicted covariances while breaking their pairing with the state.
Full beating shuffled is therefore direct evidence of useful conditioning.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from crowdcore import config
from methods.enkf.checks.run_lowrank_unetkf import load_models
from methods.enkf.covariance_only.train import selected_nll
from methods.enkf.lowrank.kalman import analysis_update
from methods.enkf.lowrank.model import STATE_SCALE
from methods.enkf.lowrank.train import random_sensor_mask, sensor_library
from methods.enkf.surrogate.data import PairFrames
from methods.enkf.surrogate.model import surrogate_mean


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pairs", type=int, default=4096)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    mean_net, cov_net, train_args, mean_path = load_models(args.checkpoint, device)
    history = int(train_args.get("input_frames", 1))
    data = PairFrames("valid", device, input_frames=history)
    count = min(args.pairs, data.n_pairs)
    rows = torch.linspace(0, data.n_pairs - 1, count, device=device).round().long().unique()
    masks, walkable = sensor_library(device, 7.0)
    obs_std = torch.tensor(config.get("observation", "obs_std"), device=device)
    state_scale = torch.tensor(STATE_SCALE, device=device).view(1, 4, 1, 1)
    generator = torch.Generator(device=device).manual_seed(args.seed + 40_000)
    modes = ("mean", "diagonal", "shuffled", "full")
    nll_sum = {mode: 0.0 for mode in modes if mode != "mean"}
    mse_sum = {mode: 0.0 for mode in modes}
    mse_count = 0

    shape = (4, data.frames.shape[-2], data.frames.shape[-1])
    moments = {key: torch.zeros(shape, dtype=torch.float64, device=device)
               for key in ("v", "e", "v2", "e2", "ve")}
    samples = 0

    with torch.inference_mode():
        for begin in range(0, len(rows), args.batch):
            x, target = data.batch(rows[begin:begin + args.batch])
            truth = target[:, 0]
            mean = surrogate_mean(mean_net, x)[:, 0]
            factor, diagonal = cov_net(x[:, -1], mean, walkable)
            permutation = torch.randperm(len(x), generator=generator, device=device)
            shuffled_factor = factor[permutation]
            shuffled_diagonal = diagonal[permutation]
            zero_factor = torch.zeros_like(factor)

            nll_sum["diagonal"] += float(selected_nll(
                truth, mean, zero_factor, diagonal, walkable.bool())) * len(x)
            nll_sum["shuffled"] += float(selected_nll(
                truth, mean, shuffled_factor, shuffled_diagonal, walkable.bool())) * len(x)
            nll_sum["full"] += float(selected_nll(
                truth, mean, factor, diagonal, walkable.bool())) * len(x)

            sensor_mask = random_sensor_mask(len(x), 3, masks, generator)
            observation = truth + torch.randn(
                truth.shape, generator=generator, device=device
            ) * obs_std.view(1, 4, 1, 1)
            outputs = {
                "mean": mean,
                "diagonal": analysis_update(mean, zero_factor, diagonal, observation,
                                              sensor_mask, obs_std.square()),
                "shuffled": analysis_update(mean, shuffled_factor, shuffled_diagonal,
                                              observation, sensor_mask, obs_std.square()),
                "full": analysis_update(mean, factor, diagonal, observation,
                                          sensor_mask, obs_std.square()),
            }
            blind_walkable = walkable[None, None].bool() & ~sensor_mask[:, None]
            blind_walkable = blind_walkable.expand_as(truth)
            for mode, estimate in outputs.items():
                error = ((estimate - truth) / state_scale).square()
                mse_sum[mode] += float(error[blind_walkable].double().sum())
            mse_count += int(blind_walkable.sum())

            variance = factor.square().sum(dim=1) + diagonal
            square_error = (truth - mean).square()
            vd, ed = variance.double(), square_error.double()
            moments["v"] += vd.sum(dim=0)
            moments["e"] += ed.sum(dim=0)
            moments["v2"] += vd.square().sum(dim=0)
            moments["e2"] += ed.square().sum(dim=0)
            moments["ve"] += (vd * ed).sum(dim=0)
            samples += len(x)

    centered_ve = moments["ve"] - moments["v"] * moments["e"] / samples
    centered_v2 = moments["v2"] - moments["v"].square() / samples
    centered_e2 = moments["e2"] - moments["e"].square() / samples
    temporal_correlation = {}
    temporal_cv = {}
    selected = walkable.bool()
    for channel, name in enumerate(("density", "vx", "vy", "var")):
        numerator = centered_ve[channel][selected].sum()
        denominator = torch.sqrt(centered_v2[channel][selected].sum().clamp_min(0)
                                 * centered_e2[channel][selected].sum().clamp_min(0))
        temporal_correlation[name] = float(numerator / denominator.clamp_min(1e-30))
        mean_variance = moments["v"][channel] / samples
        std_variance = (centered_v2[channel] / samples).clamp_min(0).sqrt()
        temporal_cv[name] = float(
            (std_variance[selected] / mean_variance[selected].clamp_min(1e-12)).mean())

    result = {
        "config": vars(args), "checkpoint": str(Path(args.checkpoint).resolve()),
        "mean_checkpoint": mean_path, "pairs": len(rows), "input_frames": history,
        "walkable_gaussian_nll": {mode: value / len(rows) for mode, value in nll_sum.items()},
        "analysis_normalized_mse_walkable_blind": {
            mode: value / mse_count for mode, value in mse_sum.items()},
        "within_location_temporal_variance_error_correlation": temporal_correlation,
        "predicted_variance_temporal_cv": temporal_cv,
    }
    result["relative"] = {
        "shuffle_nll_increase": (result["walkable_gaussian_nll"]["shuffled"]
                                 - result["walkable_gaussian_nll"]["full"]),
        "shuffle_analysis_mse_increase_fraction": (
            result["analysis_normalized_mse_walkable_blind"]["shuffled"]
            / result["analysis_normalized_mse_walkable_blind"]["full"] - 1),
        "diagonal_analysis_mse_increase_fraction": (
            result["analysis_normalized_mse_walkable_blind"]["diagonal"]
            / result["analysis_normalized_mse_walkable_blind"]["full"] - 1),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
