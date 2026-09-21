"""Compare prior and posterior uncertainty for jointly trained low-rank KFs.

Unlike the older calibration sweep, every checkpoint keeps its own fine-tuned
forecast network and causal history length.  This is required for a fair
comparison of the one-, five-, and ten-frame joint experiments.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch

from crowdcore import config
from methods.enkf.lcskf.filter import load_models
from methods.enkf.lcskf.kalman import analysis_update, posterior_diag
from methods.enkf.lcskf.covariance import STATE_SCALE
from methods.enkf.lcskf.train import random_sensor_mask, sensor_library
from methods.enkf.lcskf.dynamics.data import PairFrames
from methods.enkf.lcskf.dynamics.model import surrogate_mean

Z50 = 0.6744897501960817
Z90 = 1.6448536269514722
REGIONS = ("all", "walkable", "observed", "blind", "walkable_blind")
STAGES = ("prior", "posterior")
STAT_KEYS = ("se", "spread", "variance", "crps", "nll", "cov50", "cov90", "count",
              "se_raw", "crps_raw", "spread_raw")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True,
                        metavar="NAME=CHECKPOINT")
    parser.add_argument("--alphas", default="0.5,1,2,4")
    parser.add_argument("--pairs", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--days", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def gaussian_crps(mean, sigma, truth):
    sigma = sigma.clamp_min(1e-8)
    z = (truth - mean) / sigma
    cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    pdf = torch.exp(-0.5 * z.square()) / math.sqrt(2 * math.pi)
    return sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


def empty_stats(model_names, alphas):
    return {
        model: {
            f"alpha{alpha:g}": {
                stage: {
                    region: {key: 0.0 for key in STAT_KEYS}
                    for region in REGIONS
                }
                for stage in STAGES
            }
            for alpha in alphas
        }
        for model in model_names
    }


def accumulate(acc, estimate, variance, truth, selection, scale):
    normalized_error = (estimate - truth) / scale
    normalized_variance = (variance / scale.square()).clamp_min(1e-12)
    sigma = normalized_variance.sqrt()
    absolute = normalized_error.abs()
    crps = gaussian_crps(estimate / scale, sigma, truth / scale)
    nll = 0.5 * (math.log(2 * math.pi) + normalized_variance.log()
                 + normalized_error.square() / normalized_variance)
    count = int(selection.sum())
    # Raw physical units alongside the normalised ones.  The per-channel scales differ
    # by 5.7x, so a normalised score weighs the four channels equally while a raw one
    # is ~66% vx -- and the published numbers (compare/score_uncertainty.py) are raw.
    # Reporting both means a reader never has to guess which convention a number is in.
    raw_error = estimate - truth
    raw_sigma = variance.clamp_min(1e-12).sqrt()
    acc["se_raw"] += float(raw_error.square()[selection].double().sum())
    acc["crps_raw"] += float(gaussian_crps(estimate, raw_sigma, truth)[selection].double().sum())
    acc["spread_raw"] += float(raw_sigma[selection].double().sum())
    acc["se"] += float(normalized_error.square()[selection].double().sum())
    acc["spread"] += float(sigma[selection].double().sum())
    acc["variance"] += float(normalized_variance[selection].double().sum())
    acc["crps"] += float(crps[selection].double().sum())
    acc["nll"] += float(nll[selection].double().sum())
    acc["cov50"] += int((absolute[selection] <= Z50 * sigma[selection]).sum())
    acc["cov90"] += int((absolute[selection] <= Z90 * sigma[selection]).sum())
    acc["count"] += count


def finalize(acc):
    count = int(acc["count"])
    rmse = math.sqrt(acc["se"] / count)
    mean_sigma = acc["spread"] / count
    rms_spread = math.sqrt(acc["variance"] / count)
    rmse_raw = math.sqrt(acc["se_raw"] / count)
    return {
        "rmse_raw": rmse_raw,
        "crps_raw": acc["crps_raw"] / count,
        "sigma_mean_raw": acc["spread_raw"] / count,
        "spread_skill_raw": (acc["spread_raw"] / count) / rmse_raw if rmse_raw > 0 else float("nan"),
        "rmse": rmse,
        "mean_spread": mean_sigma,
        "rms_spread": rms_spread,
        "mean_sigma_skill": mean_sigma / rmse,
        "spread_skill": rms_spread / rmse,
        "crps": acc["crps"] / count,
        "marginal_gaussian_nll": acc["nll"] / count,
        "coverage50": acc["cov50"] / count,
        "coverage90": acc["cov90"] / count,
        "count": count,
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("no GPU; submit a gpu-debug/V100 job or add --allow-cpu")
    if args.pairs < 1 or args.batch < 1:
        raise SystemExit("pairs and batch must be positive")
    alphas = tuple(float(value) for value in args.alphas.split(","))
    if any(alpha <= 0 for alpha in alphas):
        raise SystemExit("all covariance multipliers must be positive")
    checkpoints = dict(spec.split("=", 1) for spec in args.model)
    if len(checkpoints) != len(args.model):
        raise SystemExit("model names must be unique")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = {name: load_models(path, device) for name, path in checkpoints.items()}
    histories = {name: int(parts[2].get("input_frames", 1))
                 for name, parts in loaded.items()}
    max_history = max(histories.values())
    data = PairFrames("valid", device, args.days, args.max_frames,
                      input_frames=max_history)
    pair_count = min(args.pairs, data.n_pairs)
    rows = torch.linspace(0, data.n_pairs - 1, pair_count, device=device).round().long().unique()
    masks, walkable = sensor_library(device, 7.0)
    obs_std = torch.tensor(config.get("observation", "obs_std"), device=device)
    scale = torch.tensor(STATE_SCALE, device=device).view(1, 4, 1, 1)
    generator = torch.Generator(device=device).manual_seed(args.seed + 10_000)
    totals = empty_stats(checkpoints, alphas)
    covariance_totals = {
        name: {"lowrank": 0.0, "diagonal": 0.0, "count": 0}
        for name in checkpoints
    }

    with torch.inference_mode():
        for begin in range(0, len(rows), args.batch):
            x, target = data.batch(rows[begin:begin + args.batch])
            truth = target[:, 0]
            sensor_mask = random_sensor_mask(len(x), 3, masks, generator)
            observation = truth + torch.randn(
                truth.shape, device=device, generator=generator
            ) * obs_std.view(1, 4, 1, 1)
            expanded = sensor_mask[:, None].expand_as(truth)
            walking = walkable[None, None].expand_as(truth)
            selections = {
                "all": torch.ones_like(expanded, dtype=torch.bool),
                "walkable": walking,
                "observed": expanded,
                "blind": ~expanded,
                "walkable_blind": walking & ~expanded,
            }

            for name, (mean_net, cov_net, _, _) in loaded.items():
                history = histories[name]
                mean = surrogate_mean(mean_net, x[:, -history:])[:, 0]
                factor, diagonal = cov_net(x[:, -1], mean, walkable)
                factor_variance = factor.square().sum(dim=1)
                covariance_totals[name]["lowrank"] += float(factor_variance.double().sum())
                covariance_totals[name]["diagonal"] += float(diagonal.double().sum())
                covariance_totals[name]["count"] += factor_variance.numel()

                for alpha in alphas:
                    label = f"alpha{alpha:g}"
                    scaled_factor = factor * math.sqrt(alpha)
                    scaled_diagonal = diagonal * alpha
                    prior_variance = scaled_factor.square().sum(dim=1) + scaled_diagonal
                    analysis = analysis_update(
                        mean, scaled_factor, scaled_diagonal, observation,
                        sensor_mask, obs_std.square())
                    analysis_variance = posterior_diag(
                        mean, scaled_factor, scaled_diagonal, observation,
                        sensor_mask, obs_std.square()).clamp_min(1e-12)
                    for region, selection in selections.items():
                        accumulate(totals[name][label]["prior"][region], mean,
                                   prior_variance, truth, selection, scale)
                        accumulate(totals[name][label]["posterior"][region], analysis,
                                   analysis_variance, truth, selection, scale)

            done = min(begin + args.batch, len(rows))
            if done == len(rows) or done % (args.batch * 8) == 0:
                print(f"[compare] {done:,}/{len(rows):,}", flush=True)

    results = {
        model: {
            label: {
                stage: {region: finalize(values) for region, values in regions.items()}
                for stage, regions in stages.items()
            }
            for label, stages in labels.items()
        }
        for model, labels in totals.items()
    }
    covariance = {}
    for name, values in covariance_totals.items():
        total = values["lowrank"] + values["diagonal"]
        covariance[name] = {
            "lowrank_trace_fraction": values["lowrank"] / total,
            "lowrank_variance_mean": values["lowrank"] / values["count"],
            "diagonal_variance_mean": values["diagonal"] / values["count"],
        }
    selection = {}
    for name in checkpoints:
        selection[name] = {}
        for stage in STAGES:
            candidates = [f"alpha{alpha:g}" for alpha in alphas]
            selection[name][stage] = {
                "best_crps_walkable_blind": min(
                    candidates,
                    key=lambda label: results[name][label][stage]["walkable_blind"]["crps"]),
                "best_calibrated_walkable_blind": min(
                    candidates,
                    key=lambda label: abs(math.log(max(
                        results[name][label][stage]["walkable_blind"]["spread_skill"],
                        1e-12)))),
            }
    output = {
        "config": vars(args),
        "pairs": len(rows),
        "max_input_frames": max_history,
        "input_frames": histories,
        "checkpoints": {name: os.path.abspath(path) for name, path in checkpoints.items()},
        "covariance": covariance,
        "results": results,
        "selection": selection,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps({"pairs": len(rows), "input_frames": histories,
                      "covariance": covariance, "selection": selection}, indent=2), flush=True)


if __name__ == "__main__":
    main()
