"""Diagnose and calibrate marginal uncertainty specifically in blind walkable cells.

``gamma`` is a standard-deviation multiplier applied only to the diagonal
covariance of currently unobserved walkable cells.  The learned low-rank term
and the diagonal covariance at observed cells are left unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from crowdcore import config
from methods.enkf.checks.run_lowrank_unetkf import load_models
from methods.enkf.lowrank.kalman import analysis_update, posterior_diag
from methods.enkf.lowrank.model import STATE_SCALE
from methods.enkf.lowrank.train import random_sensor_mask, sensor_library
from methods.enkf.surrogate.data import PairFrames
from methods.enkf.surrogate.model import surrogate_mean

CHANNELS = ("density", "vx", "vy", "var")
Z50, Z90 = 0.6744897501960817, 1.6448536269514722
STAT_KEYS = ("se", "spread", "variance", "crps", "nll", "cov50", "cov90", "count")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gammas", default="1,1.1,1.2,1.4,1.6")
    parser.add_argument("--factor-scale", type=float, default=1.1,
                        help="covariance multiplier for U U^T")
    parser.add_argument("--diagonal-scale", type=float, default=0.75,
                        help="baseline covariance multiplier for diag(d)")
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


def empty_accumulator():
    return {key: 0.0 for key in STAT_KEYS}


def accumulate(acc, estimate, variance, truth, selection, scale):
    error = (estimate - truth) / scale
    normalized_variance = (variance / scale.square()).clamp_min(1e-12)
    sigma = normalized_variance.sqrt()
    absolute = error.abs()
    crps = gaussian_crps(estimate / scale, sigma, truth / scale)
    nll = 0.5 * (math.log(2 * math.pi) + normalized_variance.log()
                 + error.square() / normalized_variance)
    count = int(selection.sum())
    if not count:
        return
    acc["se"] += float(error.square()[selection].double().sum())
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
    mean_spread = acc["spread"] / count
    rms_spread = math.sqrt(acc["variance"] / count)
    return {
        "rmse": rmse,
        "mean_spread": mean_spread,
        "rms_spread": rms_spread,
        "mean_sigma_skill": mean_spread / rmse,
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
    gammas = tuple(float(value) for value in args.gammas.split(","))
    if (not gammas or any(value <= 0 for value in gammas)
            or args.factor_scale < 0 or args.diagonal_scale <= 0):
        raise SystemExit("gammas/diagonal scale must be positive and factor scale non-negative")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean_net, cov_net, train_args, mean_path = load_models(args.checkpoint, device)
    history = int(train_args.get("input_frames", 1))
    data = PairFrames("valid", device, args.days, args.max_frames, input_frames=history)
    pair_count = min(args.pairs, data.n_pairs)
    rows = torch.linspace(0, data.n_pairs - 1, pair_count, device=device).round().long().unique()
    sensor_masks, walkable = sensor_library(device, 7.0)
    walking = walkable[None, None].bool()
    obs_std = torch.tensor(config.get("observation", "obs_std"), device=device)
    scale = torch.tensor(STATE_SCALE, device=device).view(1, 4, 1, 1)

    activities = []
    for begin in range(0, len(rows), args.batch):
        x, _ = data.batch(rows[begin:begin + args.batch])
        activities.append(x[:, -1, 0, walkable.bool()].sum(dim=1))
    activities = torch.cat(activities)
    activity_thresholds = torch.quantile(activities, torch.tensor([1 / 3, 2 / 3], device=device))

    region_names = ("walkable", "observed", "blind", "walkable_blind",
                    "activity_low_walkable_blind", "activity_medium_walkable_blind",
                    "activity_high_walkable_blind")
    labels = tuple(f"gamma{gamma:g}" for gamma in gammas)
    totals = {
        label: {
            stage: {
                region: {"all_channels": empty_accumulator(),
                         **{channel: empty_accumulator() for channel in CHANNELS}}
                for region in region_names
            }
            for stage in ("prior", "posterior")
        }
        for label in labels
    }
    generator = torch.Generator(device=device).manual_seed(args.seed + 10_000)
    with torch.inference_mode():
        for begin in range(0, len(rows), args.batch):
            batch_rows = rows[begin:begin + args.batch]
            x, target = data.batch(batch_rows)
            truth = target[:, 0]
            mean = surrogate_mean(mean_net, x)[:, 0]
            factor, diagonal = cov_net(x[:, -1], mean, walkable)
            factor = factor * math.sqrt(args.factor_scale)
            diagonal = diagonal * args.diagonal_scale
            mask = random_sensor_mask(len(x), 3, sensor_masks, generator)
            expanded = mask[:, None].expand_as(truth)
            walking_full = walking.expand_as(truth)
            blind_walkable = walking_full & ~expanded
            observation = truth + torch.randn(
                truth.shape, device=device, generator=generator
            ) * obs_std.view(1, 4, 1, 1)
            batch_activity = x[:, -1, 0, walkable.bool()].sum(dim=1)
            low = batch_activity <= activity_thresholds[0]
            medium = ((batch_activity > activity_thresholds[0])
                      & (batch_activity <= activity_thresholds[1]))
            high = batch_activity > activity_thresholds[1]
            selections = {
                "walkable": walking_full,
                "observed": expanded & walking_full,
                "blind": ~expanded,
                "walkable_blind": blind_walkable,
                "activity_low_walkable_blind": blind_walkable & low[:, None, None, None],
                "activity_medium_walkable_blind": blind_walkable & medium[:, None, None, None],
                "activity_high_walkable_blind": blind_walkable & high[:, None, None, None],
            }
            for gamma, label in zip(gammas, labels):
                calibrated_diagonal = torch.where(
                    blind_walkable, diagonal * gamma ** 2, diagonal)
                prior_variance = factor.square().sum(dim=1) + calibrated_diagonal
                analysis = analysis_update(
                    mean, factor, calibrated_diagonal, observation, mask, obs_std.square())
                posterior_variance = posterior_diag(
                    mean, factor, calibrated_diagonal, observation, mask,
                    obs_std.square()).clamp_min(1e-12)
                for stage, estimate, variance in (
                        ("prior", mean, prior_variance),
                        ("posterior", analysis, posterior_variance)):
                    for region, selection in selections.items():
                        group = totals[label][stage][region]
                        accumulate(group["all_channels"], estimate, variance, truth,
                                   selection, scale)
                        for channel_index, channel in enumerate(CHANNELS):
                            accumulate(group[channel], estimate, variance, truth,
                                       selection & (torch.arange(4, device=device)[None, :, None, None]
                                                    == channel_index), scale)
            done = min(begin + args.batch, len(rows))
            if done == len(rows) or done % (args.batch * 8) == 0:
                print(f"[blind-sweep] {done:,}/{len(rows):,}", flush=True)

    results = {
        label: {
            stage: {
                region: {channel: finalize(acc) for channel, acc in channels.items()}
                for region, channels in regions.items()
            }
            for stage, regions in stages.items()
        }
        for label, stages in totals.items()
    }
    target = lambda label: results[label]["posterior"]["walkable_blind"]["all_channels"]
    selection = {
        "best_crps": min(labels, key=lambda label: target(label)["crps"]),
        "best_nll": min(labels, key=lambda label: target(label)["marginal_gaussian_nll"]),
        "best_coverage90": min(labels, key=lambda label: abs(target(label)["coverage90"] - 0.9)),
        "best_spread_skill": min(
            labels, key=lambda label: abs(math.log(target(label)["spread_skill"]))),
    }
    output = {
        "config": vars(args),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "mean_checkpoint": mean_path,
        "input_frames": history,
        "pairs": len(rows),
        "activity_tertiles": [float(value) for value in activity_thresholds],
        "gamma_definition": "std multiplier; blind walkable diag(d) covariance is multiplied by gamma^2",
        "selection": selection,
        "selected_metrics": {key: target(label) for key, label in selection.items()},
        "results": results,
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"selection": selection, "selected_metrics": output["selected_metrics"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
