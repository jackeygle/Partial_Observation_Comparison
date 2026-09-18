"""One-step bias-correction and covariance-inflation sweep for low-rank KF models."""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch

from crowdcore import config
from methods.enkf.checks.run_lowrank_unetkf import load_models, project_state
from methods.enkf.lowrank.kalman import analysis_update, posterior_diag
from methods.enkf.lowrank.model import STATE_SCALE
from methods.enkf.lowrank.train import random_sensor_mask, sensor_library
from methods.enkf.surrogate.data import PairFrames
from methods.enkf.surrogate.model import surrogate_mean

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Z50, Z90 = 0.6744897501960817, 1.6448536269514722


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", default=[], metavar="NAME=CHECKPOINT")
    p.add_argument("--alphas", default="0.5,1,2,4,8")
    p.add_argument(
        "--component-scales", default="",
        help=("comma-separated covariance multipliers FACTOR:DIAGONAL; when set, "
              "overrides --alphas and scales U U^T and diag(d) independently"))
    p.add_argument("--bias-file", default=os.path.join(
        HERE, "runs/local_unetkf_bias/bias_maps.npz"))
    p.add_argument("--pairs", type=int, default=4096)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--days", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def crps_standard_normal(mu, sigma, truth):
    sigma = sigma.clamp_min(1e-8)
    z = (truth - mu) / sigma
    cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    pdf = torch.exp(-0.5 * z.square()) / math.sqrt(2 * math.pi)
    return sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


def blank_metrics(mode_names, regions):
    keys = ("se", "spread", "crps", "cov50", "cov90")
    return {mode: {region: {key: 0.0 for key in keys} for region in regions}
            for mode in mode_names}


def main():
    args = parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("no GPU; submit a gpu-debug job or add --allow-cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_specs = args.model or [
        f"a32={HERE}/runs/lowrank_a_r32_s0/best.pt",
        f"b16={HERE}/runs/lowrank_r16_s0/best.pt",
    ]
    checkpoints = dict(spec.split("=", 1) for spec in model_specs)
    loaded = {name: load_models(path, device) for name, path in checkpoints.items()}
    # Both covariance checkpoints use the same frozen mean checkpoint. Keep one copy.
    mean_net = next(iter(loaded.values()))[0]
    cov_nets = {name: value[1] for name, value in loaded.items()}
    train_args = {name: value[2] for name, value in loaded.items()}
    if args.component_scales:
        covariance_scales = []
        for spec in args.component_scales.split(","):
            factor_scale, diagonal_scale = (float(value) for value in spec.split(":"))
            if factor_scale < 0 or diagonal_scale <= 0:
                raise SystemExit("component scales require FACTOR >= 0 and DIAGONAL > 0")
            covariance_scales.append(
                (f"factor{factor_scale:g}_diag{diagonal_scale:g}",
                 factor_scale, diagonal_scale))
    else:
        alphas = tuple(float(value) for value in args.alphas.split(","))
        if any(alpha <= 0 for alpha in alphas):
            raise SystemExit("all --alphas values must be positive")
        covariance_scales = [
            (f"alpha{alpha:g}", alpha, alpha) for alpha in alphas]
    with np.load(args.bias_file) as z:
        bias = torch.from_numpy(z["train_bias"]).to(device)
    data = PairFrames("valid", device, args.days, args.max_frames)
    count = min(args.pairs, data.n_pairs)
    rows = torch.linspace(0, data.n_pairs - 1, count, device=device).round().long().unique()
    obs_std = torch.tensor(config.get("observation", "obs_std"), device=device)
    sensor_masks, walkable = sensor_library(device, 7.0)
    generator = torch.Generator(device=device).manual_seed(args.seed + 10_000)
    scale = torch.tensor(STATE_SCALE, device=device).view(1, 4, 1, 1)
    mode_names = tuple(
        f"{name}_bias{int(use_bias)}_{label}"
        for name in checkpoints for use_bias in (False, True)
        for label, _, _ in covariance_scales)
    mean_names = ("mean", "mean_bias_corrected")
    regions = ("all", "observed", "blind", "walkable_blind")
    metrics = blank_metrics(mode_names, regions)
    mean_se = {name: {region: 0.0 for region in regions} for name in mean_names}
    counts = {region: 0 for region in regions}

    with torch.inference_mode():
        for begin in range(0, len(rows), args.batch):
            x, y = data.batch(rows[begin:begin + args.batch])
            truth = y[:, 0]
            mean = surrogate_mean(mean_net, x)[:, 0]
            corrected = project_state(mean + bias[None])
            mask = random_sensor_mask(len(x), 3, sensor_masks, generator)
            observation = truth + torch.randn(
                truth.shape, device=device, generator=generator) * obs_std.view(1, 4, 1, 1)
            expanded = mask[:, None].expand_as(truth)
            walking = walkable[None, None].expand_as(truth)
            selections = {"all": torch.ones_like(expanded), "observed": expanded,
                          "blind": ~expanded, "walkable_blind": walking & ~expanded}
            normalized_truth = truth / scale
            for region, selection in selections.items():
                counts[region] += int(selection.sum())
                for name, forecast in (("mean", mean), ("mean_bias_corrected", corrected)):
                    square = ((forecast - truth) / scale).square()
                    mean_se[name][region] += float(square[selection].double().sum())

            for model_name, cov_net in cov_nets.items():
                factor, diagonal = cov_net(x[:, 0], mean, walkable)
                for label, factor_scale, diagonal_scale in covariance_scales:
                    scaled_factor = factor * math.sqrt(factor_scale)
                    scaled_diagonal = diagonal * diagonal_scale
                    variance = posterior_diag(
                        mean, scaled_factor, scaled_diagonal, observation, mask,
                        obs_std.square()).clamp_min(1e-12)
                    normalized_sigma = variance.sqrt() / scale
                    for use_bias, forecast in ((False, mean), (True, corrected)):
                        mode = f"{model_name}_bias{int(use_bias)}_{label}"
                        analysis = analysis_update(
                            forecast, scaled_factor, scaled_diagonal, observation, mask,
                            obs_std.square())
                        normalized_analysis = analysis / scale
                        square = (normalized_analysis - normalized_truth).square()
                        crps = crps_standard_normal(
                            normalized_analysis, normalized_sigma, normalized_truth)
                        absolute = (normalized_analysis - normalized_truth).abs()
                        for region, selection in selections.items():
                            acc = metrics[mode][region]
                            acc["se"] += float(square[selection].double().sum())
                            acc["spread"] += float(normalized_sigma[selection].double().sum())
                            acc["crps"] += float(crps[selection].double().sum())
                            acc["cov50"] += int((absolute[selection] <=
                                                  Z50 * normalized_sigma[selection]).sum())
                            acc["cov90"] += int((absolute[selection] <=
                                                  Z90 * normalized_sigma[selection]).sum())
            done = min(begin + args.batch, len(rows))
            if done == len(rows) or done % (args.batch * 8) == 0:
                print(f"[sweep] {done:,}/{len(rows):,}", flush=True)

    result_modes = {}
    for mode in mode_names:
        result_modes[mode] = {}
        for region in regions:
            acc, n = metrics[mode][region], counts[region]
            rmse = math.sqrt(acc["se"] / n)
            result_modes[mode][region] = {
                "rmse": rmse, "mean_spread": acc["spread"] / n,
                "spread_skill": (acc["spread"] / n) / rmse,
                "crps": acc["crps"] / n,
                "coverage50": acc["cov50"] / n, "coverage90": acc["cov90"] / n,
            }
    result_mean = {name: {region: math.sqrt(mean_se[name][region] / counts[region])
                          for region in regions} for name in mean_names}
    baseline_candidates = ("a32_bias0_alpha1", "a32_bias0_factor1_diag1")
    baseline = next((name for name in baseline_candidates if name in mode_names), mode_names[0])
    acceptable = [mode for mode in mode_names
                  if result_modes[mode]["all"]["rmse"] <=
                  1.01 * result_modes[baseline]["all"]["rmse"]]
    selection = {
        "best_all": min(mode_names, key=lambda mode: result_modes[mode]["all"]["rmse"]),
        "best_blind": min(mode_names, key=lambda mode: result_modes[mode]["blind"]["rmse"]),
        "best_calibrated_within_1pct_all": min(
            acceptable, key=lambda mode: abs(math.log(
                max(result_modes[mode]["all"]["spread_skill"], 1e-12)))),
        "reference": baseline,
    }
    result = {"config": vars(args), "pairs": len(rows), "mean_rmse": result_mean,
              "modes": result_modes, "selection": selection,
              "checkpoints": checkpoints, "train_args": train_args}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"selection": selection,
                      "selected_metrics": {key: result_modes[value]
                                           for key, value in selection.items()
                                           if key != "reference"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
