"""One-step Kalman evaluation of a learned paper-style local covariance U-Net.

The directed local maps are assembled and symmetrised, then projected onto the
matching climatological covariance eigenbasis.  Non-negative spectral weights
plus a diagonal remainder give a PSD representation accepted by the exact
Woodbury Kalman layer.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from crowdcore import config, navigation
from methods.enkf.checks.run_lowrank_unetkf import project_state
from methods.enkf.local_unetkf.model import LocalCovarianceUNet
from methods.enkf.local_unetkf.targets import extract_centered_patches
from methods.enkf.lowrank.kalman import analysis_update
from methods.enkf.lowrank.model import STATE_SCALE
from methods.enkf.lowrank.train import random_sensor_mask, sensor_library
from methods.enkf.surrogate.data import PairFrames
from methods.enkf.surrogate.model import load_pedpred3, surrogate_mean

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--covariance", default=os.path.join(
        HERE, "runs/local_unetkf_static/static_covariance.npz"))
    p.add_argument("--bias-file", default=os.path.join(
        HERE, "runs/local_unetkf_bias/bias_maps.npz"))
    p.add_argument("--pairs", type=int, default=4096)
    p.add_argument("--rank", type=int, default=0,
                   help="spectral projection rank; default raw=16, centered=64")
    p.add_argument("--patch-batch", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--days", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def sparse_indices(height, width, patch_size, valid, device):
    radius, n = patch_size // 2, height * width
    rows, columns, take = [], [], []
    linear = 0
    for p in range(n):
        row, col = divmod(p, width)
        for a in range(4):
            for b in range(4):
                for pi in range(patch_size):
                    for pj in range(patch_size):
                        if valid[row, col, pi, pj]:
                            q = (row + pi - radius) * width + col + pj - radius
                            rows.append(a * n + p)
                            columns.append(b * n + q)
                            take.append(linear)
                        linear += 1
    return (torch.tensor(rows, device=device), torch.tensor(columns, device=device),
            torch.tensor(take, device=device))


def load_all(args, device):
    checkpoint = torch.load(args.checkpoint, map_location=device)
    train_args = checkpoint["args"]
    target_kind = train_args["target"]
    patch_size = int(train_args["patch_size"])
    net = LocalCovarianceUNet(width=int(train_args["width"])).to(device)
    net.load_state_dict(checkpoint["model"])
    net.eval()
    mean_net = load_pedpred3(checkpoint.get("mean_ckpt", train_args["mean_ckpt"]), device).eval()
    with np.load(args.bias_file) as z:
        bias = torch.from_numpy(z["train_bias"]).to(device)
    with np.load(args.covariance) as z:
        valid_np = z["valid"]
        eigvals = torch.from_numpy(z[f"{target_kind}_eigenvalues"]).to(device)
        static_factor = torch.from_numpy(z[f"{target_kind}_factor"]).to(device)
    rank = args.rank or (16 if target_kind == "raw" else 64)
    if rank > len(static_factor):
        raise ValueError(f"rank {rank} exceeds stored basis rank {len(static_factor)}")
    values = eigvals[-rank:].clamp_min(1e-20)
    # Stored factor rows are sqrt(lambda_k) * eigenvector_k.
    basis = (static_factor[-rank:].reshape(rank, -1).T / values.sqrt()[None]).contiguous()
    valid = torch.from_numpy(valid_np).to(device)
    indices = sparse_indices(36, 12, patch_size, valid_np, device)
    return net, mean_net, bias, basis, valid, indices, target_kind, patch_size, train_args


def predict_maps(net, forecast, static, patch_size, patch_batch):
    n = forecast.shape[-2] * forecast.shape[-1]
    centres = torch.arange(n, device=forecast.device)[None]
    scale = forecast.new_tensor(STATE_SCALE).view(1, 4, 1, 1)
    state_patches = extract_centered_patches(forecast / scale, centres, patch_size)[0]
    map_patches = extract_centered_patches(static[None, None], centres, patch_size)[0]
    inputs = torch.cat((state_patches, map_patches), dim=1)
    pieces = [net(inputs[start:start + patch_batch])
              for start in range(0, n, patch_batch)]
    normalized = torch.cat(pieces).reshape(n, 4, 4, patch_size, patch_size)
    pair_scale = forecast.new_tensor(STATE_SCALE)
    return normalized * (pair_scale[:, None] * pair_scale[None, :]).view(1, 4, 4, 1, 1)


def psd_representation(local, basis, indices, patch_size):
    row_index, column_index, take = indices
    nstate = basis.shape[0]
    dense = local.new_zeros((nstate, nstate))
    dense[row_index, column_index] = local.reshape(-1)[take]
    asymmetry = (dense - dense.T).norm() / dense.norm().clamp_min(1e-30)
    dense = 0.5 * (dense + dense.T)
    projected = dense @ basis
    coefficients_raw = (basis * projected).sum(0)
    coefficients = coefficients_raw.clamp_min(0)
    factor_flat = basis * coefficients.sqrt()[None]
    radius = patch_size // 2
    local_diag = torch.stack(
        [local[:, a, a, radius, radius] for a in range(4)], dim=0).reshape(-1).clamp_min(1e-8)
    factor_diag = factor_flat.square().sum(1)
    row_scale = torch.minimum(torch.ones_like(local_diag),
                              (local_diag / factor_diag.clamp_min(1e-30)).sqrt())
    factor_flat = factor_flat * row_scale[:, None]
    factor_diag = factor_flat.square().sum(1)
    diagonal = (local_diag - factor_diag).clamp_min(1e-8)
    factor = factor_flat.T.reshape(basis.shape[1], 4, 36, 12)
    return factor, diagonal.reshape(4, 36, 12), {
        "asymmetry": float(asymmetry),
        "negative_spectral_fraction": float((coefficients_raw < 0).float().mean()),
        "spectral_trace": float(coefficients.sum()),
        "diagonal_trace": float(diagonal.sum()),
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("no GPU; submit a GPU job or add --allow-cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (net, mean_net, bias, basis, valid, indices, target_kind,
     patch_size, train_args) = load_all(args, device)
    data = PairFrames("valid", device, args.days, args.max_frames)
    count = min(args.pairs, data.n_pairs)
    rows = torch.linspace(0, data.n_pairs - 1, count, device=device).round().long().unique()
    static = torch.from_numpy(navigation.build_valid_mask_from_config()).to(device).float()
    sensor_masks, walkable = sensor_library(device, 7.0)
    obs_std = torch.tensor(config.get("observation", "obs_std"), device=device)
    generator = torch.Generator(device=device).manual_seed(args.seed + 10_000)
    scale = torch.tensor(STATE_SCALE, device=device).view(1, 4, 1, 1)
    modes = ("mean", "dynamic_diagonal", "dynamic_spectral")
    regions = ("all", "observed", "blind", "walkable_blind")
    sums = {mode: {region: 0.0 for region in regions} for mode in modes}
    counts = {region: 0 for region in regions}
    diagnostics = {key: 0.0 for key in (
        "asymmetry", "negative_spectral_fraction", "spectral_trace", "diagonal_trace")}
    tic = time.time()
    with torch.inference_mode():
        for number, row in enumerate(rows):
            x, y = data.batch(row[None])
            truth = y[:, 0]
            mean = surrogate_mean(mean_net, x)[:, 0]
            forecast = project_state(mean + bias[None]) if target_kind == "centered" else mean
            local = predict_maps(net, forecast, static, patch_size, args.patch_batch)
            factor, diagonal, diag = psd_representation(
                local, basis, indices, patch_size)
            for key in diagnostics:
                diagnostics[key] += diag[key]
            mask = random_sensor_mask(1, 3, sensor_masks, generator)
            observation = truth + torch.randn(
                truth.shape, device=device, generator=generator) * obs_std.view(1, 4, 1, 1)
            estimates = {
                "mean": forecast,
                "dynamic_diagonal": analysis_update(
                    forecast, torch.zeros_like(factor)[None], diagonal[None],
                    observation, mask, obs_std.square()),
                "dynamic_spectral": analysis_update(
                    forecast, factor[None], diagonal[None],
                    observation, mask, obs_std.square()),
            }
            expanded = mask[:, None].expand_as(truth)
            walking = walkable[None, None].expand_as(truth)
            selections = {"all": torch.ones_like(expanded), "observed": expanded,
                          "blind": ~expanded, "walkable_blind": walking & ~expanded}
            for region, selection in selections.items():
                counts[region] += int(selection.sum())
                for mode, estimate in estimates.items():
                    square = ((estimate - truth) / scale).square()
                    sums[mode][region] += float(square[selection].double().sum())
            if number == len(rows) - 1 or (number + 1) % 25 == 0:
                print(f"[eval] {number+1}/{len(rows)}, {time.time()-tic:.1f}s", flush=True)
    result = {
        "config": vars(args), "target": target_kind, "rank": basis.shape[1],
        "pairs": len(rows), "seconds": round(time.time() - tic, 3),
        "normalized_rmse": {
            mode: {region: (sums[mode][region] / counts[region]) ** 0.5 for region in regions}
            for mode in modes},
        "projection_diagnostics": {
            key: value / len(rows) for key, value in diagnostics.items()},
        "train_args": train_args,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
