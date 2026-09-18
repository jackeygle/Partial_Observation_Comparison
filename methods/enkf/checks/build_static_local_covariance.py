"""Build and diagnose raw/centered climatological local covariance teachers."""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from methods.enkf.local_unetkf.targets import accumulate_static_local_covariance
from methods.enkf.lowrank.model import STATE_SCALE
from methods.enkf.surrogate.data import PairFrames
from methods.enkf.surrogate.model import load_pedpred3, surrogate_mean

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHANNELS = ("density", "vx", "vy", "var")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mean-ckpt", default=os.path.join(HERE, "runs/surrogate_mean_s0/best.pt"))
    p.add_argument("--bias-file", default=os.path.join(HERE, "runs/local_unetkf_bias/bias_maps.npz"))
    p.add_argument("--outdir", default=os.path.join(HERE, "runs/local_unetkf_static"))
    p.add_argument("--patch-size", type=int, default=15)
    p.add_argument("--sample-pairs", type=int, default=32768)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max-rank", type=int, default=256)
    p.add_argument("--days", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def sampled_rows(n_pairs, requested, device):
    count = min(n_pairs, requested) if requested else n_pairs
    return torch.linspace(0, n_pairs - 1, count, device=device).round().long().unique()


def padding_valid(height, width, patch_size, device):
    radius = patch_size // 2
    ones = torch.ones((1, 1, height, width), device=device)
    unfolded = F.unfold(ones, patch_size, padding=radius)
    return unfolded.squeeze(0).transpose(0, 1).reshape(
        height, width, patch_size, patch_size).bool()


def local_to_dense(local, valid):
    """Assemble channel-major dense rows from ``(H,W,4,4,P,P)`` local maps."""
    height, width, channels, _, patch_size, _ = local.shape
    radius, n_cells = patch_size // 2, height * width
    dense = local.new_zeros((channels * n_cells, channels * n_cells))
    channel_offsets = torch.arange(channels, device=local.device) * n_cells
    for row in range(height):
        for col in range(width):
            centre = row * width + col
            for pi in range(patch_size):
                for pj in range(patch_size):
                    if not bool(valid[row, col, pi, pj]):
                        continue
                    nr, nc = row + pi - radius, col + pj - radius
                    neighbour = nr * width + nc
                    rows = channel_offsets + centre
                    columns = channel_offsets + neighbour
                    dense[rows[:, None], columns[None, :]] = local[row, col, :, :, pi, pj]
    return dense


def factorize_and_report(name, local, valid, max_rank):
    dense_directed = local_to_dense(local, valid)
    asymmetry = (dense_directed - dense_directed.T).norm() / dense_directed.norm().clamp_min(1e-30)
    dense = 0.5 * (dense_directed + dense_directed.T)
    eigvals, eigvecs = torch.linalg.eigh(dense)
    positive = eigvals.clamp_min(0)
    positive_sum = positive.sum().clamp_min(1e-30)
    keep = min(max_rank, int((positive > 0).sum()))
    top_values = positive[-keep:]
    top_vectors = eigvecs[:, -keep:]
    factor_flat = top_vectors * top_values.sqrt()[None]
    psd_diag = (eigvecs.square() * positive[None]).sum(1)
    residual_diag = (psd_diag - factor_flat.square().sum(1)).clamp_min(1e-10)
    n_cells = local.shape[0] * local.shape[1]
    factor = factor_flat.T.reshape(keep, 4, local.shape[0], local.shape[1])
    diagonal = residual_diag.reshape(4, local.shape[0], local.shape[1])
    trace_by_rank = {}
    for rank in (8, 16, 32, 64, 128, 256):
        if rank <= len(positive):
            trace_by_rank[str(rank)] = float(positive[-rank:].sum() / positive_sum)
    report = {
        "name": name,
        "directed_relative_asymmetry": float(asymmetry),
        "min_eigenvalue": float(eigvals[0]),
        "max_eigenvalue": float(eigvals[-1]),
        "negative_eigenvalues": int((eigvals < -1e-10).sum()),
        "negative_trace_fraction": float((-eigvals.clamp_max(0).sum()) / positive_sum),
        "positive_numerical_rank": int((positive > positive[-1] * 1e-8).sum()),
        "psd_projection_trace": float(positive_sum),
        "saved_rank": keep,
        "saved_rank_trace_fraction": float(top_values.sum() / positive_sum),
        "top_rank_trace_fraction": trace_by_rank,
        "mean_std_by_channel": {
            channel: float(psd_diag.reshape(4, n_cells)[i].mean().sqrt())
            for i, channel in enumerate(CHANNELS)
        },
    }
    arrays = {"local": local.float().cpu().numpy(),
              "factor": factor.float().cpu().numpy(),
              "diagonal": diagonal.float().cpu().numpy(),
              "eigenvalues": eigvals.float().cpu().numpy()}
    return report, arrays


def main():
    args = parse_args()
    if args.patch_size % 2 != 1:
        raise SystemExit("--patch-size must be odd")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("no GPU; submit the H200 job or add --allow-cpu for a smoke run")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)
    with np.load(args.bias_file) as z:
        bias = torch.from_numpy(z["train_bias"]).to(device)
    data = PairFrames("train", device, args.days, args.max_frames)
    rows = sampled_rows(data.n_pairs, args.sample_pairs, device)
    net = load_pedpred3(args.mean_ckpt, device).eval()
    height, width = data.frames.shape[-2:]
    shape = (height, width, 4, 4, args.patch_size, args.patch_size)
    raw_sum = torch.zeros(shape, dtype=torch.float64, device=device)
    centered_sum = torch.zeros_like(raw_sum)
    tic = time.time()
    with torch.inference_mode():
        for begin in range(0, len(rows), args.batch):
            x, target = data.batch(rows[begin:begin + args.batch])
            error = target[:, 0] - surrogate_mean(net, x)[:, 0]
            raw_sum = accumulate_static_local_covariance(raw_sum, error, args.patch_size)
            centered_sum = accumulate_static_local_covariance(
                centered_sum, error - bias[None], args.patch_size)
            done = min(begin + args.batch, len(rows))
            if done == len(rows) or done % (args.batch * 50) == 0:
                print(f"[covariance] {done:,}/{len(rows):,} pairs", flush=True)
    valid = padding_valid(height, width, args.patch_size, device)
    raw = raw_sum / len(rows)
    centered = centered_sum / len(rows)
    # Padded entries are already zero, but explicitly preserve the contract.
    raw *= valid[:, :, None, None]
    centered *= valid[:, :, None, None]
    reports, output = {}, {"valid": valid.cpu().numpy()}
    for name, covariance in (("raw", raw), ("centered", centered)):
        print(f"[eigh] {name}", flush=True)
        reports[name], arrays = factorize_and_report(name, covariance, valid, args.max_rank)
        output.update({f"{name}_{key}": value for key, value in arrays.items()})
    scale = torch.tensor(STATE_SCALE, dtype=torch.float64, device=device)
    reports["normalization_scale"] = dict(zip(CHANNELS, scale.cpu().tolist()))
    report = {"config": vars(args), "device": str(device),
              "available_pairs": data.n_pairs, "sampled_pairs": len(rows),
              "seconds": round(time.time() - tic, 3), "diagnostics": reports}
    np.savez_compressed(os.path.join(args.outdir, "static_covariance.npz"), **output)
    with open(os.path.join(args.outdir, "static_covariance.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(reports, indent=2), flush=True)
    print(f"[done] {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
