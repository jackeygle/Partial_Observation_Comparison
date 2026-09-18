"""Measure PedPred3's systematic one-step forecast bias before covariance learning."""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from crowdcore import navigation
from methods.enkf.surrogate.data import PairFrames
from methods.enkf.surrogate.model import EMPTY_DENSITY, load_pedpred3, surrogate_mean

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHANNELS = ("density", "vx", "vy", "var")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mean-ckpt", default=os.path.join(HERE, "runs/surrogate_mean_s0/best.pt"))
    p.add_argument("--outdir", default=os.path.join(HERE, "runs/local_unetkf_bias"))
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--days", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--max-pairs", type=int, default=0,
                   help="evenly sample this many pairs per split; 0 uses every pair")
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def selected_rows(data, max_pairs, device):
    if not max_pairs or max_pairs >= data.n_pairs:
        return torch.arange(data.n_pairs, device=device)
    # Even spacing covers every day/time range and is deterministic.
    return torch.linspace(0, data.n_pairs - 1, max_pairs, device=device).round().long().unique()


def region_accumulator(device):
    return {name: {"sum": torch.zeros(4, dtype=torch.float64, device=device),
                   "sq": torch.zeros(4, dtype=torch.float64, device=device),
                   "count": torch.zeros((), dtype=torch.float64, device=device)}
            for name in ("all", "occupied", "empty", "walkable", "nonwalkable")}


def add_region(stats, name, error, mask):
    # error B,C,H,W; mask B,H,W or H,W. Counts are spatial samples per channel.
    if mask.ndim == 2:
        mask = mask[None].expand(error.shape[0], -1, -1)
    values = error * mask[:, None]
    stats[name]["sum"] += values.double().sum(dim=(0, 2, 3))
    stats[name]["sq"] += values.double().square().sum(dim=(0, 2, 3))
    stats[name]["count"] += mask.sum(dtype=torch.float64)


def finish_regions(stats):
    result = {}
    for name, values in stats.items():
        count = values["count"].clamp_min(1)
        bias = values["sum"] / count
        rmse = (values["sq"] / count).sqrt()
        result[name] = {"count_per_channel": int(values["count"].item()),
                        "bias": dict(zip(CHANNELS, bias.cpu().tolist())),
                        "rmse": dict(zip(CHANNELS, rmse.cpu().tolist()))}
    return result


def evaluate_split(split, net, device, args, train_bias=None):
    load_t = time.time()
    data = PairFrames(split, device, args.days, args.max_frames)
    rows = selected_rows(data, args.max_pairs, device)
    walkable = torch.from_numpy(navigation.build_valid_mask_from_config()).to(device)
    raw_stats = region_accumulator(device)
    corrected_stats = region_accumulator(device) if train_bias is not None else None
    sum_map = torch.zeros((4, *data.frames.shape[-2:]), dtype=torch.float64, device=device)
    sq_map = torch.zeros_like(sum_map)
    tic = time.time()
    with torch.inference_mode():
        for begin in range(0, len(rows), args.batch):
            x, target = data.batch(rows[begin:begin + args.batch])
            truth = target[:, 0]
            error = truth - surrogate_mean(net, x)[:, 0]
            sum_map += error.double().sum(0)
            sq_map += error.double().square().sum(0)
            masks = {
                "all": torch.ones_like(truth[:, 0], dtype=torch.bool),
                "occupied": truth[:, 0] >= EMPTY_DENSITY,
                "empty": truth[:, 0] < EMPTY_DENSITY,
                "walkable": walkable,
                "nonwalkable": ~walkable,
            }
            for name, mask in masks.items():
                add_region(raw_stats, name, error, mask)
                if corrected_stats is not None:
                    add_region(corrected_stats, name, error - train_bias[None], mask)
            done = min(begin + args.batch, len(rows))
            if done == len(rows) or done % (args.batch * 100) == 0:
                print(f"[{split}] {done:,}/{len(rows):,} pairs", flush=True)
    n = len(rows)
    bias_map = sum_map / n
    mse_map = sq_map / n
    centered_mse_map = (mse_map - bias_map.square()).clamp_min(0)
    per_channel_rmse = mse_map.mean((1, 2)).sqrt()
    per_channel_centered_rmse = centered_mse_map.mean((1, 2)).sqrt()
    per_channel_bias_rms = bias_map.square().mean((1, 2)).sqrt()
    summary = {
        "split": split,
        "available_pairs": data.n_pairs,
        "evaluated_pairs": n,
        "days": data.n_days,
        "load_seconds": round(tic - load_t, 3),
        "forecast_seconds": round(time.time() - tic, 3),
        "per_channel": {
            channel: {
                "forecast_rmse": float(per_channel_rmse[i]),
                "bias_rms": float(per_channel_bias_rms[i]),
                "bias_ratio": float(per_channel_bias_rms[i] / per_channel_rmse[i].clamp_min(1e-12)),
                "intrinsic_centered_rmse": float(per_channel_centered_rmse[i]),
            } for i, channel in enumerate(CHANNELS)
        },
        "regions_raw": finish_regions(raw_stats),
    }
    if corrected_stats is not None:
        summary["regions_after_train_bias_removal"] = finish_regions(corrected_stats)
    arrays = {"bias_map": bias_map.float().cpu().numpy(),
              "mse_map": mse_map.float().cpu().numpy(),
              "centered_mse_map": centered_mse_map.float().cpu().numpy()}
    del data
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary, arrays


def main():
    args = parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("no GPU; submit the H200 job or add --allow-cpu for a smoke run")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)
    net = load_pedpred3(args.mean_ckpt, device).eval()
    train_summary, train_arrays = evaluate_split("train", net, device, args)
    train_bias = torch.from_numpy(train_arrays["bias_map"]).to(device)
    valid_summary, valid_arrays = evaluate_split("valid", net, device, args, train_bias)
    np.savez_compressed(os.path.join(args.outdir, "bias_maps.npz"),
                        train_bias=train_arrays["bias_map"],
                        train_mse=train_arrays["mse_map"],
                        train_centered_mse=train_arrays["centered_mse_map"],
                        valid_bias=valid_arrays["bias_map"],
                        valid_mse=valid_arrays["mse_map"],
                        valid_centered_mse=valid_arrays["centered_mse_map"])
    report = {"config": vars(args), "device": str(device),
              "channels": CHANNELS, "train": train_summary, "valid": valid_summary}
    with open(os.path.join(args.outdir, "bias_audit.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"train": train_summary["per_channel"],
                      "valid": valid_summary["per_channel"]}, indent=2), flush=True)
    print(f"[done] {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
