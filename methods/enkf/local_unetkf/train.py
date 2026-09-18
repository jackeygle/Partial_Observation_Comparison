"""Train Lu-style local covariance maps from PedPred3 forecast residuals."""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from crowdcore import navigation
from methods.enkf.checks.run_lowrank_unetkf import project_state
from methods.enkf.local_unetkf.model import LocalCovarianceUNet
from methods.enkf.local_unetkf.targets import extract_centered_patches, sampled_normalized_targets
from methods.enkf.lowrank.model import STATE_SCALE
from methods.enkf.surrogate.data import PairFrames
from methods.enkf.surrogate.model import load_pedpred3, surrogate_mean

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=("raw", "centered"), required=True)
    p.add_argument("--mean-ckpt", default=os.path.join(HERE, "runs/surrogate_mean_s0/best.pt"))
    p.add_argument("--bias-file", default=os.path.join(HERE, "runs/local_unetkf_bias/bias_maps.npz"))
    p.add_argument("--out", default="")
    p.add_argument("--patch-size", type=int, default=15)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--frame-batch", type=int, default=16)
    p.add_argument("--centres-per-frame", type=int, default=4)
    p.add_argument("--train-pairs-per-epoch", type=int, default=131072)
    p.add_argument("--eval-pairs", type=int, default=4096)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--diag-penalty", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--days", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def make_examples(mean, truth, static, bias, target_kind, centres, patch_size):
    if target_kind == "centered":
        forecast = project_state(mean + bias[None])
        # Keep the supervised definition identical to the static teacher:
        # centred residual = (truth - frozen mean) - training-set bias.
        residual = truth - mean - bias[None]
    else:
        forecast = mean
        residual = truth - mean
    scale = forecast.new_tensor(STATE_SCALE).view(1, 4, 1, 1)
    forecast_patches = extract_centered_patches(forecast / scale, centres, patch_size)
    static_batch = static[None, None].expand(len(mean), -1, -1, -1)
    static_patches = extract_centered_patches(static_batch, centres, patch_size)
    inputs = torch.cat((forecast_patches, static_patches), dim=2)
    targets, valid = sampled_normalized_targets(residual, centres, patch_size)
    b, s = centres.shape
    return (inputs.reshape(b * s, 5, patch_size, patch_size),
            targets.reshape(b * s, 16, patch_size, patch_size),
            valid.reshape(b * s, 1, patch_size, patch_size))


def covariance_loss(prediction, target, valid, diag_penalty):
    expanded = valid.expand_as(prediction)
    mse = ((prediction - target).square() * expanded).sum() / expanded.sum().clamp_min(1)
    centre = prediction.shape[-1] // 2
    diagonal = prediction[:, (0, 5, 10, 15), centre, centre]
    penalty = diagonal.clamp_max(0).square().mean()
    return mse + diag_penalty * penalty, mse, penalty


def run_epoch(net, mean_net, data, rows, static, bias, args, generator, optimizer=None):
    training = optimizer is not None
    net.train(training)
    sums = torch.zeros(4, dtype=torch.float64, device=static.device)
    seen = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for begin in range(0, len(rows), args.frame_batch):
            batch_rows = rows[begin:begin + args.frame_batch]
            x, y = data.batch(batch_rows)
            with torch.no_grad():
                mean = surrogate_mean(mean_net, x)[:, 0]
            truth = y[:, 0]
            centres = torch.randint(
                truth.shape[-2] * truth.shape[-1],
                (len(x), args.centres_per_frame), device=truth.device, generator=generator)
            inputs, targets, valid = make_examples(
                mean, truth, static, bias, args.target, centres, args.patch_size)
            metrics = covariance_loss(net(inputs), targets, valid, args.diag_penalty)
            expanded = valid.expand_as(targets)
            zero_mse = (targets.square() * expanded).sum() / expanded.sum().clamp_min(1)
            if training:
                optimizer.zero_grad(set_to_none=True)
                metrics[0].backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                optimizer.step()
            count = len(inputs)
            values = (*metrics, zero_mse)
            sums += torch.stack([value.detach().double() for value in values]) * count
            seen += count
    return (sums / max(seen, 1)).cpu().tolist()


def main():
    args = parse_args()
    if args.patch_size % 2 != 1:
        raise SystemExit("--patch-size must be odd")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("no GPU; submit a GPU job or add --allow-cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = args.out or os.path.join(HERE, "runs", f"local_unetkf_{args.target}_s{args.seed}")
    os.makedirs(out, exist_ok=True)
    train = PairFrames("train", device, args.days, args.max_frames)
    valid = PairFrames("valid", device, args.days, args.max_frames)
    mean_net = load_pedpred3(args.mean_ckpt, device).eval()
    for parameter in mean_net.parameters():
        parameter.requires_grad_(False)
    with np.load(args.bias_file) as z:
        bias = torch.from_numpy(z["train_bias"]).to(device)
    static = torch.from_numpy(navigation.build_valid_mask_from_config()).to(device).float()
    net = LocalCovarianceUNet(width=args.width).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    train_gen = torch.Generator(device=device).manual_seed(args.seed + 100)
    eval_gen = torch.Generator(device=device).manual_seed(args.seed + 10_000)
    row_gen = torch.Generator().manual_seed(args.seed)
    start, best = 0, float("inf")
    last_path, best_path = os.path.join(out, "last.pt"), os.path.join(out, "best.pt")
    if args.resume and os.path.exists(last_path):
        checkpoint = torch.load(last_path, map_location=device)
        net.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start, best = checkpoint["epoch"] + 1, checkpoint["best"]
    with open(os.path.join(out, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[data] train={train.n_pairs:,}, valid={valid.n_pairs:,}, device={device}", flush=True)
    train_count = min(train.n_pairs, args.train_pairs_per_epoch)
    eval_count = min(valid.n_pairs, args.eval_pairs)
    eval_rows = torch.linspace(0, valid.n_pairs - 1, eval_count).round().long().to(device)
    for epoch in range(start, args.epochs):
        tic = time.time()
        train_rows = torch.randperm(train.n_pairs, generator=row_gen)[:train_count].to(device)
        train_metrics = run_epoch(net, mean_net, train, train_rows, static, bias,
                                  args, train_gen, optimizer)
        # Resetting makes the validation centre samples identical across epochs.
        eval_gen.manual_seed(args.seed + 10_000)
        valid_metrics = run_epoch(net, mean_net, valid, eval_rows, static, bias,
                                  args, eval_gen)
        scheduler.step(valid_metrics[0])
        record = {
            "epoch": epoch, "seconds": round(time.time() - tic, 2),
            "lr": optimizer.param_groups[0]["lr"],
            "train": dict(zip(("loss", "covariance_mse", "negative_diag_penalty",
                               "zero_prediction_mse"), train_metrics)),
            "valid": dict(zip(("loss", "covariance_mse", "negative_diag_penalty",
                               "zero_prediction_mse"), valid_metrics)),
        }
        print(json.dumps(record), flush=True)
        with open(os.path.join(out, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")
        payload = {"model": net.state_dict(), "optimizer": optimizer.state_dict(),
                   "scheduler": scheduler.state_dict(), "epoch": epoch, "best": best,
                   "args": vars(args), "mean_ckpt": args.mean_ckpt}
        if valid_metrics[0] < best:
            best = valid_metrics[0]
            payload["best"] = best
            torch.save(payload, best_path)
        payload["best"] = best
        torch.save(payload, last_path)
    print(f"[done] best={best:.8g} -> {best_path}", flush=True)


if __name__ == "__main__":
    main()
