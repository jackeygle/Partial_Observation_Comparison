"""Train ``B(x) = U U^T + diag(d)`` by forecast likelihood only.

This trainer deliberately contains no observation masks, Kalman update, or
analysis loss.  PedPred3 is frozen; only the covariance U-Net is optimized.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from crowdcore import navigation
from methods.enkf.lowrank.kalman import gaussian_nll, masked_gaussian_nll, student_t_nll
from methods.enkf.lowrank.model import CovarianceUNet
from methods.enkf.surrogate.data import PairFrames
from methods.enkf.surrogate.model import EMPTY_DENSITY, load_pedpred3, surrogate_mean

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mean-ckpt", default=os.path.join(HERE, "runs/surrogate_mean_s0/best.pt"))
    p.add_argument("--cov-ckpt", default="",
                   help="initialize covariance U-Net weights from an existing checkpoint")
    p.add_argument("--input-frames", type=int, default=1,
                   help="history length consumed by the frozen PedPred3 mean")
    p.add_argument("--target-kind", choices=("raw", "centered"), default="raw")
    p.add_argument("--bias-file", default=os.path.join(
        HERE, "runs/local_unetkf_bias/bias_maps.npz"))
    p.add_argument("--scope", choices=("all", "walkable", "defined-walkable"),
                   default="walkable")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--diagonal-floor", type=float, default=1e-5)
    p.add_argument("--diagonal-floor-fraction", type=float, default=0.0)
    p.add_argument("--likelihood", choices=("gaussian", "student-t"), default="gaussian")
    p.add_argument("--student-df", type=float, default=3.0)
    p.add_argument("--variance-anchor-weight", type=float, default=0.0)
    p.add_argument("--marginal-weight", type=float, default=0.0)
    p.add_argument("--marginal-likelihood", choices=("gaussian", "student-t"),
                   default="gaussian")
    p.add_argument("--marginal-student-df", type=float, default=5.0,
                   help="df > 2; marginal variance remains the distribution variance")
    p.add_argument("--occupied-weight", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--days", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--eval-pairs", type=int, default=2048)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def selected_nll(truth, location, factor, diagonal, spatial_mask):
    """Gaussian NLL on a fixed spatial scope, retaining all four channels."""
    if spatial_mask is None:
        return gaussian_nll(truth, location, factor, diagonal)
    # gaussian_nll accepts arbitrary H,W. Keep a singleton W after selecting cells.
    truth = truth[..., spatial_mask].unsqueeze(-1)
    location = location[..., spatial_mask].unsqueeze(-1)
    diagonal = diagonal[..., spatial_mask].unsqueeze(-1)
    factor = factor[..., spatial_mask].unsqueeze(-1)
    return gaussian_nll(truth, location, factor, diagonal)


def batch_metrics(cov_net, mean_net, x, target, static_mask, spatial_mask, bias,
                  variance_anchor, args):
    with torch.no_grad():
        raw_mean = surrogate_mean(mean_net, x)[:, 0]
        location = raw_mean if bias is None else raw_mean + bias[None]
    truth = target[:, 0]
    # The covariance sees the state at forecast origin. With a multi-frame
    # history this is the final input frame, not the oldest one.
    factor, diagonal = cov_net(x[:, -1], location, static_mask)
    defined = None
    if args.scope == "defined-walkable":
        occupied = truth[:, 0] > 0
        defined = torch.stack((torch.ones_like(occupied), occupied, occupied,
                               truth[:, 3] > 0), dim=1)
        defined = defined & static_mask[None, None].bool()
    if defined is not None:
        if args.likelihood != "gaussian":
            raise ValueError("defined-walkable currently supports Gaussian likelihood only")
        likelihood = masked_gaussian_nll(truth, location, factor, diagonal, defined)
    elif args.likelihood == "gaussian":
        likelihood = selected_nll(truth, location, factor, diagonal, spatial_mask)
    else:
        if spatial_mask is None:
            likelihood = student_t_nll(
                truth, location, factor, diagonal, args.student_df)
        else:
            likelihood = student_t_nll(
                truth[..., spatial_mask].unsqueeze(-1),
                location[..., spatial_mask].unsqueeze(-1),
                factor[..., spatial_mask].unsqueeze(-1),
                diagonal[..., spatial_mask].unsqueeze(-1), args.student_df)
    if defined is not None:
        full_lowrank_diagonal = factor.square().sum(dim=1)
        selected_d = diagonal[defined]
        lowrank_diagonal = full_lowrank_diagonal[defined]
    elif spatial_mask is None:
        selected_d = diagonal
        selected_u = factor
        lowrank_diagonal = selected_u.square().sum(dim=1)
    else:
        selected_d = diagonal[..., spatial_mask]
        selected_u = factor[..., spatial_mask]
        lowrank_diagonal = selected_u.square().sum(dim=1)
    marginal_variance = selected_d + lowrank_diagonal
    anchor_penalty = likelihood.new_zeros(())
    if variance_anchor is not None:
        if defined is not None:
            selected_anchor = variance_anchor[None].expand_as(diagonal)[defined]
        else:
            selected_anchor = (variance_anchor if spatial_mask is None
                               else variance_anchor[..., spatial_mask])
        anchor_penalty = (marginal_variance.clamp_min(1e-8).log()
                          - selected_anchor[None].clamp_min(1e-8).log()).square().mean()
    marginal_penalty = likelihood.new_zeros(())
    if args.marginal_weight:
        if defined is not None:
            selected_truth = truth[defined]
            selected_location = location[defined]
        else:
            selected_truth = truth if spatial_mask is None else truth[..., spatial_mask]
            selected_location = location if spatial_mask is None else location[..., spatial_mask]
        residual = selected_truth - selected_location
        if defined is not None:
            weights = torch.ones_like(residual)
        else:
            occupied = selected_truth[:, 0:1] >= EMPTY_DENSITY
            weights = torch.where(occupied, args.occupied_weight, 1.0).expand_as(residual)
        marginal_variance = marginal_variance.clamp_min(1e-8)
        if args.marginal_likelihood == "gaussian":
            point_nll = 0.5 * (residual.square() / marginal_variance
                               + marginal_variance.log())
        else:
            # Parameterize the Student-t by its actual variance so the learned
            # covariance can still be consumed directly by the Kalman update.
            # For df > 2, variance = scale^2 * df / (df - 2).
            nu = residual.new_tensor(args.marginal_student_df)
            scale2 = marginal_variance * (nu - 2) / nu
            point_nll = (torch.lgamma(0.5 * nu) - torch.lgamma(0.5 * (nu + 1))
                         + 0.5 * torch.log(nu * torch.pi * scale2)
                         + 0.5 * (nu + 1) * torch.log1p(
                             residual.square() / (nu * scale2)))
        marginal_penalty = (weights * point_nll).sum() / weights.sum().clamp_min(1)
    loss = (likelihood + args.variance_anchor_weight * anchor_penalty
            + args.marginal_weight * marginal_penalty)
    return (loss, likelihood, selected_d.mean(), lowrank_diagonal.mean(),
            anchor_penalty, marginal_penalty)


def main():
    args = parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("no GPU; submit a GPU job or add --allow-cpu for a smoke run")
    if args.epochs < 1 or args.batch < 1 or args.eval_pairs < 1 or args.input_frames < 1:
        raise SystemExit("epochs, batch, eval-pairs, and input-frames must be positive")
    if args.marginal_likelihood == "student-t" and args.marginal_student_df <= 2:
        raise SystemExit("--marginal-student-df must be > 2 to define covariance")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    load_tic = time.time()
    train, valid = (PairFrames(split, device, args.days, args.max_frames,
                               input_frames=args.input_frames, output_frames=1)
                    for split in ("train", "valid"))
    print(f"[data] train={train.n_pairs:,}, valid={valid.n_pairs:,}, "
          f"load={time.time()-load_tic:.1f}s, device={device}, "
          f"frames={args.input_frames}->1", flush=True)

    mean_net = load_pedpred3(args.mean_ckpt, device).eval()
    for parameter in mean_net.parameters():
        parameter.requires_grad_(False)
    cov_net = CovarianceUNet(
        rank=args.rank, width=args.width, diagonal_floor=args.diagonal_floor,
        diagonal_floor_fraction=args.diagonal_floor_fraction).to(device)
    if args.cov_ckpt:
        initialization = torch.load(args.cov_ckpt, map_location=device)
        cov_net.load_state_dict(initialization["model"])
        print(f"[init] covariance={args.cov_ckpt}", flush=True)
    optimizer = torch.optim.AdamW(cov_net.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
    perm_gen = torch.Generator().manual_seed(args.seed)
    static_mask = torch.from_numpy(navigation.build_valid_mask_from_config()).to(device)
    spatial_mask = static_mask.bool() if args.scope == "walkable" else None
    bias = None
    variance_anchor = None
    if args.target_kind == "centered":
        with np.load(args.bias_file) as z:
            bias = torch.from_numpy(z["train_bias"]).to(device)
    if args.variance_anchor_weight:
        with np.load(args.bias_file) as z:
            key = "train_centered_mse" if args.target_kind == "centered" else "train_mse"
            variance_anchor = torch.from_numpy(z[key]).to(device)
    print(f"[objective] pure {args.likelihood} likelihood; target={args.target_kind}; "
          f"scope={args.scope}; floor_fraction={args.diagonal_floor_fraction:g}; "
          f"anchor={args.variance_anchor_weight:g}; marginal={args.marginal_weight:g}; "
          f"marginal_likelihood={args.marginal_likelihood}; "
          f"marginal_df={args.marginal_student_df:g}; "
          "Kalman update=disabled", flush=True)

    start, best = 0, float("inf")
    last_path = os.path.join(args.out, "last.pt")
    best_path = os.path.join(args.out, "best.pt")
    if args.resume and os.path.exists(last_path):
        checkpoint = torch.load(last_path, map_location=device)
        cov_net.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        # map_location=device moves serialized RNG state to CUDA, while the
        # CPU permutation generator requires a CPU ByteTensor.
        perm_gen.set_state(checkpoint["perm_gen"].cpu())
        start, best = checkpoint["epoch"] + 1, checkpoint["best"]

    for epoch in range(start, args.epochs):
        cov_net.train()
        permutation = torch.randperm(train.n_pairs, generator=perm_gen).to(device)
        train_sums = torch.zeros(6, dtype=torch.float64, device=device)
        count, skipped, tic = 0, 0, time.time()
        for begin in range(0, train.n_pairs, args.batch):
            x, target = train.batch(permutation[begin:begin + args.batch])
            metrics = batch_metrics(
                cov_net, mean_net, x, target, static_mask, spatial_mask, bias,
                variance_anchor, args)
            loss = metrics[0]
            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                skipped += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cov_net.parameters(), 5.0)
            optimizer.step()
            train_sums += torch.stack([value.detach().double() for value in metrics]) * len(x)
            count += len(x)

        cov_net.eval()
        eval_n = min(valid.n_pairs, args.eval_pairs)
        rows = torch.linspace(0, valid.n_pairs - 1, eval_n, device=device).round().long().unique()
        valid_sums = torch.zeros(6, dtype=torch.float64, device=device)
        with torch.no_grad():
            for begin in range(0, len(rows), args.batch):
                x, target = valid.batch(rows[begin:begin + args.batch])
                metrics = batch_metrics(
                    cov_net, mean_net, x, target, static_mask, spatial_mask, bias,
                    variance_anchor, args)
                valid_sums += torch.stack([value.double() for value in metrics]) * len(x)
        train_values = (train_sums / max(count, 1)).cpu().tolist()
        valid_values = (valid_sums / max(len(rows), 1)).cpu().tolist()
        scheduler.step(valid_values[0])
        names = ("loss", "likelihood", "diagonal_mean", "lowrank_diagonal_mean",
                 "anchor_penalty", "marginal_penalty")
        record = {
            "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
            "skipped": skipped, "seconds": round(time.time() - tic, 1),
            "train": dict(zip(names, train_values)),
            "valid": dict(zip(names, valid_values)),
        }
        print(json.dumps(record), flush=True)
        with open(os.path.join(args.out, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")
        payload = {
            "model": cov_net.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "perm_gen": perm_gen.get_state(),
            "epoch": epoch, "best": best, "args": vars(args),
            "mean_ckpt": args.mean_ckpt,
            "bias": None if bias is None else bias.detach().cpu(),
            "objective": "pure_gaussian_nll",
        }
        if valid_values[0] < best:
            best = valid_values[0]
            payload["best"] = best
            torch.save(payload, best_path)
        payload["best"] = best
        torch.save(payload, last_path)
    print(f"[done] best validation NLL={best:.6g} -> {best_path}", flush=True)


if __name__ == "__main__":
    main()
