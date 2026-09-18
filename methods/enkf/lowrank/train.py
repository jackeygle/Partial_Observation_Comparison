"""Train a low-rank covariance U-Net through a differentiable Kalman update.

PedPred3 can remain frozen (the original experiment), or its output head/full
network can be jointly fine-tuned through the analysis loss. Training samples
legal robot positions and uses their exact map/line-of-sight footprints;
evaluation on moving-robot masks is done by ``checks/run_lowrank_unetkf.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from crowdcore import config, navigation
from methods.enkf.lowrank.kalman import analysis_update, gaussian_nll
from methods.enkf.lowrank.model import CovarianceUNet, STATE_SCALE
from methods.enkf.surrogate.data import PairFrames
from methods.enkf.surrogate.model import load_pedpred3, surrogate_mean

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # methods/enkf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mean-ckpt", default=os.path.join(HERE, "runs/surrogate_mean_s0/best.pt"))
    p.add_argument("--cov-ckpt", default="", help="initialize covariance U-Net from a checkpoint")
    p.add_argument("--mean-train", choices=("frozen", "head", "full"), default="frozen")
    p.add_argument("--mean-lr", type=float, default=1e-5)
    p.add_argument("--forecast-weight", type=float, default=0.0)
    p.add_argument("--input-frames", type=int, default=1,
                   help="causal state-history length consumed by PedPred3")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--nll-weight", type=float, default=0.01)
    p.add_argument("--num-agents", type=int, default=3)
    p.add_argument("--sensing-radius", type=float, default=7.0)
    p.add_argument("--days", type=int, default=0, help="debug: first N days per split")
    p.add_argument("--max-frames", type=int, default=0, help="debug: first N frames per day")
    p.add_argument("--eval-pairs", type=int, default=2048)
    p.add_argument("--out", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def sensor_library(device, radius):
    """Every legal robot position's exact radius + map line-of-sight mask."""
    valid = navigation.build_valid_mask_from_config()
    visibility = navigation.cell_visibility(radius)[np.flatnonzero(valid)]
    library = torch.from_numpy(visibility.reshape(-1, *valid.shape)).to(device)
    return library, torch.from_numpy(valid).to(device)


def random_sensor_mask(batch, num_agents, library, generator):
    """Union exact sensor footprints at independently sampled legal robot positions."""
    pick = torch.randint(len(library), (batch, num_agents), device=library.device, generator=generator)
    return library[pick].any(dim=1)


def batch_loss(cov_net, mean_net, x, target, obs_std, static_mask, masks, generator, args):
    if args.mean_train == "frozen":
        with torch.no_grad():
            mean = surrogate_mean(mean_net, x)[:, 0]
    else:
        mean = surrogate_mean(mean_net, x)[:, 0]
    x_grid, target_grid = x[:, -1], target[:, 0]
    factor, diagonal = cov_net(x_grid, mean, static_mask)
    mask = random_sensor_mask(len(x), args.num_agents, masks, generator)
    noise = torch.randn(target_grid.shape, device=target.device, dtype=target.dtype,
                        generator=generator) * obs_std.view(1, -1, 1, 1)
    observation = target_grid + noise
    analysis = analysis_update(mean, factor, diagonal, observation, mask, obs_std.square())
    scale = torch.as_tensor(STATE_SCALE, device=target.device, dtype=target.dtype).view(1, -1, 1, 1)
    sq = ((analysis - target_grid) / scale).square()
    analysis_loss = sq.mean()
    forecast_loss = ((mean - target_grid) / scale).square().mean()
    # Experiment A is the pure end-to-end analysis objective.  Do not merely
    # multiply NLL by zero: evaluating it would still perform its Cholesky
    # factorisation and can fail even though it contributes no gradient/loss.
    if args.nll_weight:
        nll = gaussian_nll(target_grid, mean, factor, diagonal)
    else:
        nll = analysis_loss.new_zeros(())
    blind = (~mask)[:, None].expand_as(sq)
    observed = mask[:, None].expand_as(sq)
    blind_mse = sq[blind].mean() if blind.any() else sq.new_zeros(())
    observed_mse = sq[observed].mean() if observed.any() else sq.new_zeros(())
    total = (analysis_loss + args.nll_weight * nll
             + args.forecast_weight * forecast_loss)
    return total, analysis_loss, forecast_loss, nll, blind_mse, observed_mse


def main():
    args = parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("no GPU; use the sbatch launcher or add --allow-cpu for a smoke run")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = args.out or os.path.join(HERE, "runs", f"lowrank_r{args.rank}_s{args.seed}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    t0 = time.time()
    train, valid = (PairFrames(split, device, args.days, args.max_frames,
                               input_frames=args.input_frames)
                    for split in ("train", "valid"))
    print(f"[data] train={train.n_pairs:,}, valid={valid.n_pairs:,}, load={time.time()-t0:.0f}s, "
          f"device={device}", flush=True)

    mean_net = load_pedpred3(args.mean_ckpt, device).eval()
    for parameter in mean_net.parameters():
        parameter.requires_grad_(False)
    if args.mean_train == "head":
        for parameter in mean_net.forecaster[9].parameters():
            parameter.requires_grad_(True)
    elif args.mean_train == "full":
        for parameter in mean_net.parameters():
            parameter.requires_grad_(True)
    cov_net = CovarianceUNet(rank=args.rank, width=args.width).to(device)
    if args.cov_ckpt:
        covariance_checkpoint = torch.load(args.cov_ckpt, map_location=device)
        cov_net.load_state_dict(covariance_checkpoint["model"])
    parameter_groups = [{"params": cov_net.parameters(), "lr": args.lr, "name": "covariance"}]
    mean_parameters = [parameter for parameter in mean_net.parameters() if parameter.requires_grad]
    if mean_parameters:
        parameter_groups.append({"params": mean_parameters, "lr": args.mean_lr, "name": "mean"})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
    train_gen = torch.Generator(device=device).manual_seed(args.seed + 100)
    perm_gen = torch.Generator().manual_seed(args.seed)
    obs_std = torch.tensor(config.get("observation", "obs_std"), device=device)
    masks, static_mask = sensor_library(device, args.sensing_radius)
    print(f"[mask] {len(masks)} legal centers, random-union coverage proxy; static walkable "
          f"cells={int(static_mask.sum())}", flush=True)
    print(f"[joint] mean_train={args.mean_train}; cov_lr={args.lr:g}; "
          f"mean_lr={args.mean_lr:g}; forecast_weight={args.forecast_weight:g}; "
          f"input_frames={args.input_frames}; cov_init={args.cov_ckpt or 'random'}", flush=True)
    start, best = 0, float("inf")
    last_path, best_path = os.path.join(out, "last.pt"), os.path.join(out, "best.pt")
    if args.resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device)
        cov_net.load_state_dict(ckpt["model"])
        if args.mean_train != "frozen" and "mean_model" in ckpt:
            mean_net.load_state_dict(ckpt["mean_model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        # torch.load(..., map_location=device) also moves serialized RNG states
        # to CUDA, while Generator.set_state requires a CPU ByteTensor.
        train_gen.set_state(ckpt["train_gen"].cpu())
        perm_gen.set_state(ckpt["perm_gen"].cpu())
        start, best = ckpt["epoch"] + 1, ckpt["best"]

    for epoch in range(start, args.epochs):
        cov_net.train()
        mean_net.train(args.mean_train != "frozen")
        perm = torch.randperm(train.n_pairs, generator=perm_gen).to(device)
        sums = torch.zeros(6, device=device, dtype=torch.float64)
        count, skipped, tic = 0, 0, time.time()
        for begin in range(0, train.n_pairs, args.batch):
            x, target = train.batch(perm[begin:begin + args.batch])
            metrics = batch_loss(cov_net, mean_net, x, target, obs_std, static_mask,
                                 masks, train_gen, args)
            loss = metrics[0]
            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                skipped += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for group in optimizer.param_groups for parameter in group["params"]], 5.0)
            optimizer.step()
            sums += torch.stack([m.detach().double() for m in metrics]) * len(x)
            count += len(x)

        cov_net.eval()
        mean_net.eval()
        eval_n = min(valid.n_pairs, args.eval_pairs)
        eval_gen = torch.Generator(device=device).manual_seed(args.seed + 10_000)
        val_sums = torch.zeros(6, device=device, dtype=torch.float64)
        with torch.no_grad():
            for begin in range(0, eval_n, args.batch):
                rows = torch.arange(begin, min(begin + args.batch, eval_n), device=device)
                x, target = valid.batch(rows)
                metrics = batch_loss(cov_net, mean_net, x, target, obs_std, static_mask,
                                     masks, eval_gen, args)
                val_sums += torch.stack([m.double() for m in metrics]) * len(x)
        train_metrics = (sums / max(count, 1)).cpu().tolist()
        val_metrics = (val_sums / max(eval_n, 1)).cpu().tolist()
        scheduler.step(val_metrics[0])
        record = {
            "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
            "mean_lr": (optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else 0.0),
            "skipped": skipped,
            "seconds": round(time.time() - tic, 1),
            "train": dict(zip(("loss", "analysis_mse_norm", "forecast_mse_norm", "forecast_nll",
                                "blind_mse_norm", "observed_mse_norm"), train_metrics)),
            "valid": dict(zip(("loss", "analysis_mse_norm", "forecast_mse_norm", "forecast_nll",
                                "blind_mse_norm", "observed_mse_norm"), val_metrics)),
        }
        print(json.dumps(record), flush=True)
        with open(os.path.join(out, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")
        payload = {"model": cov_net.state_dict(), "mean_model": mean_net.state_dict(),
                   "optimizer": optimizer.state_dict(),
                   "scheduler": scheduler.state_dict(), "train_gen": train_gen.get_state(),
                   "perm_gen": perm_gen.get_state(), "epoch": epoch, "best": best,
                   "args": vars(args), "mean_ckpt": args.mean_ckpt}
        if val_metrics[0] < best:
            best = val_metrics[0]
            payload["best"] = best
            torch.save(payload, best_path)
        payload["best"] = best
        torch.save(payload, last_path)
    print(f"[done] best validation loss={best:.6g} -> {best_path}")


if __name__ == "__main__":
    main()
