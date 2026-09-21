"""Train a low-rank covariance U-Net through a differentiable Kalman update.

PedPred3 can remain frozen (the original experiment), or its output head/full
network can be jointly fine-tuned through the analysis loss. Training samples
legal robot positions and uses their exact map/line-of-sight footprints;
evaluation on moving-robot masks is done by ``lcskf/filter.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from crowdcore import config, navigation
from methods.enkf.lcskf.kalman import analysis_update
from methods.enkf.lcskf.covariance import CovarianceUNet, STATE_SCALE, load_covariance_state
from methods.enkf.lcskf.dynamics.data import PairFrames
from methods.enkf.lcskf.dynamics.model import load_pedpred3, surrogate_mean

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # methods/enkf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mean-ckpt", default=os.path.join(HERE, "runs/surrogate_mean_s0/best.pt"))
    p.add_argument("--cov-ckpt", default="", help="initialize covariance U-Net from a checkpoint")
    p.add_argument("--mean-train", choices=("frozen", "full"), default="frozen")
    p.add_argument("--mean-lr", type=float, default=1e-5)
    p.add_argument("--input-frames", type=int, default=1,
                   help="causal state-history length consumed by PedPred3")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-agents", type=int, default=3)
    p.add_argument("--sensing-radius", type=float, default=7.0)
    p.add_argument("--days", type=int, default=0, help="debug: first N days per split")
    p.add_argument("--max-frames", type=int, default=0, help="debug: first N frames per day")
    p.add_argument("--init-from", default="",
                   help="resume a finished joint checkpoint: take BOTH the covariance\n"
                        "net and the fine-tuned PedPred3 out of it. Overrides\n"
                        "--cov-ckpt/--mean-ckpt; this is how a clean truth-trained\n"
                        "pipeline gets adapted to a new input regime.")
    p.add_argument("--history-mix", type=float, default=0.0,
                   help="probability that a batch uses the TRUE history instead of\n"
                        "--history's. 0 = the degraded regime only (which trades clean\n"
                        "accuracy for robustness), 0.5 = equal-parts multi-condition\n"
                        "training. Truth batches resample a robot layout, exactly as a\n"
                        "pure --history truth run does.")
    p.add_argument("--consistency", type=float, default=0.0,
                   help="weight on a clean-view consistency term: the forecast made from "
                        "the degraded history is pulled towards the one the same weights "
                        "make from the TRUE history (detached). Targets the robustness/"
                        "clean trade-off that training on the degraded input alone "
                        "produces. Costs one extra no-grad forward.")
    p.add_argument("--cov-obs-mask", action="store_true",
                   help="give the covariance net the robots' 0/1 mask for the frame it is "
                        "handed as x_t. Only meaningful with a non-truth --history, where "
                        "x_t's cells differ in how recently they were measured.")
    p.add_argument("--history-fill", default="", choices=("", "prev", "zeros"),
                   help="how --history observed fills unobserved cells; default follows "
                        "config (prev). 'zeros' is closer to the truth but needs "
                        "--cov-obs-mask to stay interpretable.")
    p.add_argument("--history", default="truth",
                   help="what the nets see as the past: 'truth' (the true state, free but "
                        "not what the running filter ever sees), 'observed' (the robots' "
                        "partial observation, carried forward into a complete grid -- the "
                        "same X0 4DVarNet starts from), or a directory of bg_<day>.npz "
                        "written by checks/export_backgrounds.py. Either non-truth mode "
                        "also takes the Kalman correction's mask from the data.")
    p.add_argument("--ckpt-every", type=int, default=0,
                   help="also keep ckpt_<epoch>.pt every N epochs. best.pt is chosen on "
                        "the first --eval-pairs validation pairs, which are the empty early "
                        "morning, so a run that will be judged elsewhere needs snapshots "
                        "to choose from afterwards.")
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


def batch_loss(cov_net, mean_net, x, target, obs_std, static_mask, masks, generator, args,
               data_mask=None, x_observed=None, x_clean=None):
    if args.mean_train == "frozen":
        with torch.no_grad():
            mean = surrogate_mean(mean_net, x)[:, 0]
    else:
        mean = surrogate_mean(mean_net, x)[:, 0]
    x_grid, target_grid = x[:, -1], target[:, 0]
    factor, diagonal = cov_net(x_grid, mean, static_mask, x_observed)
    # When the history came from the robots, the robots that left those gaps are the
    # ones observing now, exactly as at deployment.  On a true-state history there is
    # no such trajectory, so resample a layout and keep B from specialising to one.
    mask = (data_mask.bool() if data_mask is not None
            else random_sensor_mask(len(x), args.num_agents, masks, generator))
    noise = torch.randn(target_grid.shape, device=target.device, dtype=target.dtype,
                        generator=generator) * obs_std.view(1, -1, 1, 1)
    observation = target_grid + noise
    analysis = analysis_update(mean, factor, diagonal, observation, mask, obs_std.square())
    scale = torch.as_tensor(STATE_SCALE, device=target.device, dtype=target.dtype).view(1, -1, 1, 1)
    sq = ((analysis - target_grid) / scale).square()
    analysis_loss = sq.mean()
    total = analysis_loss
    if x_clean is not None:
        # Teacher: the same weights fed the TRUE past.  Pulling the degraded-input
        # forecast towards it says "give the same answer despite the worse input",
        # which the analysis loss alone does not ask for.  Kept out of
        # ``analysis_loss`` so the logged diagnostic stays the analysis error itself.
        with torch.no_grad():
            clean_mean = surrogate_mean(mean_net, x_clean)[:, 0]
        total = total + args.consistency * (((mean - clean_mean) / scale).square().mean())
    # Reported as a diagnostic only: the objective is the analysis error alone.
    # A forecast-MSE term and a forecast-NLL term were both swept and neither
    # earned its place -- see the README's provenance table.
    forecast_loss = ((mean - target_grid) / scale).square().mean()
    blind = (~mask)[:, None].expand_as(sq)
    observed = mask[:, None].expand_as(sq)
    blind_mse = sq[blind].mean() if blind.any() else sq.new_zeros(())
    observed_mse = sq[observed].mean() if observed.any() else sq.new_zeros(())
    return total, analysis_loss, forecast_loss, blind_mse, observed_mse


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
                               input_frames=args.input_frames,
                               history=args.history, history_fill=args.history_fill)
                    for split in ("train", "valid"))
    print(f"[data] train={train.n_pairs:,}, valid={valid.n_pairs:,}, load={time.time()-t0:.0f}s, "
          f"device={device}", flush=True)

    mean_net = load_pedpred3("" if args.init_from else args.mean_ckpt, device).eval()
    for parameter in mean_net.parameters():
        parameter.requires_grad_(False)
    if args.mean_train == "full":
        for parameter in mean_net.parameters():
            parameter.requires_grad_(True)
    cov_net = CovarianceUNet(rank=args.rank, width=args.width,
                             observation_mask=args.cov_obs_mask).to(device)
    if args.init_from:
        finished = torch.load(args.init_from, map_location=device)
        load_covariance_state(cov_net, finished["model"])
        if "mean_model" not in finished:
            raise SystemExit(f"{args.init_from} carries no mean_model; it is not a joint run")
        mean_net.load_state_dict(finished["mean_model"])
        print(f"[init] covariance and PedPred3 both from {args.init_from}", flush=True)
    elif args.cov_ckpt:
        covariance_checkpoint = torch.load(args.cov_ckpt, map_location=device)
        # A checkpoint without the mask channel still warm-starts: the new channel's
        # weights come in at zero, so the model begins numerically identical to it.
        load_covariance_state(cov_net, covariance_checkpoint["model"])
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
          f"mean_lr={args.mean_lr:g}; "
          f"input_frames={args.input_frames}; cov_init={args.cov_ckpt or 'random'}; "
          f"history={args.history}/{args.history_fill or 'prev'}; "
          f"history_mix={args.history_mix:g}; consistency={args.consistency:g}; "
          f"cov_obs_mask={args.cov_obs_mask}; "
          f"init_from={args.init_from or '-'}", flush=True)
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
        sums = torch.zeros(5, device=device, dtype=torch.float64)
        count, skipped, tic = 0, 0, time.time()
        for begin in range(0, train.n_pairs, args.batch):
            rows = perm[begin:begin + args.batch]
            # Multi-condition training: some batches see the true past, the rest the
            # degraded one. A truth batch has no robot trajectory behind its gaps, so
            # it resamples a layout, exactly as a pure --history truth run does.
            use_truth = (args.history_mix > 0
                         and float(torch.rand((), generator=train_gen, device=device)) < args.history_mix)
            x, target = train.batch(rows, truth_history=use_truth)
            data_mask = None if use_truth else train.observed_at(rows)
            x_observed = (train.observed_at(rows, -1)
                          if (args.cov_obs_mask and not use_truth) else None)
            x_clean = (train.batch(rows, truth_history=True)[0]
                       if (args.consistency > 0 and not use_truth) else None)
            metrics = batch_loss(cov_net, mean_net, x, target, obs_std, static_mask,
                                 masks, train_gen, args, data_mask, x_observed, x_clean)
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
        val_sums = torch.zeros(5, device=device, dtype=torch.float64)
        with torch.no_grad():
            for begin in range(0, eval_n, args.batch):
                rows = torch.arange(begin, min(begin + args.batch, eval_n), device=device)
                x, target = valid.batch(rows)
                metrics = batch_loss(cov_net, mean_net, x, target, obs_std, static_mask,
                                     masks, eval_gen, args, valid.observed_at(rows),
                                     valid.observed_at(rows, -1) if args.cov_obs_mask else None)
                val_sums += torch.stack([m.double() for m in metrics]) * len(x)
        train_metrics = (sums / max(count, 1)).cpu().tolist()
        val_metrics = (val_sums / max(eval_n, 1)).cpu().tolist()
        scheduler.step(val_metrics[0])
        record = {
            "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
            "mean_lr": (optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else 0.0),
            "skipped": skipped,
            "seconds": round(time.time() - tic, 1),
            "train": dict(zip(("loss", "analysis_mse_norm", "forecast_mse_norm",
                                "blind_mse_norm", "observed_mse_norm"), train_metrics)),
            "valid": dict(zip(("loss", "analysis_mse_norm", "forecast_mse_norm",
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
        if args.ckpt_every and (epoch + 1) % args.ckpt_every == 0:
            torch.save(payload, os.path.join(out, f"ckpt_{epoch + 1:03d}.pt"))
    print(f"[done] best validation loss={best:.6g} -> {best_path}")


if __name__ == "__main__":
    main()
