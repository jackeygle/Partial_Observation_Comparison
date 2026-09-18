"""
train.py — the PedPred3 surrogate as a deep ensemble: a mean network and a sigma network
===========================================================================================

Two arms, five seeds each, PedPred3's architecture unchanged:

  --arm mean   the forecast mu(x), trained with the ORIGINAL loss
               ('mean total weighted NLLL', enkf_lab/pedpred/metrics.py) and the original
               recipe: Adam(lr 1e-3, amsgrad), ReduceLROnPlateau(factor 0.5, patience 10) on the
               validation loss, batch 50, one input frame -> one output frame by default,
               15 epochs,
               non-finite iterations skipped. (The original train.py instantiates PedPred2 while
               the vendored checkpoint is a PedPred3, so the exact recipe behind that checkpoint
               cannot be recovered; checks/eval_surrogate.py compares the two forecasts.)
  --arm sigma  a second PedPred3 whose four raw output channels are read as log sigma^2 of the
               four state channels (density, vx, vy, var), trained with a Gaussian NLL against
               the FROZEN mean network of the same seed (its best.pt):
                   0.5 * (log sigma^2 + (x_{t+1} - mu(x_t))^2 / sigma^2)
               mu is the mean the EnKF propagates (model.surrogate_mean: vx/vy/var zeroed where
               the forecast density < 0.01, clipped to the EnKF's bounds), computed under
               no_grad, so this arm cannot change the forecast.

Pair s = (mean_s, sigma_s). In the EnKF each member samples a pair per step and forecasts
mu + sigma * eps, in place of the injected 0.01 x Q noise.

Data: grid_cache, train split for fitting, valid split for the scheduler and the checkpoint.
last.pt, best.pt (lowest validation loss -- the checkpoint used) and epoch_XX.pt for every
epoch are kept. best.pt holds {'model': state_dict}, so a mean-arm checkpoint loads
with enkf_lab's own load_model().

    source sbatch/_env.sh && cd methods/enkf
    python3 -u -m methods.enkf.surrogate.train --arm mean  --seed 0
    python3 -u -m methods.enkf.surrogate.train --arm sigma --seed 0     # needs surrogate_mean_s0
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from methods.enkf.surrogate.data import PairFrames
from methods.enkf.surrogate.model import (CH, check_vendor_forward, load_pedpred3, mean_forecast,
                                          original_loss, raw_forecast, surrogate_mean)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # methods/enkf
#: Clamp on log sigma^2 before exp: keeps an untrained head from overflowing in the first steps.
LOGVAR_MIN, LOGVAR_MAX = -20.0, 10.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["mean", "sigma"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=15, help="the original train.py's max_epochs")
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--input-frames", type=int, default=1)
    p.add_argument("--output-frames", type=int, default=1)
    p.add_argument("--mean-ckpt", default="",
                   help="--arm sigma: the frozen mean network (default runs/surrogate_mean_s<seed>/best.pt)")
    p.add_argument("--days", type=int, default=0, help="debug: only the first N days of each split")
    p.add_argument("--max-frames", type=int, default=0, help="debug: only the first N frames per day")
    p.add_argument("--eval-batch", type=int, default=2000)
    p.add_argument("--out", default="", help="default runs/surrogate_<arm>_s<seed>")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def sigma_nll(logvar, mu, y):
    logvar = logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)
    return 0.5 * (logvar + (y - mu) ** 2 * torch.exp(-logvar))


def main():
    a = parse_args()
    if a.input_frames < 1 or a.output_frames < 1:
        raise SystemExit("--input-frames and --output-frames must be positive")
    if not torch.cuda.is_available() and not a.allow_cpu:
        raise SystemExit("no GPU -- train on a GPU node (sbatch/submit_surrogate.sbatch); "
                         "add --allow-cpu to force CPU")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = a.out or os.path.join(HERE, "runs", f"surrogate_{a.arm}_s{a.seed}")
    os.makedirs(out, exist_ok=True)

    t0 = time.time()
    train, valid = (PairFrames(s, dev, a.days, a.max_frames,
                               input_frames=a.input_frames,
                               output_frames=a.output_frames)
                    for s in ("train", "valid"))
    print(f"[data] train {train.n_days} days / {train.n_pairs:,} pairs, valid {valid.n_days} days / "
          f"{valid.n_pairs:,} pairs, {time.time() - t0:.0f}s  dev={dev} "
          f"frames={a.input_frames}->{a.output_frames}", flush=True)

    model = load_pedpred3(device=dev)
    mean_net = None
    if a.arm == "sigma":
        mp = a.mean_ckpt or os.path.join(HERE, "runs", f"surrogate_mean_s{a.seed}", "best.pt")
        mean_net = load_pedpred3(mp, dev).eval()
        for q in mean_net.parameters():
            q.requires_grad_(False)
        print(f"[sigma] frozen mean network: {mp}", flush=True)
    check_vendor_forward(mean_net if mean_net is not None else model,
                         valid.batch(torch.arange(min(8, valid.n_pairs), device=dev))[0],
                         horizon=a.output_frames)
    print("[check] mean_forecast == PedPred3.forward (bit-identical)", flush=True)

    def losses(x, y):
        """Per-batch loss (scalar) and, for the metrics, the forecast mean and log sigma^2."""
        if a.arm == "mean":
            mu = mean_forecast(model, x, horizon=a.output_frames)
            return original_loss(mu, y), mu, None
        with torch.no_grad():
            mu = surrogate_mean(mean_net, x, horizon=a.output_frames)
        logvar = raw_forecast(model, x, horizon=a.output_frames)
        return sigma_nll(logvar, mu, y).mean(), mu, logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)

    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.9, 0.999), amsgrad=True)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10, cooldown=0, threshold=0)
    gen = torch.Generator().manual_seed(a.seed)
    last_p, best_p = os.path.join(out, "last.pt"), os.path.join(out, "best.pt")
    start, best = 0, float("inf")
    if a.resume and os.path.exists(last_p):
        ck = torch.load(last_p, map_location=dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        gen.set_state(ck["gen"].cpu())
        start, best = ck["epoch"] + 1, ck["best"]
        print(f"[resume] from epoch {start}", flush=True)

    for epoch in range(start, a.epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(train.n_pairs, generator=gen).to(dev)
        tot = torch.zeros((), device=dev)
        n, skipped = 0, 0
        for i in range(0, train.n_pairs, a.batch):
            x, y = train.batch(perm[i:i + a.batch])
            loss, _, _ = losses(x, y)
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                skipped += 1
                continue
            loss.backward()
            opt.step()
            tot += loss.detach() * len(x)
            n += len(x)
        train_loss, train_s = float(tot) / max(n, 1), time.time() - t0

        model.eval()
        vtot, vn = 0.0, 0
        se = torch.zeros(4, device=dev, dtype=torch.float64)
        lead_se = torch.zeros(a.output_frames, device=dev, dtype=torch.float64)
        lead_target_sq = torch.zeros(a.output_frames, device=dev, dtype=torch.float64)
        sig = torch.zeros(4, device=dev, dtype=torch.float64)
        with torch.no_grad():
            for i in range(0, valid.n_pairs, a.eval_batch):
                x, y = valid.batch(torch.arange(i, min(i + a.eval_batch, valid.n_pairs), device=dev))
                loss, mu, logvar = losses(x, y)
                vtot += float(loss) * len(x)
                vn += len(x)
                se += ((y - mu) ** 2).double().sum(dim=(0, 1, 3, 4))
                lead_se += ((y - mu) ** 2).double().sum(dim=(0, 2, 3, 4))
                lead_target_sq += y.double().square().sum(dim=(0, 2, 3, 4))
                if logvar is not None:
                    sig += torch.exp(0.5 * logvar).double().sum(dim=(0, 1, 3, 4))
        cells = vn * a.output_frames * valid.frames.shape[-1] * valid.frames.shape[-2]
        rmse = torch.sqrt(se / cells)
        lead_values = vn * 4 * valid.frames.shape[-1] * valid.frames.shape[-2]
        lead_rmse = torch.sqrt(lead_se / lead_values)
        lead_nmse = lead_se / lead_target_sq.clamp_min(torch.finfo(torch.float64).tiny)
        valid_loss = vtot / vn
        sched.step(valid_loss)
        rec = {"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss,
               "lr": opt.param_groups[0]["lr"], "skipped": skipped, "train_s": round(train_s, 1),
               "valid_rmse": {c: float(v) for c, v in zip(CH, rmse)}}
        rec["valid_lead_rmse"] = [float(v) for v in lead_rmse]
        rec["valid_lead_normalized_mse"] = [float(v) for v in lead_nmse]
        if a.arm == "sigma":
            rec["valid_spread_skill"] = {c: float(s / cells / r) for c, s, r in zip(CH, sig, rmse)}
        print(json.dumps(rec), flush=True)
        with open(os.path.join(out, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")

        # Every epoch is kept (3.5 MB each), for inspecting how the unconstrained empty-cell
        # velocity/variance drifts from epoch to epoch (see model.EMPTY_DENSITY).
        torch.save({"model": model.state_dict(), "epoch": epoch, "valid_loss": valid_loss,
                    "args": vars(a), "init": "random"},
                   os.path.join(out, f"epoch_{epoch:02d}.pt"))
        if valid_loss < best:
            best = valid_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "valid_loss": best,
                        "args": vars(a), "init": "random"}, best_p)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "gen": gen.get_state(), "epoch": epoch, "best": best,
                    "args": vars(a), "init": "random"}, last_p)
    print(f"[done] best valid loss {best:.6g} -> {best_p}", flush=True)


if __name__ == "__main__":
    main()
