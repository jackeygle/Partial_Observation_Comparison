"""
train.py — the PedPred3 mean forecast network
=============================================

PedPred3's architecture unchanged, trained with the ORIGINAL loss
('mean total weighted NLLL', enkf_lab/pedpred/metrics.py) and the original
recipe: Adam(lr 1e-3, amsgrad), ReduceLROnPlateau(factor 0.5, patience 10) on the
validation loss, batch 50, one input frame -> one output frame by default,
15 epochs, non-finite iterations skipped. (The original train.py instantiates
PedPred2 while the vendored checkpoint is a PedPred3, so the exact recipe behind
that checkpoint cannot be recovered; checks/eval_surrogate.py compares the two
forecasts.)

This is the frozen mean the learned covariance sits on top of -- see
lowrank/train.py, which takes it through --mean-ckpt.

Data: grid_cache, train split for fitting, valid split for the scheduler and the
checkpoint. last.pt, best.pt (lowest validation loss -- the checkpoint used) and
epoch_XX.pt for every epoch are kept. best.pt holds {'model': state_dict}, so it
loads with enkf_lab's own load_model().

    source sbatch/_env.sh && cd methods/enkf
    python3 -u -m methods.enkf.lcskf.dynamics.train --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from methods.enkf.lcskf.dynamics.data import PairFrames
from methods.enkf.lcskf.dynamics.model import (CH, check_vendor_forward, load_pedpred3, mean_forecast,
                                          original_loss, raw_forecast, surrogate_mean)

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # methods/enkf
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=15, help="the original train.py's max_epochs")
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--input-frames", type=int, default=1)
    p.add_argument("--output-frames", type=int, default=1)
    p.add_argument("--mean-ckpt", default="",
                   help="resume the mean network from this checkpoint instead of the vendored one")
    p.add_argument("--days", type=int, default=0, help="debug: only the first N days of each split")
    p.add_argument("--max-frames", type=int, default=0, help="debug: only the first N frames per day")
    p.add_argument("--eval-batch", type=int, default=2000)
    p.add_argument("--out", default="", help="default runs/surrogate_<arm>_s<seed>")
    p.add_argument("--clip-grad", type=float, default=0.0,
                   help="clip the gradient norm to this value; 0 keeps the original "
                        "recipe, which has none. The read-outs are exp(ch0) and exp(ch3), "
                        "so one large step can push the variance channel past what float32 "
                        "survives: a run diverged at epoch 30 with the variance reaching "
                        "7e22, after which every batch was non-finite and skipped.")
    p.add_argument("--init-weights", default="",
                   help="start from this checkpoint's weights with a fresh optimiser, "
                        "unlike --resume which also restores the optimiser and epoch "
                        "counter. For restarting after a divergence from the last good "
                        "epoch, where the optimiser state is what has to be discarded.")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


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
    out = a.out or os.path.join(HERE, "runs", f"surrogate_mean_s{a.seed}")
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
    check_vendor_forward(model,
                         valid.batch(torch.arange(min(8, valid.n_pairs), device=dev))[0],
                         horizon=a.output_frames)
    print("[check] mean_forecast == PedPred3.forward (bit-identical)", flush=True)

    def losses(x, y):
        """Per-batch loss (scalar) and, for the metrics, the forecast mean."""
        mu = mean_forecast(model, x, horizon=a.output_frames)
        return original_loss(mu, y), mu, None

    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.9, 0.999), amsgrad=True)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10, cooldown=0, threshold=0)
    gen = torch.Generator().manual_seed(a.seed)
    last_p, best_p = os.path.join(out, "last.pt"), os.path.join(out, "best.pt")
    start, best = 0, float("inf")
    if a.init_weights:
        model.load_state_dict(torch.load(a.init_weights, map_location=dev)["model"])
        print(f"[init] weights from {a.init_weights}, optimiser fresh", flush=True)
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
            if a.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip_grad)
            opt.step()
            tot += loss.detach() * len(x)
            n += len(x)
        train_loss, train_s = float(tot) / max(n, 1), time.time() - t0

        model.eval()
        vtot, vn = 0.0, 0
        se = torch.zeros(4, device=dev, dtype=torch.float64)
        lead_se = torch.zeros(a.output_frames, device=dev, dtype=torch.float64)
        lead_target_sq = torch.zeros(a.output_frames, device=dev, dtype=torch.float64)
        with torch.no_grad():
            for i in range(0, valid.n_pairs, a.eval_batch):
                x, y = valid.batch(torch.arange(i, min(i + a.eval_batch, valid.n_pairs), device=dev))
                loss, mu, _ = losses(x, y)
                vtot += float(loss) * len(x)
                vn += len(x)
                se += ((y - mu) ** 2).double().sum(dim=(0, 1, 3, 4))
                lead_se += ((y - mu) ** 2).double().sum(dim=(0, 2, 3, 4))
                lead_target_sq += y.double().square().sum(dim=(0, 2, 3, 4))
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
