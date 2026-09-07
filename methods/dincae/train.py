"""
train.py — train DINCAE (ATC crowd field, strictly following the paper)
================================================

Optimisation settings follow 2.0 sec.4: Adam (beta1=0.9, beta2=0.999, eps=1e-8),
**gradient value-clipped to 5** (`ClipGrad(5.0)`, element-wise clipping, not norm
clipping), learning rate `lr = lr0 * 0.5^(epoch/decay_epoch)` (decay_epoch=0 means
no decay), L2 beta=1e-4, nearest upsampling. Their specific numbers were tuned by
Bayesian optimization on their dev set; we copy the structural choices and tune
the numbers on our own dev set later.

Later on, a checkpoint is saved every `--save-every` epochs, for **output
averaging** (1.0 Fig.3: averaging the outputs of epochs 200-1000 beats any single
epoch; the reference implementation's `save_epochs = 200:10:epochs` is this
default behaviour). Averaging happens in `evaluate.py`.

**No** `truth_uncertain`, no age-based shrinkage, no multi-scale temporal
aggregation -- none of these are in the paper.

Usage (**GPU node**):
    python3 state.py                          # compute per-cell statistics first (once, the login node is fine)
    sbatch sbatch/submit_train.sbatch
    # smoke test:
    srun -p gpu-debug --gres=gpu:1 -t 00:15:00 bash -c \
      'module load scicomp-pytorch-env/2026.1; python3 -u train.py --days 2 --epochs 4 --max-frames 4000'
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# append rather than insert(0): 4dvarnet_enkf also has a losses.py, inserting at
# the front would shadow this directory's same-named module
from crowdcore import observation_model as om                                    # noqa: E402

from methods.dincae.state import CHANNELS, StateStats, NCH                # noqa: E402
from methods.dincae.dataset import ChunkedDays, obs_config                        # noqa: E402
from methods.dincae.encoding import N_IN                                          # noqa: E402
from methods.dincae.losses import residual_mse, dincae_loss                        # noqa: E402
from methods.dincae.model import DINCAE                                           # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-decay-epoch", type=float, default=0.0, help="0 = no decay")
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--clip-grad", type=float, default=5.0, help="element-wise clipping (2.0 sec.4)")
    p.add_argument("--enc", type=int, nargs="+", default=[32, 64, 96])
    p.add_argument("--loss-weights", type=float, nargs="+", default=[0.3, 0.7],
                   help="refinement-step weights (2.0 Eq.4). A single value = no refinement")
    p.add_argument("--pool", choices=["mean", "max"], default="mean",
                   help="the reference code hardcodes mean; 2.0's text says max (paper and code disagree)")
    p.add_argument("--stride", type=int, default=4, help="frame-subsampling stride (adjacent frames are highly redundant)")
    p.add_argument("--days", type=int, default=0, help="use only the first N training days (debug)")
    p.add_argument("--max-frames", type=int, default=0, help="use only the first N frames per day (debug)")
    p.add_argument("--days-per-chunk", type=int, default=4)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--save-from", type=int, default=0, help="which epoch to start saving from (default epochs/5)")
    p.add_argument("--out", default=os.path.join(HERE, "runs", "dincae"))
    p.add_argument("--cache-dir", default=os.path.join(HERE, "cache"),
                   help="encoding cache directory (deterministic encoding, built once). Empty string = no cache")
    p.add_argument("--amp", action="store_true", help="bf16 autocast")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--full-field-loss", action="store_true",
                   help="ABLATION: supervise every cell (the target on undefined "
                        "cells is physical 0), instead of the information form's "
                        "valid cells. This trains DINCAE under the same "
                        "convention as the other three methods, used to separate "
                        "'the contribution of the architecture' from 'the "
                        "contribution of the information form'. The cache key "
                        "carries _ff1, so it never touches the existing 13 GB cache.")
    return p.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("!! no GPU -- do not train torch on the login node (see README)", flush=True)

    stats = StateStats()
    train_files = om.split_files("train")
    dev_files = om.split_files("valid")
    if a.days:
        train_files, dev_files = train_files[: a.days], dev_files[:1]
    oc = obs_config()
    print(f"observation config (from 4dvarnet_enkf/config.yaml): num_agents={oc['num_agents']} "
          f"range={oc['sensing_range']} every_k={oc['obs_every_k']} seed={oc['seed']}")
    print(f"train {len(train_files)} days, dev {len(dev_files)} days, "
          f"per-cell statistics from {stats.n_train_days} days, walkable={stats.valid.sum()}")

    model = DINCAE(N_IN, NCH, enc_internal=tuple(a.enc),
                   loss_weights=tuple(a.loss_weights), pool=a.pool).to(dev)
    print(f"n_in={N_IN} n_var={NCH} levels={len(a.loss_weights)} "
          f"params={model.n_params():,}")

    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.9, 0.999),
                           eps=1e-8, weight_decay=a.l2)
    start_epoch = 0
    ckpt_last = os.path.join(a.out, "last.pt")
    if a.resume and os.path.exists(ckpt_last):
        st = torch.load(ckpt_last, map_location=dev)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        start_epoch = st["epoch"] + 1
        print(f"resume from epoch {start_epoch}")

    save_from = a.save_from or max(1, a.epochs // 5)
    amp_dtype = torch.bfloat16 if a.amp else None
    metrics_path = os.path.join(a.out, "metrics.jsonl")
    mk = lambda files, stride, shuffle, seed: ChunkedDays(
        files, stats, batch_size=a.batch, stride=stride,
        days_per_chunk=a.days_per_chunk, max_frames=a.max_frames,
        shuffle=shuffle, seed=seed, cache_dir=a.cache_dir or None,
        full_field=a.full_field_loss)

    for epoch in range(start_epoch, a.epochs):
        lr = a.lr * (0.5 ** (epoch / a.lr_decay_epoch)) if a.lr_decay_epoch else a.lr
        for g in opt.param_groups:
            g["lr"] = lr

        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        lvl_sum = np.zeros(len(a.loss_weights))
        for xb, tb in mk(train_files, a.stride, True, epoch):
            xb, tb = xb.to(dev, non_blocking=True), tb.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            ctx = (torch.autocast("cuda", dtype=amp_dtype) if amp_dtype
                   else torch.enable_grad())
            with ctx:
                outs = model(xb)
                loss, per_level = dincae_loss(outs, tb, a.loss_weights)
            loss.backward()
            if a.clip_grad > 0:
                torch.nn.utils.clip_grad_value_(model.parameters(), a.clip_grad)
            opt.step()
            tot += loss.detach().item(); nb += 1
            lvl_sum += np.array([v.detach().item() for v in per_level])

        # ---- dev set: NLL + residual MSE (MSE only to match scale with 4dvarnet_enkf) ----
        model.eval()
        dv, dn = 0.0, 0
        mse_sum = np.zeros(NCH)
        with torch.no_grad():
            # a coarser stride on dev: it has to be evaluated every epoch, so
            # subsampling 4x saves time and also makes the cache 1/4 the size
            for xb, tb in mk(dev_files, a.stride * 4, False, 0):
                xb, tb = xb.to(dev), tb.to(dev)
                outs = model(xb)
                l, _ = dincae_loss(outs, tb, a.loss_weights)
                dv += l.item(); dn += 1
                mse_sum += residual_mse(outs, tb).cpu().numpy()

        # the network works in normalised residual space; multiplying back by
        # std^2 gives physical units, matching scale with 4dvarnet_enkf
        mse_norm = mse_sum / max(dn, 1)
        mse_phys = mse_norm * (stats.std.astype(np.float64) ** 2)
        rec = {"epoch": epoch, "lr": lr, "train_nll": tot / max(nb, 1),
               "train_nll_per_level": (lvl_sum / max(nb, 1)).tolist(),
               "dev_nll": dv / max(dn, 1),
               "dev_resid_mse": dict(zip(CHANNELS, mse_phys.tolist())),
               "dev_resid_mse_norm": dict(zip(CHANNELS, mse_norm.tolist())),
               "batches": nb, "sec": round(time.time() - t0, 1)}
        print(json.dumps(rec), flush=True)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch, "args": vars(a)}, ckpt_last)
        if epoch >= save_from and (epoch - save_from) % a.save_every == 0:
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(a)},
                       os.path.join(a.out, f"ckpt_{epoch:05d}.pt"))

    print("done")


if __name__ == "__main__":
    main()
