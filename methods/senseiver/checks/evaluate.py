"""
evaluate.py — evaluate Senseiver on held-out days, convention exactly matching 4DVarNet / EnKF
=======================================================================

The metric definitions are **copied verbatim** from
`4dvarnet_enkf/checks/eval_test_days.py`, not a single thing changed:

  * Blind MSE -- squared error on (channel, cell) pairs not observed (`mask < 0.5`)
  * Full-field MSE -- squared error over the entire 4x36x12 (the paper's R-score)
  * Raw field, four channels unweighted, **including non-walkable cells**
  * Same physical clipping as the EnKF: density[0,5], vx/vy[-5,5], var[0,2] (on by default)
  * Computed per day, then mean/std over the 7 days (RMSE is computed per day first, then averaged)
  * Observations are freshly regenerated from the same config.yaml, with
    `obs_every_k` read from the checkpoint's args, ensuring the model is never
    scored against an observation pattern it never saw during training

The timing convention is also copied as-is: only the model's forward pass is
timed, not data preparation (data preparation is a constant shared by all three methods).

Output `check_outputs/eval/senseiver_metrics<tag>.json`, with field names aligned
to `4dvarnet_enkf/check_outputs/eval/test_metrics_*.json`, so it can go straight
into the comparison figures.

Usage (GPU node):
    sbatch sbatch/submit_eval.sbatch
    srun -p gpu-debug --gres=gpu:1 -t 00:14:00 bash -c \
      'module load scicomp-pytorch-env/2026.1; python3 -u checks/evaluate.py --days 1 --frames 400'
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from methods.senseiver import dataset as ds
from methods.senseiver import sensors
from methods.senseiver.network import Senseiver


def clip_bounds(x):
    """Same physical bounds as the EnKF's _clip_bounds (4dvarnet_enkf/checks/eval_test_days.py:44)."""
    x = x.clone()
    x[:, 0].clamp_(0, 5)          # density
    x[:, 1].clamp_(-5, 5)         # vx
    x[:, 2].clamp_(-5, 5)         # vy
    x[:, 3].clamp_(0, 2)          # var
    return x


def load_model(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    m = Senseiver(**ck["hparams"]).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck


@torch.no_grad()
def eval_day(model, day, dev, batch, frames, clip, obs_every_k, chans, ablate=False):
    C, H, W = ds.state_shape()
    X, Y, Om = ds.load_day(day, stride=1, seed=0, frames=frames, obs_every_k=obs_every_k)
    pe = model.pos_enc.detach().cpu().numpy()
    mean = model.in_mean.detach().cpu().numpy()
    std = model.in_std.detach().cpu().numpy()

    se_b = n_b = se_f = n_f = 0.0
    se_c = np.zeros(C); n_c = np.zeros(C)
    solve_s = 0.0
    for i in range(0, X.shape[0], batch):
        sl = slice(i, i + batch)
        tok, pad, _ = sensors.build_batch(Y[sl], Om[sl], pe, mean, std)
        if ablate:
            # Ablation: erase the sensors' "readings", keep only their
            # "positions" and pad_mask. If blind MSE barely changes, the model
            # isn't using the observations, it's just memorised an average field.
            tok[:, :, :C] = 0.0
        tok, pad = tok.to(dev), pad.to(dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()          # CUDA is async: without this we'd only time the kernel launch
        t0 = time.perf_counter()
        xr = model.reconstruct(tok, pad)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        solve_s += time.perf_counter() - t0
        if clip:
            xr = clip_bounds(xr)
        xb = torch.from_numpy(X[sl]).reshape(-1, C, H, W).to(dev)
        # Omega_c = Omega broadcast across the 4 channels (config's default
        # obs_channels=None means all four channels are observed together),
        # matching eval_test_days.py's `mb < 0.5` and score_enkf.py's
        # ~Omega.repeat(4)
        obs = torch.from_numpy(Om[sl]).reshape(-1, 1, H, W).to(dev).expand(-1, C, -1, -1)
        d2 = (xr - xb) ** 2
        se_b += float(d2[~obs].sum()); n_b += int((~obs).sum())
        se_f += float(d2.sum()); n_f += d2.numel()
        for c in range(C):
            m = ~obs[:, c]
            se_c[c] += float(d2[:, c][m].sum()); n_c[c] += int(m.sum())
    per_ch = {chans[c]: se_c[c] / max(n_c[c], 1) for c in range(C)}
    return se_b / n_b, se_f / n_f, X.shape[0], solve_s, per_ch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/senseiver_A/best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--frames", type=int, default=0, help="when >0, only take the first N frames per day")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-clip", dest="clip", action="store_false")
    ap.set_defaults(clip=True)
    ap.add_argument("--outdir", default="check_outputs/eval")
    ap.add_argument("--ablate-values", action="store_true",
                    help="ablation experiment: erases sensor readings (keeps positions), to check whether the model actually uses the observations")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_model(args.ckpt, dev)
    k = ck["args"].get("obs_every_k") or ds.obs_config()["obs_every_k"]
    chans = ds.channels()
    print(f"[model] {args.ckpt}  epoch={ck.get('epoch')}  params={model.num_params:,}  "
          f"obs_every_k={k}  clip={args.clip}  device={dev}", flush=True)

    days = ds.om.split_files(args.split)
    if args.days:
        days = days[:args.days]
    print(f"[data] {len(days)} held-out {args.split} days\n", flush=True)
    print(f"{'day':16s} {'blindMSE':>10} {'blindRMSE':>11} {'fullMSE':>10} {'frames':>8} {'ms/frame':>10}",
          flush=True)
    print("-" * 72, flush=True)

    per_day = []
    for d in days:
        stem = os.path.basename(d).split("_")[0]
        bl, fu, nf, sec, pc = eval_day(model, d, dev, args.batch, args.frames, args.clip, k,
                                       chans, args.ablate_values)
        print(f"{stem:16s} {bl:>10.4f} {bl**0.5:>11.4f} {fu:>10.4f} {nf:>8} "
              f"{sec/nf*1000:>10.3f}", flush=True)
        per_day.append({"day": stem, "blind_mse": bl, "full_mse": fu, "n_frames": nf,
                        "solve_s": sec, "per_frame_ms": sec / nf * 1000,
                        "blind_mse_per_channel": pc})

    bl = np.array([p["blind_mse"] for p in per_day])
    fu = np.array([p["full_mse"] for p in per_day])
    rm = np.sqrt(bl)                       # square-root per day, then average (mean of the sqrt != sqrt of the mean)
    tot_s = sum(p["solve_s"] for p in per_day); tot_f = sum(p["n_frames"] for p in per_day)
    print("-" * 72, flush=True)
    print(f"{'MEAN':16s} {bl.mean():>10.4f} {rm.mean():>11.4f} {fu.mean():>10.4f} "
          f"{'':>8} {tot_s/tot_f*1000:>10.3f}", flush=True)
    print(f"{'STD':16s} {bl.std():>10.4f} {rm.std():>11.4f} {fu.std():>10.4f}", flush=True)
    pc_mean = {c: float(np.mean([p["blind_mse_per_channel"][c] for p in per_day])) for c in chans}
    print(f"[per-channel blind MSE] " + "  ".join(f"{c}={v:.4f}" for c, v in pc_mean.items()), flush=True)

    summary = {"method": "Senseiver (per-frame, faithful port)"
                         + (" [ABLATION: sensor values zeroed]" if args.ablate_values else ""),
               "ablate_values": args.ablate_values, "ckpt": args.ckpt,
               "epoch": ck.get("epoch"), "split": args.split, "obs_every_k": k,
               "clip": args.clip, "frames_per_day": args.frames,
               "n_params": model.num_params, "device": str(dev),
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "node": os.environ.get("SLURMD_NODENAME", ""),
               "solve_total_s": float(tot_s), "n_frames": int(tot_f),
               "per_frame_ms": float(tot_s / tot_f * 1000),
               "blind_mse_mean": float(bl.mean()), "blind_mse_std": float(bl.std()),
               "blind_rmse_mean": float(rm.mean()), "blind_rmse_std": float(rm.std()),
               "full_mse_mean": float(fu.mean()), "full_mse_std": float(fu.std()),
               "blind_mse_per_channel": pc_mean, "per_day": per_day}
    outp = os.path.join(args.outdir, f"senseiver_metrics{args.tag}.json")
    with open(outp, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[out] {outp}", flush=True)


if __name__ == "__main__":
    main()
