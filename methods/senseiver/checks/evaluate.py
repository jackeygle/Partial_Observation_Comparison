"""
evaluate.py — 在留出日上评估 Senseiver，口径与 4DVarNet / EnKF 完全一致
=======================================================================

指标定义**逐字复刻** `4dvarnet_enkf/checks/eval_test_days.py`，一个字都不改：

  * 盲区 MSE —— 未被观测到的 (通道, 格子) 上的平方误差（`mask < 0.5`）
  * 全场 MSE —— 全部 4x36x12 上的平方误差（论文的 R-score）
  * 原始场、四通道无权重、**包含非 walkable 格**
  * 与 EnKF 相同的物理裁剪：density[0,5]、vx/vy[-5,5]、var[0,2]（默认开）
  * 逐日算，再对 7 天取 mean/std（RMSE 先逐日开方再平均）
  * 观测由同一份 config.yaml 现场重新生成，`obs_every_k` 从 checkpoint 的 args 里读，
    保证模型不会被拿去打分一个它训练时从没见过的观测模式

计时口径也照抄：只计模型前向，不计数据准备（数据准备是三种方法共享的常数）。

输出 `check_outputs/eval/senseiver_metrics<tag>.json`，字段名与
`4dvarnet_enkf/check_outputs/eval/test_metrics_*.json` 对齐，可以直接进对比图。

用法（GPU 节点）:
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
    """与 EnKF 的 _clip_bounds 相同的物理界（4dvarnet_enkf/checks/eval_test_days.py:44）。"""
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
            # 对照：抹掉传感器"读数"，只保留它们的"位置"和 pad_mask。
            # 若 blind MSE 几乎不变，说明模型没在用观测，只是背了一个平均场。
            tok[:, :, :C] = 0.0
        tok, pad = tok.to(dev), pad.to(dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()          # CUDA 异步：不同步就只计到 kernel launch
        t0 = time.perf_counter()
        xr = model.reconstruct(tok, pad)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        solve_s += time.perf_counter() - t0
        if clip:
            xr = clip_bounds(xr)
        xb = torch.from_numpy(X[sl]).reshape(-1, C, H, W).to(dev)
        # Omega_c = Omega 在 4 个通道上广播（config 默认 obs_channels=None，四通道同时观测），
        # 与 eval_test_days.py 的 `mb < 0.5` 和 score_enkf.py 的 ~Omega.repeat(4) 一致
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
    ap.add_argument("--frames", type=int, default=0, help=">0 只取每天前 N 帧")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-clip", dest="clip", action="store_false")
    ap.set_defaults(clip=True)
    ap.add_argument("--outdir", default="check_outputs/eval")
    ap.add_argument("--ablate-values", action="store_true",
                    help="对照实验：抹掉传感器读数(保留位置)，检验模型是否真在用观测")
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
    print(f"[data] {len(days)} 个 {args.split} 日（留出）\n", flush=True)
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
    rm = np.sqrt(bl)                       # 先逐日开方再平均（均值的平方根 != 平方根的均值）
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
