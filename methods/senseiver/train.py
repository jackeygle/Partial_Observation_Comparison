"""
train.py — Senseiver 在 ATC 上的训练
=====================================

对应参考实现的 `train.py` + `network_light.py::training_step`，但用纯 PyTorch 循环
（理由见 `network.py` 的文件头）。训练目标与参考实现逐字一致：原始场上的四通道
无权重 MSE（`losses.senseiver_loss`）。

超参的默认值取自参考实现的 `s_parser.py`，两处按本问题的规模调整并在此说明：

  * `--space-bands 16`（参考默认 32）。频率是 `linspace(1, dim/2, bands)`，
    W=12 时最高频只有 6，32 个 band 纯属冗余。
  * `--lr 1e-3`（参考的 argparse 默认是 1e-4）。README 里帧数上万的那个例子
    （pipe）用的就是 1e-3，我们是 128 万帧的量级，同一档。

用法（**GPU 节点**）：
    sbatch sbatch/submit_train.sbatch
    # 冒烟
    srun -p gpu-debug --gres=gpu:1 -t 00:15:00 bash -c \
      'module load scicomp-pytorch-env/2026.1; python3 -u train.py --days 1 --steps 50'
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from methods.senseiver import dataset as ds
from methods.senseiver import sensors
from methods.senseiver.losses import diagnostics, senseiver_loss
from methods.senseiver.network import Senseiver


def make_batch(bank, idx, pos_enc_np, mean, std, dev):
    tok, pad, n = sensors.build_batch(bank.Y[idx], bank.Omega[idx], pos_enc_np, mean, std)
    C, H, W = ds.state_shape()
    x = torch.from_numpy(bank.X[idx]).reshape(len(idx), C, H, W)
    om = torch.from_numpy(bank.Omega[idx]).reshape(len(idx), H, W)
    return tok.to(dev), pad.to(dev), x.to(dev), om.to(dev), n


@torch.no_grad()
def validate(model, bank, pos_enc_np, mean, std, dev, batch, chans, max_batches=40):
    model.eval()
    agg, nb = {}, 0
    rng = np.random.default_rng(0)
    for i, idx in enumerate(bank.batches(batch, rng)):
        if i >= max_batches:
            break
        tok, pad, x, om, _ = make_batch(bank, idx, pos_enc_np, mean, std, dev)
        pred = model.reconstruct(tok, pad)
        d = diagnostics(pred.float(), x, om, chans)
        for k, v in d.items():
            agg[k] = agg.get(k, 0.0) + v
        nb += 1
    model.train()
    return {k: v / max(nb, 1) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser(description="Senseiver on ATC")
    # 数据
    ap.add_argument("--split", default="train")
    ap.add_argument("--days", type=int, default=0, help="最多用多少天 (0 = 全部)")
    ap.add_argument("--stride", type=int, default=4, help="抽帧步长(相邻秒高度冗余)")
    ap.add_argument("--valid-days", type=int, default=3)
    ap.add_argument("--valid-stride", type=int, default=20)
    ap.add_argument("--frames", type=int, default=0,
                    help=">0 时每天只取前 N 帧(冒烟用；观测模拟按整天跑很花时间)")
    ap.add_argument("--obs-every-k", type=int, default=None, help="覆盖 config 的 observation.obs_every_k")
    # 模型（默认取自 reference/Senseiver/s_parser.py）
    ap.add_argument("--space-bands", type=int, default=16)
    ap.add_argument("--enc-preproc-ch", type=int, default=32)
    ap.add_argument("--num-latents", type=int, default=64)
    ap.add_argument("--enc-num-latent-channels", type=int, default=32)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--num-cross-attention-heads", type=int, default=2)
    ap.add_argument("--enc-num-self-attention-heads", type=int, default=2)
    ap.add_argument("--num-self-attention-layers-per-block", type=int, default=3)
    ap.add_argument("--dec-preproc-ch", type=int, default=32)
    ap.add_argument("--dec-num-latent-channels", type=int, default=32)
    ap.add_argument("--dec-num-cross-attention-heads", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.0)
    # 训练
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--steps", type=int, default=0, help=">0 时每轮只跑这么多步(调试)")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out", default="runs/senseiver_A")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("没有 GPU。本项目的规矩是 torch 只在 GPU 节点跑；"
                         "确实要用 CPU 就加 --allow-cpu。")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    C, H, W = ds.state_shape()
    chans = ds.channels()
    print(f"[config] 观测配置 = {ds.obs_config()}", flush=True)

    print(f"[data] 载入 {args.split} split ...", flush=True)
    train_bank = ds.DayBank(ds.om.split_files(args.split), stride=args.stride,
                            seed=args.seed, max_days=args.days, frames=args.frames,
                            obs_every_k=args.obs_every_k)
    print(f"[data] 训练样本 {train_bank.n} 帧", flush=True)
    valid_bank = ds.DayBank(ds.om.split_files("valid"), stride=args.valid_stride,
                            seed=args.seed, max_days=args.valid_days, frames=args.frames,
                            obs_every_k=args.obs_every_k)
    print(f"[data] 验证样本 {valid_bank.n} 帧", flush=True)

    mean, std = train_bank.input_stats()
    print(f"[stats] 编码器输入标准化 mean={mean} std={std}", flush=True)

    model = Senseiver(
        im_ch=C, grid=(H, W), space_bands=args.space_bands,
        enc_preproc_ch=args.enc_preproc_ch, num_latents=args.num_latents,
        enc_num_latent_channels=args.enc_num_latent_channels,
        num_layers=args.num_layers,
        num_cross_attention_heads=args.num_cross_attention_heads,
        enc_num_self_attention_heads=args.enc_num_self_attention_heads,
        num_self_attention_layers_per_block=args.num_self_attention_layers_per_block,
        dec_preproc_ch=args.dec_preproc_ch,
        dec_num_latent_channels=args.dec_num_latent_channels,
        dec_num_cross_attention_heads=args.dec_num_cross_attention_heads,
        dropout=args.dropout, in_mean=mean, in_std=std).to(dev)
    print(f"[model] {model.num_params:,} 参数", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and dev.type == "cuda")
    pe_np = model.pos_enc.detach().cpu().numpy()

    start_epoch, best = 0, float("inf")
    last_p = os.path.join(args.out, "last.pt")
    if args.resume and os.path.exists(last_p):
        ck = torch.load(last_p, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"] + 1; best = ck.get("best", best)
        print(f"[resume] 从 epoch {start_epoch} 继续 (best={best:.5f})", flush=True)

    mpath = os.path.join(args.out, "metrics.jsonl")
    rng = np.random.default_rng(args.seed + start_epoch)
    for ep in range(start_epoch, args.epochs):
        t0, tot, nel, nstep = time.perf_counter(), 0.0, 0, 0
        for i, idx in enumerate(train_bank.batches(args.batch, rng)):
            if args.steps and i >= args.steps:
                break
            tok, pad, x, _om, _n = make_batch(train_bank, idx, pe_np, mean, std, dev)
            with torch.amp.autocast("cuda", enabled=args.amp and dev.type == "cuda"):
                pred = model.reconstruct(tok, pad)
                loss = senseiver_loss(pred, x)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += float(loss.detach()); nel += x.numel(); nstep += 1

        v = validate(model, valid_bank, pe_np, mean, std, dev, args.batch, chans)
        rec = {"epoch": ep, "train_mse": tot / max(nel, 1), "steps": nstep,
               "sec": time.perf_counter() - t0, "lr": args.lr,
               **{f"valid_{k}": val for k, val in v.items()}}
        with open(mpath, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[ep {ep:3d}] train {rec['train_mse']:.5f}  valid mse {v['mse']:.5f}  "
              f"blind {v['mse_blind']:.5f}  density_occ {v['mse_density_occ']:.5f}  "
              f"({rec['sec']:.0f}s)", flush=True)

        ck = {"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": ep,
              "hparams": model.hparams, "args": vars(args), "best": best,
              "in_mean": mean.tolist(), "in_std": std.tolist()}
        torch.save(ck, last_p)
        if v["mse_blind"] < best:
            best = v["mse_blind"]; ck["best"] = best
            torch.save(ck, os.path.join(args.out, "best.pt"))
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
