"""
train.py — 训练 DINCAE（ATC 人群场，严格照论文）
================================================

优化设定照 2.0 §4：Adam（β₁=0.9, β₂=0.999, ε=1e-8）、**梯度按值裁到 5**
（`ClipGrad(5.0)`，逐元素裁值，不是裁范数）、学习率 `lr = lr₀·0.5^(epoch/decay_epoch)`
（decay_epoch=0 即不衰减）、L2 β=1e-4、nearest 上采样。他们那些具体数值是 Bayesian
optimization 在 dev 集上调出来的；我们照抄结构性选择，数值以后在自己的 dev 集上调。

后段每 `--save-every` 个 epoch 存一次 checkpoint，供**输出平均**用（1.0 Fig.3：
epoch 200-1000 的输出平均比任何单个 epoch 都好；参考实现 `save_epochs = 200:10:epochs`
就是默认行为）。平均在 `evaluate.py` 里做。

**没有** `truth_uncertain`、没有 age 收缩、没有多尺度时间聚合 —— 这些都不是论文里的东西。

用法（**GPU 节点**）:
    python3 state.py                          # 先算逐格统计量（一次，登录节点即可）
    sbatch sbatch/submit_train.sbatch
    # 冒烟测试:
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

# append 而非 insert(0)：4dvarnet_enkf 里也有 losses.py，插到最前会把本目录的同名模块顶掉
sys.path.append("/scratch/work/zhangx29/Thesis_Project/4dvarnet_enkf")
import observation_model as om                                    # noqa: E402

from state import CHANNELS, StateStats, NCH                # noqa: E402
from dataset import ChunkedDays, obs_config                        # noqa: E402
from encoding import N_IN                                          # noqa: E402
from losses import residual_mse, dincae_loss                        # noqa: E402
from model import DINCAE                                           # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-decay-epoch", type=float, default=0.0, help="0 = 不衰减")
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--clip-grad", type=float, default=5.0, help="逐元素裁值(2.0 §4)")
    p.add_argument("--enc", type=int, nargs="+", default=[32, 64, 96])
    p.add_argument("--loss-weights", type=float, nargs="+", default=[0.3, 0.7],
                   help="精化步权重(2.0 Eq.4)。给单个值 = 不做精化")
    p.add_argument("--pool", choices=["mean", "max"], default="mean",
                   help="参考代码硬编码 mean；2.0 正文写的是 max（论文与代码不一致）")
    p.add_argument("--stride", type=int, default=4, help="抽帧步长(相邻帧高度冗余)")
    p.add_argument("--days", type=int, default=0, help="只用前 N 个训练日(调试)")
    p.add_argument("--max-frames", type=int, default=0, help="每天只用前 N 帧(调试)")
    p.add_argument("--days-per-chunk", type=int, default=4)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--save-from", type=int, default=0, help="从第几个 epoch 开始存(默认 epochs/5)")
    p.add_argument("--out", default=os.path.join(HERE, "runs", "dincae"))
    p.add_argument("--cache-dir", default=os.path.join(HERE, "cache"),
                   help="编码缓存目录（确定性编码，只建一次）。空串 = 不缓存")
    p.add_argument("--amp", action="store_true", help="bf16 autocast")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("!! 没有 GPU —— torch 训练不要在登录节点跑（见 README）", flush=True)

    stats = StateStats()
    train_files = om.split_files("train")
    dev_files = om.split_files("valid")
    if a.days:
        train_files, dev_files = train_files[: a.days], dev_files[:1]
    oc = obs_config()
    print(f"观测配置(取自 4dvarnet_enkf/config.yaml): num_agents={oc['num_agents']} "
          f"range={oc['sensing_range']} every_k={oc['obs_every_k']} seed={oc['seed']}")
    print(f"train {len(train_files)} 天, dev {len(dev_files)} 天, "
          f"逐格统计量用 {stats.n_train_days} 天, walkable={stats.valid.sum()}")

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
        shuffle=shuffle, seed=seed, cache_dir=a.cache_dir or None)

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

        # ---- dev 集：NLL + 残差 MSE（MSE 只为了和 4dvarnet_enkf 对量级）----
        model.eval()
        dv, dn = 0.0, 0
        mse_sum = np.zeros(NCH)
        with torch.no_grad():
            # dev 用更粗的 stride：每 epoch 都要评，抽稀 4 倍既省时间、缓存也只有 1/4 大
            for xb, tb in mk(dev_files, a.stride * 4, False, 0):
                xb, tb = xb.to(dev), tb.to(dev)
                outs = model(xb)
                l, _ = dincae_loss(outs, tb, a.loss_weights)
                dv += l.item(); dn += 1
                mse_sum += residual_mse(outs, tb).cpu().numpy()

        # 网络工作在归一化残差空间；乘回 std² 才是物理单位，才能和 4dvarnet_enkf 对量级
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
