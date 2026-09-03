"""
trace_pipeline.py — 打印一帧数据从原始网格走到损失的每一步形状
================================================================

不是测试，是**文档的可执行版本**：README 的 "Pipeline" 一节里的每个形状都由这个
脚本产生。改了数据管线就重跑它，README 和代码就不会各说各话。

用法（GPU 节点）:
    srun -p gpu-debug --gres=gpu:1 -t 00:12:00 --mem=24G bash -c \
      'module load scicomp-pytorch-env/2026.1; python3 -u checks/trace_pipeline.py'
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

import dataset as ds
import sensors
from network import Senseiver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/senseiver_A/best.pt",
                    help="有 checkpoint 就用它的超参；没有就用默认超参新建一个")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    B = args.batch

    print("=== 1. 原始一天 ===")
    day = ds.om.split_files("test")[0]
    X, _ = ds.om.load_state(day)
    X = np.asarray(X)[:args.frames]
    print(f"  grid_cache 一天: {X.shape} {X.dtype}  (T, C, H, W)")

    print("=== 2. 观测模型 (4dvarnet_enkf) ===")
    valid = ds.nav.build_valid_mask_from_config(X)
    out = ds.om.generate_observations(X, valid_mask=valid, seed=0)
    n_obs = out["Omega"].sum(axis=(1, 2))
    print(f"  Y      : {out['Y'].shape}  带噪部分观测, 未观测处=0")
    print(f"  Omega  : {out['Omega'].shape}  bool, 该秒哪些格被看到 "
          f"-> {n_obs[:5].tolist()} ...  (min {n_obs.min()} max {n_obs.max()})")
    print(f"  walkable: {int(valid.sum())} / {valid.size}")

    print("=== 3. dataset.load_day 展平 ===")
    Xs, Ys, Om = ds.load_day(day, stride=1, seed=0, frames=args.frames)
    print(f"  X : {Xs.shape}  (N, C, HW)   <- 目标")
    print(f"  Y : {Ys.shape}  (N, C, HW)   <- 观测")
    print(f"  Om: {Om.shape}  (N, HW) bool")

    print("=== 4. sensors.build_batch 变长 token ===")
    if os.path.exists(args.ckpt):
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        m = Senseiver(**ck["hparams"])
        m.load_state_dict(ck["model"])
        print(f"  (超参来自 {args.ckpt}, epoch {ck.get('epoch')})")
    else:
        C, H, W = ds.state_shape()
        m = Senseiver(im_ch=C, grid=(H, W), space_bands=16, enc_preproc_ch=32,
                      num_latents=64, enc_num_latent_channels=32, num_layers=3,
                      num_cross_attention_heads=2, enc_num_self_attention_heads=2,
                      num_self_attention_layers_per_block=3, dec_preproc_ch=32,
                      dec_num_latent_channels=32, dec_num_cross_attention_heads=1)
        print("  (没找到 checkpoint, 用默认超参新建)")
    m.eval()
    pe = m.pos_enc.numpy(); mu = m.in_mean.numpy(); sd = m.in_std.numpy()
    idx = np.arange(B)
    tok, pad, n = sensors.build_batch(Ys[idx], Om[idx], pe, mu, sd)
    C = m.im_ch
    print(f"  pos_enc : {pe.shape}  (HW, P)   P = 2 * 2维 * bands")
    print(f"  tokens  : {tuple(tok.shape)}  (B, Nmax, C+P) = ({B}, {tok.shape[1]}, {C}+{pe.shape[1]})")
    print(f"  pad_mask: {tuple(pad.shape)}  True=padding位")
    print(f"  每帧真实传感器数: {n.tolist()}")

    print("=== 5. 模型前向 ===")
    with torch.no_grad():
        z = m.encoder(tok, pad)
        print(f"  encoder ->  z: {tuple(z.shape)}  (B, num_latents, ch)  <-- 与传感器数量无关!")
        coords = m.pos_enc[None].expand(B, -1, -1)
        print(f"  查询坐标 coords: {tuple(coords.shape)}  (B, HW, P)  整张网格")
        o = m.decoder(z, coords)
        print(f"  decoder ->   : {tuple(o.shape)}  (B, HW, C)")
        rec = m.reconstruct(tok, pad)
        print(f"  reshape ->   : {tuple(rec.shape)}  (B, C, H, W)")

    print("=== 6. 损失 / 指标 对齐 ===")
    Cc, H, W = ds.state_shape()
    xb = torch.from_numpy(Xs[idx]).reshape(B, Cc, H, W)
    obs = torch.from_numpy(Om[idx]).reshape(B, 1, H, W).expand(-1, Cc, -1, -1)
    d2 = (rec - xb) ** 2
    nb, nf = int((~obs).sum()), d2.numel()
    print(f"  盲区格数 {nb} / 全部 {nf}  ({nb/nf*100:.1f}%)")
    print(f"  输出通道数由 im_ch 决定 = {C}  (postproc.weight "
          f"{tuple(m.decoder.postproc.weight.shape)}), 与 latent_size="
          f"{m.hparams['latent_size']} 无关")


if __name__ == "__main__":
    main()
