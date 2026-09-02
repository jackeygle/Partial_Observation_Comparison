"""
check_model.py — 移植后的模型自检
==================================

Senseiver 之所以适合我们这个问题，全靠 cross-attention 对**无序、变长**的传感器
集合的置换不变性。这个性质必须被证明，而不是假定——参考实现从没测过它（它的
传感器集定长且顺序固定，测不出来）。

检查项：
  1. **置换不变**：打乱传感器 token 的顺序，输出不变
  2. **padding 不变**：同一组真实 token，padding 到不同长度，输出不变
  3. 空传感器集不产生 NaN
  4. 可微：loss.backward() 后所有参数都有非零梯度通路
  5. 维度断言：dec != enc 的 latent 通道数会被拒绝（参考实现缺这个检查）
  6. 参数量与前向形状

用法: python3 checks/check_model.py     (GPU 节点)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from network import Senseiver

FAIL = []


def ck(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' — ' + extra) if extra else ''}")
    if not cond:
        FAIL.append(name)


def build(**kw):
    d = dict(im_ch=4, grid=(36, 12), space_bands=16, enc_preproc_ch=32, num_latents=64,
             enc_num_latent_channels=32, num_layers=3, num_cross_attention_heads=2,
             enc_num_self_attention_heads=2, num_self_attention_layers_per_block=3,
             dec_preproc_ch=32, dec_num_latent_channels=32,
             dec_num_cross_attention_heads=1, dropout=0.0)
    d.update(kw)
    return Senseiver(**d)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    m = build().to(dev).eval()
    P = m.pos_enc.shape[1]
    print(f"[model] {m.num_params:,} 参数   位置编码 {P} 通道   device={dev}")

    B, Ns = 4, 190
    tok = torch.randn(B, Ns, 4 + P, device=dev)
    pad = torch.zeros(B, Ns, dtype=torch.bool, device=dev)
    with torch.no_grad():
        out = m.reconstruct(tok, pad)
    ck("6. 前向形状 (B,C,H,W)", tuple(out.shape) == (B, 4, 36, 12), str(tuple(out.shape)))

    # 1. 置换不变
    perm = torch.randperm(Ns, device=dev)
    with torch.no_grad():
        out_p = m.reconstruct(tok[:, perm], pad[:, perm])
    d1 = float((out - out_p).abs().max())
    ck("1. 传感器顺序置换不变", d1 < 1e-4, f"max|Δ| = {d1:.2e}")

    # 2. padding 不变：真实 100 个 token，分别 pad 到 100 / 260
    n_real = 100
    t_short = tok[:, :n_real].clone()
    p_short = torch.zeros(B, n_real, dtype=torch.bool, device=dev)
    t_long = torch.zeros(B, 260, 4 + P, device=dev)
    t_long[:, :n_real] = t_short
    p_long = torch.ones(B, 260, dtype=torch.bool, device=dev)
    p_long[:, :n_real] = False
    with torch.no_grad():
        d2 = float((m.reconstruct(t_short, p_short) - m.reconstruct(t_long, p_long)).abs().max())
    ck("2. padding 长度不变", d2 < 1e-4, f"max|Δ| = {d2:.2e}")

    # 3. 空集合（哑 token，pad_mask 全 False 只留一位）
    t_e = torch.zeros(B, 1, 4 + P, device=dev)
    p_e = torch.zeros(B, 1, dtype=torch.bool, device=dev)
    with torch.no_grad():
        oe = m.reconstruct(t_e, p_e)
    ck("3. 空传感器集不产生 NaN", bool(torch.isfinite(oe).all()))

    # 4. 可微
    m.train()
    out = m.reconstruct(tok, pad)
    out.pow(2).sum().backward()
    no_grad = [n for n, p in m.named_parameters()
               if p.requires_grad and (p.grad is None or float(p.grad.abs().sum()) == 0)]
    ck("4. 全部参数有梯度", not no_grad, f"无梯度: {no_grad[:3]}" if no_grad else "")

    # 5. 维度断言
    try:
        build(dec_num_latent_channels=16)
        raised = False
    except ValueError:
        raised = True
    ck("5. dec != enc latent 通道数被拒绝", raised)

    print("\n" + ("ALL PASS" if not FAIL else f"FAILED: {FAIL}"))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
