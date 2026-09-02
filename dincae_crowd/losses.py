"""
losses.py — DINCAE 的高斯负对数似然（含 truth_uncertain 的 KL 分支）
=====================================================================

对照参考实现 `reference/DINCAE.jl/src/model.jl:54-148`。论文 2.0 Eq.3：

    J = 1/(2N) · Σ [ ((y − ŷ)/σ̂)² + log σ̂² + 2log√(2π) ]

代码里省掉了 1/2 和常数项（不影响梯度），并且多了两个论文里没写清的东西：

  1. **无观测处的 log 项被置零**：`σ²_masked = σ²_rec·mask + (1 − mask)`，于是 log(1)=0。
     这一步很关键 —— 否则网络可以在没有目标的格子上把 σ̂ 压到下限来降低 log 项。
  2. **归一化用的 N 不参与求导**（`ChainRulesCore.ignore_derivatives`）。N 每个 minibatch
     都不同（1.0 §3.1 特别强调了这点）。

**多变量**（`model.jl:132-148`）：每个输出变量**独立算、各自用自己的 N 归一化、然后直接相加**。
所以通道之间的相对权重完全由学出来的 1/σ̂² 决定，不需要手工设 —— 这正是它和"手工加权"
的本质区别（后者在本项目里失败过 8 次，见 4dvarnet_enkf 的记录）。

**truth_uncertain 分支**（`model.jl:68-86`，两篇论文都没写）：当"真值"自己带不确定度 σ²_true
时，损失换成两个高斯之间的 KL：

    2·KL(p_true ‖ q_rec) = log(σ²_rec/σ²_true) + (σ²_true + (m_rec − m_true)²)/σ²_rec − 1

对我们有用是因为网格化的 truth 本身有误差：速度的 σ²_true = `vel_var / density`
（格内速度方差 / 人数 = 格均值的标准误），`vel_var` 就是 state 的第 4 通道。
"""
from __future__ import annotations

import torch


def dincae_cost_single(m_rec, s2_rec, m_true, s2_true, mask, truth_uncertain=False):
    """单个输出变量的代价。所有张量 (B,1,H,W) 或 (B,H,W)，mask 为 0/1 float。"""
    n = mask.sum().detach().clamp(min=1.0)          # N 不参与求导
    if truth_uncertain:
        ratio = (s2_rec / s2_true) * mask + (1.0 - mask)        # 无观测处 = 1 -> log = 0
        d2 = ((m_rec - m_true) ** 2 + s2_true) * mask
        return (torch.log(ratio).sum() + (d2 / s2_rec).sum()) / n
    s2n = s2_rec * mask + (1.0 - mask)
    d2 = ((m_rec - m_true) ** 2) * mask
    return (torch.log(s2n).sum() + (d2 / s2_rec).sum()) / n


def decode_target(target, s2_floor=1e-6):
    """把 information-form 的目标解回 (m_true, σ²_true, mask)。

    target (B, 2*nvar, H, W)：偶数片 = a/σ²_true，奇数片 = 1/σ²_true。
    mask = (1/σ²_true ≠ 0) —— 缺测/无定义 = 精度为零（`model.jl:109-114`）。
    """
    inv = target[:, 1::2]
    mask = (inv != 0).to(target.dtype)
    s2 = 1.0 / torch.clamp(inv, min=s2_floor)                   # 掩掉的位置这个值无意义
    m = target[:, 0::2] * s2
    return m * mask, s2, mask


def dincae_loss(outs, target, loss_weights, truth_uncertain=False):
    """全部输出级 × 全部变量的总损失（2.0 Eq.4）。

    outs : [(mean, σ²), ...] 每级一个，来自 DINCAE.forward
    返回 (total, per_level)  —— per_level 便于日志里看精化步有没有起作用
    """
    m_true, s2_true, mask = decode_target(target)
    per_level = []
    total = target.new_zeros(())
    for w, (m_rec, s2_rec) in zip(loss_weights, outs):
        lvl = target.new_zeros(())
        for c in range(m_rec.shape[1]):                          # 逐变量独立归一化后相加
            lvl = lvl + dincae_cost_single(
                m_rec[:, c], s2_rec[:, c], m_true[:, c], s2_true[:, c], mask[:, c],
                truth_uncertain=truth_uncertain)
        per_level.append(lvl)
        total = total + w * lvl
    return total, per_level


@torch.no_grad()
def residual_mse(outs, target, per_channel=True):
    """诊断用：最后一级在有效格上的残差 MSE（和损失同口径的掩膜，但不含 σ̂ 项）。

    这是给人看的指标 —— 训练目标是 NLL，但 MSE 才能和 4dvarnet_enkf 的数对上量级。
    """
    m_true, _, mask = decode_target(target)
    m_rec, _ = outs[-1]
    se = ((m_rec - m_true) ** 2) * mask
    if per_channel:
        n = mask.sum(dim=(0, 2, 3)).clamp(min=1.0)
        return (se.sum(dim=(0, 2, 3)) / n)
    return se.sum() / mask.sum().clamp(min=1.0)
