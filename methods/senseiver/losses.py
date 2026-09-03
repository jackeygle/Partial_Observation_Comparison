"""
losses.py — 训练损失与诊断拆分
================================

**训练损失照抄参考实现**：`network_light.py:68` 的
`F.mse_loss(pred_values, field_values, reduction='sum')`，原始场上的四通道无权重
平方误差。不加任何逐通道/逐格权重——那已经不是论文的方法了。

`reduction='sum'` 沿用参考实现。它相对 `'mean'` 只差一个常数因子，而在 Adam 下
这个因子被逐参数的二阶矩归一化基本吸收掉，所以是等价的；保留 `'sum'` 是为了和
参考实现逐字对齐。日志里记的是除以元素数之后的值（参考实现也是这么记的）。

下面的拆分函数只用于**诊断**，不进入梯度。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def senseiver_loss(pred, target):
    """论文/参考实现的训练损失。pred/target: (B, Nq, C) 或 (B, C, H, W)。"""
    return F.mse_loss(pred, target, reduction="sum")


@torch.no_grad()
def diagnostics(pred, target, obs_mask, channels):
    """逐通道 / 观测区 vs 盲区 的 MSE 拆分。

    pred/target (B,C,H,W)，obs_mask (B,H,W) bool（True = 该格被观测到）。
    另外报"占用格"(density>0) 上的密度 MSE：单一 MSE 会被大量空格子摊薄，
    看不出模型是否只是在输出接近零的平滑场。
    """
    se = (pred - target) ** 2
    obs = obs_mask[:, None].expand_as(se)
    out = {"mse": float(se.mean()),
           "mse_blind": float(se[~obs].mean()) if (~obs).any() else float("nan"),
           "mse_obs": float(se[obs].mean()) if obs.any() else float("nan")}
    for i, c in enumerate(channels):
        out[f"mse_{c}"] = float(se[:, i].mean())
    occ = target[:, 0] > 0
    out["mse_density_occ"] = float(se[:, 0][occ].mean()) if occ.any() else float("nan")
    out["occ_frac"] = float(occ.float().mean())
    return out
