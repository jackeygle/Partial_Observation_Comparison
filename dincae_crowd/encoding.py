"""
encoding.py — DINCAE 的 information-form 输入/目标编码（严格照论文）
====================================================================

DINCAE 不"填充"缺测。输入是两片：`y/σ²` 和 `1/σ²`，缺测 = 两者皆 0 = 精度为零
（1.0 §3；2.0 §2.3；参考实现 `reference/DINCAE.jl/src/data.jl:311-325`）。
输入、目标都在同一个"信息空间"坐标系里，所以**没有填充这一步**。

**σ²_obs 取常数 1**，即论文/代码的默认 `obs_err_std = 1`。1.0 §3 原话：这个常数的具体值
不重要，因为它会被第一层的权重矩阵吸收，而那个矩阵是训练出来的。取 1 的好处是两片天然
O(1)，不需要任何额外的尺度处理。于是：

    观测到  ->  scaled = (fwd(y) − mean)/std ,  invvar = 1
    缺测    ->  scaled = 0                   ,  invvar = 0

**时间上只用相邻 3 帧**（`ntime_win = 3`，1.0 §3 的"前一天/当天/后一天"）。一个格子若不在
{t−1, t, t+1} 里被观测到，它就是缺测 —— 论文里没有"沿用更早的观测"这回事。

**目标**：论文的目标是有缺测的观测本身（他们没有 truth）。**我们有完整 truth，这是唯一
一处无法照搬的地方** —— 我们用 truth 当目标，掩膜取"该通道在该处有没有定义"
（`channel_valid`）。把"无定义"当成缺测、精度置零，正是论文自己的机制。
"""
from __future__ import annotations

import numpy as np

from state import (CHANNELS, NCH, StateStats, channel_valid,  # noqa: F401
                         fwd_channel)

FRESH_OFFSETS = (-1, 0, 1)          # ntime_win = 3（论文默认，必须奇数）
CYCLE_PERIODS = (86400.0, 604800.0)  # 日周期 + 周周期(秒)。论文用 (365.25 d)，
#                                     参考代码的 `cycle_periods` 本就是一个列表；
#                                     ATC 的周期性在"一天之内"和"星期",不在一年之内。

N_STATIC = 2 + 2 * len(CYCLE_PERIODS)                    # row, col + cos/sin × 周期数
N_IN = N_STATIC + 2 * NCH * len(FRESH_OFFSETS)           # = 6 + 24 = 30
N_TGT = 2 * NCH                                          # 每通道一对 (a/σ²_true, 1/σ²_true)


def observed_pair(Y, M, mean, std):
    """观测 -> information form 两片（σ²_obs = 1，在**归一化的残差空间**里）。

    Input : Y (T,NCH,H,W) 观测值（未观测处为 0）；M (T,NCH,H,W) bool 观测掩膜
            mean (NCH,H,W) 逐格时间均值 ；std (NCH,) 该通道残差标准差
    Output: scaled (T,NCH,H,W) = ((y − mean)/std)·M ；invvar (T,NCH,H,W) = M

    除以 std 的理由见 `state.StateStats` 的 docstring：论文所有变量都用
    `obs_err_std = 1`，而我们四个通道的残差方差差三个数量级，不归一化会让高斯 NLL
    的 `log σ̂²` 项在小量级通道上白赚负 loss。归一化后 σ²_obs=1 与数据量级匹配。
    """
    m = M.astype(np.float32)
    scaled = np.empty(Y.shape, dtype=np.float32)
    for c in range(NCH):                              # 逐通道：var 走 log1p，见 CHANNEL_TRANSFORM
        a = (fwd_channel(Y[:, c].astype(np.float64), c) - mean[c][None]) / std[c]
        scaled[:, c] = (a * m[:, c]).astype(np.float32)
    return scaled, m


def static_channels(t_unix, H, W):
    """坐标 + 周期时间。论文输入清单里的经纬度和 cos/sin(day-of-year)
    （1.0 §3；代码 `data.jl:409-419`）。Output: (T, N_STATIC, H, W)"""
    T = len(t_unix)
    out = np.empty((T, N_STATIC, H, W), dtype=np.float32)
    out[:, 0] = np.linspace(-1, 1, H, dtype=np.float32)[:, None]
    out[:, 1] = np.linspace(-1, 1, W, dtype=np.float32)[None, :]
    for k, period in enumerate(CYCLE_PERIODS):
        ph = 2 * np.pi * (np.asarray(t_unix, dtype=np.float64) / period)
        out[:, 2 + 2 * k] = np.cos(ph).astype(np.float32)[:, None, None]
        out[:, 3 + 2 * k] = np.sin(ph).astype(np.float32)[:, None, None]
    return out


def encode_target(X, mean, std, walkable):
    """目标也用 information form，mask 由第二片是否为 0 决定（`model.jl:109-114`）。

    σ²_true = 1（论文没有 `truth_uncertain` 这个分支，它只存在于代码里），于是
        第 1 片 = 归一化残差·mask ，第 2 片 = mask
    mask = `channel_valid` ∩ walkable，即"该通道在此处有定义"。
    归一化口径与输入一致（同一个 mean / std），见 `observed_pair`。

    Output: (T, N_TGT, H, W)
    """
    T, _, H, W = X.shape
    out = np.zeros((T, N_TGT, H, W), dtype=np.float32)
    cv = channel_valid(X)                                        # (NCH,T,H,W)
    for c in range(NCH):
        m = (cv[c] & walkable[None]).astype(np.float32)
        a = (fwd_channel(X[:, c].astype(np.float64), c) - mean[c][None]) / std[c]
        out[:, 2 * c] = (a * m).astype(np.float32)
        out[:, 2 * c + 1] = m
    return out
