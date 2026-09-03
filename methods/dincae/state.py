"""
state.py — 状态定义：四通道、有效性规则、逐通道变换、逐格统计量
=====================================================

四件事都在这里：`CHANNELS`/`NCH`（四通道）、`channel_valid`（逐通道有效性规则）、
`CHANNEL_TRANSFORM`/`fwd_channel`/`inv_channel`（逐通道变换）、`StateStats`（逐格统计量）。

**逐格均值场 —— 一句话：每个格子在 32 个训练日上的平均值。**
（论文管这一项叫 climatology / `remove_mean`，那是海洋气象的行话；本目录一律叫"逐格均值场"，
术语对照见 README 的"术语"一节。）实测走廊内不同位置的"常态人流"差几十倍
（最繁忙的格子平均 0.26 人/格，角落里接近 0），这个差异跨越所有 92 天都不变。

DINCAE 在重建前先减掉它，网络工作在残差空间（1.0 §3 的 `remove_mean`；2.0 §2.3 对
辅助变量也一样）。参考实现 `reference/DINCAE.jl/src/data.jl:266-276`：

    meandata = sum(非 NaN 的值, dims=4) ./ sum(非 NaN, dims=4)

注意这是**对整个时间轴求一次均值**，每个像素每个变量一个数 —— **不按季节/时段分箱**。
周期性是靠输入里的 `cos/sin` 通道处理的，不是靠这张均值表。我们照此办理：逐格一个均值，
昼夜周期交给 `cos/sin(time-of-day)` 输入通道。

**为什么值得减**：网络因此只需判断"现在比平常多还是少"，不必再记住"这个位置平常多少人"
（后者一张查表就能精确给出，不需要学）。更重要的是盲区的兜底 —— 有 36.9% 的格子在
{t−1,t,t+1} 里完全没被观测到，减了均值之后输出 0 的含义是"和平常一样"，而不是"一个人
都没有"。实测只用这张均值表当预测，就比 carry-forward 填充好 30~39%。

**逐通道有效性**（`channel_valid`）—— 均值只在该通道**有定义**的 (帧,格) 上算，否则占位
符 0 会污染均值。这不是对论文的改动，而是论文"缺测=精度为零"这条原则在我们数据上的落实：

  density : 处处有定义（density=0 是真实测量："这里没人"）
  vx, vy  : `density > 0` —— 空格子的速度是占位符
            （`h5_to_grid.py`: `vel = where(density>0, vel/density, 0)`）
  var     : `vel_var > 0` ⟺ 格内至少 2 人
            （`h5_to_grid.py:128`: `vel_var[nnz <= 1] = 0`，1 个点的方差无定义）

输出: artifacts/state_stats.npz —— mean[NCH,H,W]、std[NCH]、count、valid_mask

用法（纯 numpy/scipy，登录节点可跑）:
    python3 state.py                # 全部 32 训练日
    python3 state.py --days 3       # 冒烟测试
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np

# append 而非 insert(0)：4dvarnet_enkf 里也有 losses.py，插到最前会把本目录的同名模块顶掉
from crowdcore import navigation as nav                                         # noqa: E402
from crowdcore import observation_model as om                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CHANNELS = ("density", "vx", "vy", "var")
NCH = len(CHANNELS)


def channel_valid(X):
    """(NCH, T, H, W) bool —— 每个通道在哪些 (帧, 格) 上**有定义**。见模块 docstring。

    在**原始物理值**上判定（与 CHANNEL_TRANSFORM 无关）。注意 log1p 是单调的，所以
    `var > 0` 和 `log1p(var) > 0` 是等价条件，这里用原始值只为可读性。
    """
    occ = X[:, 0] > 0
    return np.stack([np.ones_like(occ), occ, occ, X[:, 3] > 0])


# --------------------------------------------------------------------------- #
# 逐通道变换
# --------------------------------------------------------------------------- #
#: `var` 用 log1p。1.0 结论段：这套方法"可以很容易推广到参数化概率分布，特别是用于浓度类
#: 变量的 log-normal 分布"。`var`(格内速度方差) 正是这一类 —— 非负、重尾：它的
#: 残差标准差只有 0.23，但拥挤格子里 `vel_var` 能超过 2，归一化后仍有大量 10σ 级样本，
#: 少数极端点主导梯度。实测未变换时 `var` 的归一化 dev MSE 在 8~62 之间震荡
#: （1.0 = "只输出逐格均值场"的水平），即重建比什么都不做差一个数量级。
CHANNEL_TRANSFORM = (None, None, None, "log1p")


def fwd_channel(v, c):
    """原始物理值 -> 模型工作的空间（逐通道）。"""
    if CHANNEL_TRANSFORM[c] == "log1p":
        return np.log1p(np.maximum(v, 0.0))       # var 是方差，负值只可能来自观测噪声
    return v


#: `exp(μ + σ²/2)` 的指数上限。见 inv_channel 的说明。
_INV_EXP_CAP = 20.0


def inv_channel(v, c, var_in_space=None):
    """模型空间 -> 原始物理值（`fwd_channel` 的逆）。

    默认返回**中位数** `expm1(μ)`。

    `var_in_space` 传入时改用**对数正态均值** `exp(μ + σ²/2) − 1`（若 x ~ N(μ,σ²) 且
    y = expm1(x)，这才是 E[y]）。**但默认不要用它做点估计**：σ̂² 被 Eq.6 钳在 1/µ = 1000，
    乘回 std² 后 `0.5σ²` 能到 11，指数直接爆掉 —— 实测某次冒烟里 `var` 的物理空间 MSE
    因此达到 10²²。数学上没错，可是几个"我不知道"的格子会用天文数字统治 MSE，所以
    对**重尾且 σ̂ 未校准**的通道，中位数是更可用的点估计，而 `var` 的误差应主要在
    变换后（log）空间里看。这里对指数额外加了 `_INV_EXP_CAP` 兜底，防止溢出成 inf。
    """
    if CHANNEL_TRANSFORM[c] == "log1p":
        m = v if var_in_space is None else v + 0.5 * var_in_space
        return np.expm1(np.minimum(m, _INV_EXP_CAP))
    return v


class StateStats:
    """state_stats.npz 的读取。

    `mean[c,H,W]` 逐格时间均值；`std[c]` 该通道残差的标准差（标量，在 walkable ∩
    有定义的 (帧,格) 上算）。

    **为什么需要 std**：论文所有变量都用 `obs_err_std = 1`（代码默认），这在 SST 上没问题
    —— 它的残差量级本身就是 O(1) °C。我们四个通道的残差方差差三个数量级
    （density 0.036 / vx 0.53 / vy 0.13 / var 0.037），全按 σ²=1 处理会让高斯 NLL 失衡：
    `log σ̂²` 项无下界，残差量级远小于 1 的通道可以把 σ̂² 直接压到 Eq.6 的下限白赚一截
    负 loss，而重建并未变好（实测 train NLL 降 5 个单位、density 的 dev MSE 却从 0.0254
    涨到 0.0419）。把各通道残差归一到单位方差后，σ²_obs=1 就和数据量级匹配了，
    这也正是参考代码里 `normalize2`（`data.jl:111-120`）在做的事。
    """

    def __init__(self, path=os.path.join(HERE, "artifacts", "state_stats.npz")):
        z = np.load(path, allow_pickle=False)
        self.mean = z["mean"].astype(np.float32)          # (NCH,H,W)
        self.std = z["std"].astype(np.float32)            # (NCH,)  残差标准差
        self.count = z["count"]
        self.valid = z["valid_mask"].astype(bool)         # (H,W) walkable
        self.n_train_days = int(z["n_train_days"])
        assert self.mean.shape[0] == NCH and self.std.shape == (NCH,)
        assert (self.std > 0).all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0, help="只用前 N 个训练日(冒烟测试)")
    ap.add_argument("--out", default=os.path.join(HERE, "artifacts", "state_stats"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    train = om.split_files("train")
    if args.days:
        train = train[: args.days]
    valid = nav.build_valid_mask_from_config()        # 跨日一致(训练集 visited 并集)
    H, W = valid.shape
    print(f"train days {len(train)}, walkable {valid.sum()}/{valid.size}, channels {CHANNELS}")

    s = np.zeros((NCH, H, W))       # Σ x
    sq = np.zeros((NCH, H, W))      # Σ x²   (供残差方差用)
    n = np.zeros((NCH, H, W))       # 有定义的样本数
    for k, fp in enumerate(train):
        with h5py.File(fp, "r") as f:
            X = f["grid"][:]
        cv = channel_valid(X)
        for c in range(NCH):
            # 逐格均值场与残差标准差都在**变换后**的空间里算，与编码/损失口径一致
            v = np.where(cv[c], fwd_channel(X[:, c].astype(np.float64), c), 0.0)
            s[c] += v.sum(0)
            sq[c] += (v ** 2).sum(0)
            n[c] += cv[c].sum(0)
        print(f"  {k + 1}/{len(train)} {os.path.basename(fp)}", flush=True)

    mean = np.where(n > 0, s / np.maximum(n, 1), 0.0)
    # 残差方差（在 walkable 上汇总）：Σ_frames (x − mean_cell)² = Σx² − n·mean²
    # 所以一次遍历就能精确得到，不必再扫一遍数据。
    ss = np.where(valid[None], sq - n * mean ** 2, 0.0).sum(axis=(1, 2))
    nn = np.where(valid[None], n, 0.0).sum(axis=(1, 2))
    var_resid = ss / np.maximum(nn, 1)
    std = np.sqrt(np.maximum(var_resid, 1e-12))

    np.savez_compressed(args.out + ".npz", mean=mean, std=std, count=n,
                        valid_mask=valid, n_train_days=len(train),
                        channels=np.asarray(CHANNELS))
    with open(args.out + ".json", "w") as f:
        json.dump({"n_train_days": len(train), "channels": list(CHANNELS),
                   "walkable_cells": int(valid.sum()),
                   "mean_over_walkable": {c: float(mean[i][valid].mean())
                                          for i, c in enumerate(CHANNELS)},
                   "resid_std": {c: float(std[i]) for i, c in enumerate(CHANNELS)},
                   "resid_var": {c: float(var_resid[i]) for i, c in enumerate(CHANNELS)},
                   "samples_per_cell_median": {c: float(np.median(n[i][valid]))
                                               for i, c in enumerate(CHANNELS)}}, f, indent=2)

    print("\n逐格均值 + 残差标准差（归一化用）:")
    for i, c in enumerate(CHANNELS):
        print("  %-8s mean %9.5f   resid_std %8.5f   每格样本数中位数 %.0f"
              % (c, mean[i][valid].mean(), std[i], np.median(n[i][valid])))
    print(f"\nwrote {args.out}.npz / .json")


if __name__ == "__main__":
    main()
