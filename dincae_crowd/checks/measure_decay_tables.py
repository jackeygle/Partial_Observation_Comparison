"""
calibrate_decay.py — 标定 age-dependent σ² 需要的表
===================================================

DINCAE 的输入是 `y/σ²` 和 `1/σ²`，缺测 = 两者皆 0（1.0 §3 / 2.0 §2.3）。我们的观测还多一个
维度：**它有多旧**。诊断显示未观测格子上"最近一次观测"的年龄中位数是 9 s，而场 2 s 就去相关
（见 README），所以照搬 carry-forward 会系统性地喂过期数据。

正确做法是把 age 秒前的观测按其相关性向逐格均值场收缩（BLUE）：

    ŷ(age)      = stats + AC(age) · (y_old − stats)
    σ²_eff(age) = AC(age)² · σ²_obs + (1 − AC(age)²) · Var_resid

age=0 时退化为 σ²_obs；age→∞ 时 AC→0，σ²_eff→Var_resid，输入自动变成"无信息"。
本脚本产出标定这条公式所需的一切：

  1. **逐格均值场** stats[c, cell, bin] —— 逐格 × time-of-day 分箱均值（在训练日上算）。
     箱宽由**留出日的 MSE** 选，不是拍的。
  2. **AC_c(lag)** —— 残差（已减逐格均值场）的时间自相关，逐通道
  3. **Var_resid[c, cell]** —— 逐格残差方差（= AC→0 时 σ²_eff 的上限）
  4. **P_occ(lag)** —— P(t+lag 有人 | t 有人)。速度信息衰减有两个原因：流动变了、以及
     格子空了。AC 只管第一个，这张表管第二个。

**逐通道有效性**（`channel_valid`）—— 4 个通道"在哪些 (帧,格) 上有定义"各不相同，所有
统计量都只在有定义的地方算，否则占位符 0 会污染统计：

  density : 处处有定义（density=0 是一个真实测量："这里没人"）
  vx, vy  : `density > 0` —— 空格子的速度是占位符
            （`h5_to_grid.py`: `vel = where(density>0, vel/density, 0)`）
  var     : `vel_var > 0` ⟺ 格内至少 2 个人
            （`h5_to_grid.py:128`: `vel_var[nnz <= 1] = 0`，1 个点的方差无定义）

输出: decay_tables.npz + decay_tables.json（人类可读摘要）

用法（纯 numpy/scipy，登录节点可跑）:
    python3 calibrate_decay.py                  # 全部 32 训练日
    python3 calibrate_decay.py --days 3         # 冒烟测试
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np

# checks/ 里的脚本从项目根导入源码模块；根要插在最前(本目录的 losses.py 优先)，
# 4dvarnet_enkf 只能 append(它也有 losses.py，插到最前会把本目录的顶掉)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("/scratch/work/zhangx29/Thesis_Project/4dvarnet_enkf")
import config                                                    # noqa: E402
import navigation as nav                                         # noqa: E402
import observation_model as om                                    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ATC 在大阪 (UTC+9)。time 是 unix 秒；time-of-day 必须用当地时间，否则分箱会跨日切断。
TZ_OFFSET = 9 * 3600
BASE_BIN = 300                       # 累加分辨率(s)；候选箱宽必须是它的整数倍
#   86400 = 单箱 = 退化成"逐格全时段均值"(不做 time-of-day 分箱)，作为对照下界。
#   候选必须两端都留出余量，否则"数据驱动"的选择只是撞到搜索范围的边界。
BIN_CANDIDATES = (300, 900, 1800, 3600, 7200, 14400, 86400)
CHANNELS = ("density", "vx", "vy", "var")
NCH = len(CHANNELS)

# 稠密到几何递增：p90(age)=35 s，尾部留到 512 s
LAGS = (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)


def tod_bin(t: np.ndarray) -> np.ndarray:
    """unix 秒 -> BASE_BIN 分辨率的 time-of-day 箱号。"""
    return (((t.astype(np.int64) + TZ_OFFSET) % 86400) // BASE_BIN).astype(np.int64)


def channel_valid(X):
    """(NCH, T, H, W) bool —— 每个通道在哪些 (帧, 格) 上**有定义**。见模块 docstring。"""
    occ = X[:, 0] > 0
    return np.stack([np.ones_like(occ), occ, occ, X[:, 3] > 0])


def load_day(fp):
    with h5py.File(fp, "r") as f:
        return f["grid"][:], f["time"][:]


# --------------------------------------------------------------------------- #
# Pass A：累加逐格均值场（BASE_BIN 分辨率，之后池化到更宽的箱）
# --------------------------------------------------------------------------- #
def accumulate_mean_field(files, valid):
    nbin = 86400 // BASE_BIN
    H, W = valid.shape
    s = np.zeros((NCH, H, W, nbin))          # Σ 值（只累加有定义的）
    n = np.zeros((NCH, H, W, nbin))          # 有定义的样本数

    for k, fp in enumerate(files):
        X, t = load_day(fp)
        b = tod_bin(t)
        cv = channel_valid(X)                                    # (NCH,T,H,W)
        for bi in np.unique(b):
            sel = b == bi
            for c in range(NCH):
                v = cv[c][sel]
                s[c, :, :, bi] += np.where(v, X[sel, c], 0.0).sum(0)
                n[c, :, :, bi] += v.sum(0)
        print(f"  [stats] {k + 1}/{len(files)} {os.path.basename(fp)}", flush=True)
    return s, n


def pool_mean_field(s, n, bin_width):
    """池化到 bin_width。样本数为 0 的箱回退到该格子的全时段均值，全无样本则 0。"""
    assert bin_width % BASE_BIN == 0
    g = bin_width // BASE_BIN
    pool = lambda a: a.reshape(*a.shape[:-1], -1, g).sum(-1)
    sp, np_ = pool(s), pool(n)

    stats = np.zeros_like(sp)
    tot_s, tot_n = sp.sum(-1, keepdims=True), np_.sum(-1, keepdims=True)
    fb = np.where(tot_n > 0, tot_s / np.maximum(tot_n, 1), 0.0)     # 该格子的全时段均值
    stats = np.where(np_ > 0, sp / np.maximum(np_, 1), fb)
    return stats


def score_mean_field(files, valid, clims):
    """留出日上的逐格均值场 MSE，逐通道只在**有定义**的 (帧,格) 上算。

    `clims` = {bin_width: stats}。每个 dev 日**只读一次**，同时给所有候选箱宽打分
    （否则 7 候选 × 7 天 = 49 次 280 MB 读盘）。
    """
    se = {bw: np.zeros(NCH) for bw in clims}
    cnt = {bw: np.zeros(NCH) for bw in clims}
    for fp in files:
        X, t = load_day(fp)
        tb = tod_bin(t)
        cv = channel_valid(X)[:, :, valid]                        # (NCH,T,N)
        Xv = np.stack([X[:, c][:, valid] for c in range(NCH)])     # (NCH,T,N)
        for bw, stats in clims.items():
            b = tb // (bw // BASE_BIN)
            pred = np.moveaxis(stats[:, :, :, b], -1, 1)[:, :, valid]
            for c in range(NCH):
                d = (Xv[c] - pred[c])[cv[c]]
                se[bw][c] += (d ** 2).sum(); cnt[bw][c] += d.size
        print(f"  [score] {os.path.basename(fp)}", flush=True)
    return {bw: se[bw] / np.maximum(cnt[bw], 1) for bw in clims}


# --------------------------------------------------------------------------- #
# Pass B：残差的时间自相关 + 占用持续概率
# --------------------------------------------------------------------------- #
def accumulate_autocorr(files, valid, stats, bin_width):
    g = bin_width // BASE_BIN
    nL = len(LAGS)
    cross = np.zeros((NCH, nL))       # Σ a_t · a_{t+L}
    e_lo = np.zeros((NCH, nL))        # Σ a_t²      （同样的配对）
    e_hi = np.zeros((NCH, nL))        # Σ a_{t+L}²
    cnt = np.zeros((NCH, nL))
    occ_pair = np.zeros(nL)
    occ_lo = np.zeros(nL)
    N = int(valid.sum())
    var_cell = np.zeros((NCH, N))
    var_n = np.zeros((NCH, N))

    for k, fp in enumerate(files):
        X, t = load_day(fp)
        b = tod_bin(t) // g
        pred = np.moveaxis(stats[:, :, :, b], -1, 1)[:, :, valid]   # (NCH,T,N)
        cv = channel_valid(X)[:, :, valid]                         # (NCH,T,N)
        occ = cv[1]                                                # density>0
        a = np.stack([X[:, c][:, valid] - pred[c] for c in range(NCH)])

        for c in range(NCH):
            var_cell[c] += np.where(cv[c], a[c] ** 2, 0.0).sum(0)
            var_n[c] += cv[c].sum(0)

        for li, L in enumerate(LAGS):
            lo = slice(None, None) if L == 0 else slice(None, -L)
            hi = slice(None, None) if L == 0 else slice(L, None)
            occ_pair[li] += (occ[lo] & occ[hi]).sum(); occ_lo[li] += occ[lo].sum()
            for c in range(NCH):
                m = cv[c][lo] & cv[c][hi]                          # 两端都有定义
                x, y = a[c][lo], a[c][hi]
                cross[c, li] += (x * y * m).sum()
                e_lo[c, li] += (x * x * m).sum()
                e_hi[c, li] += (y * y * m).sum()
                cnt[c, li] += m.sum()
        print(f"  [ac] {k + 1}/{len(files)} {os.path.basename(fp)}", flush=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        ac = cross / np.sqrt(np.maximum(e_lo * e_hi, 1e-30))
        p_occ = occ_pair / np.maximum(occ_lo, 1)
        var_cell = var_cell / np.maximum(var_n, 1)
    return np.nan_to_num(ac), p_occ, var_cell, cnt


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0, help="只用前 N 个训练日(冒烟测试)")
    ap.add_argument("--out", default=os.path.join(ROOT, "decay_tables"))
    args = ap.parse_args()

    train = om.split_files("train")
    dev_days = om.split_files("valid")
    if args.days:
        train, dev_days = train[: args.days], dev_days[:1]
    print(f"train days {len(train)}, dev days {len(dev_days)}, channels {CHANNELS}")

    valid = nav.build_valid_mask_from_config()               # 跨日一致(训练集 visited 并集)
    print(f"walkable cells {valid.sum()}/{valid.size}")

    print("Pass A: state_stats")
    s, n = accumulate_mean_field(train, valid)

    print("\n箱宽选择(dev 日 MSE，越小越好):")
    clims = {bw: pool_mean_field(s, n, bw) for bw in BIN_CANDIDATES}
    scores = score_mean_field(dev_days, valid, clims)
    for bw in BIN_CANDIDATES:
        tag = "  (= 无 time-of-day 分箱)" if bw == 86400 else ""
        print("  bin %6ds : " % bw
              + "  ".join("%s %.5f" % (c, scores[bw][i]) for i, c in enumerate(CHANNELS))
              + tag)
    best = min(BIN_CANDIDATES, key=lambda bw: scores[bw][0])
    if best in (BIN_CANDIDATES[0], BIN_CANDIDATES[-1]):
        print(f"  ! 最优落在候选范围的边界({best}s) —— 该扩大 BIN_CANDIDATES 再选一次")
    print(f"  -> 选 bin_width = {best} s  (按 density 的 dev MSE)")
    stats = clims[best]

    print("\nPass B: residual autocorrelation")
    ac, p_occ, var_cell, cnt = accumulate_autocorr(train, valid, stats, best)

    obs_std = np.asarray(config.get("observation", "obs_std"), dtype=float)

    np.savez_compressed(
        args.out + ".npz",
        lags=np.asarray(LAGS), ac=ac, p_occ=p_occ,
        var_cell=var_cell, stats=stats, bin_width=best,
        valid_mask=valid, obs_std=obs_std, channels=np.asarray(CHANNELS),
        n_train_days=len(train),
    )
    with open(args.out + ".json", "w") as f:
        json.dump({
            "n_train_days": len(train), "bin_width_s": int(best),
            "channels": list(CHANNELS),
            "bin_width_dev_mse": {str(k): v.tolist() for k, v in scores.items()},
            "walkable_cells": int(valid.sum()), "lags": list(LAGS),
            "ac": {c: ac[i].tolist() for i, c in enumerate(CHANNELS)},
            "p_occ_persist": p_occ.tolist(),
            "var_resid_mean": {c: float(var_cell[i].mean()) for i, c in enumerate(CHANNELS)},
            "obs_std": obs_std.tolist(), "obs_var": (obs_std ** 2).tolist(),
        }, f, indent=2)

    print("\n=== AC(lag) ===")
    print("lag(s)   " + " ".join("%6d" % L for L in LAGS))
    for i, c in enumerate(CHANNELS):
        print("%-8s " % c + " ".join("%6.3f" % v for v in ac[i]))
    print("P(occ|occ)" + " ".join("%6.3f" % v for v in p_occ))
    print("\nVar_resid (mean over cells): "
          + "  ".join("%s %.5f" % (c, var_cell[i].mean()) for i, c in enumerate(CHANNELS)))
    print("σ²_obs                    : "
          + "  ".join("%s %.5f" % (c, (obs_std ** 2)[i]) for i, c in enumerate(CHANNELS)))
    print(f"\nwrote {args.out}.npz / .json")


if __name__ == "__main__":
    main()
