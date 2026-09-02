"""
dataset.py — 把 ATC 的一天编成可训练的样本（论文版）
===================================================

**观测配置完全沿用 `4dvarnet_enkf/config.yaml`**（3 个机器人 / 半径 7 / 视线遮挡 / 同一套
`obs_std` / 同一个固定种子），两条技术路线面对的观测场景完全一致，差异只来自方法本身。
不做 DINCAE 2.0 §4 那种"按 track 随机丢弃"的增广 —— 那会改变观测场景。

与 DINCAE 的一处结构差别：**逐帧重建**（论文的样本也是"一个目标时刻 + 邻近若干帧输入"，
所以这一条其实是照搬，不是改动）。相邻帧高度冗余，按 `--stride` 抽帧当目标（默认 4）。

通道布局（`N_IN = 6 + 24 = 30`，见 `encoding.py`）：

    [0:2]    row, col
    [2:6]    cos/sin 日周期, cos/sin 周周期
    [6:30]   dt ∈ {-1,0,+1} × (残差[4], mask[4])

目标 (8,H,W)：每通道一对 (残差·mask, mask)，见 `encoding.encode_target`。
"""
from __future__ import annotations

import os
import sys

import h5py
import numpy as np
import torch

# append 而非 insert(0)：4dvarnet_enkf 里也有 losses.py，插到最前会把本目录的同名模块顶掉
sys.path.append("/scratch/work/zhangx29/Thesis_Project/4dvarnet_enkf")
import config as cfg4d                                            # noqa: E402
import observation_model as om                                     # noqa: E402

from state import StateStats, NCH                           # noqa: E402
from encoding import (FRESH_OFFSETS, N_IN, N_STATIC, encode_target,  # noqa: E402
                      observed_pair, static_channels)

CACHE_VER = 4   # 1=age/aggregation; 2=论文版; 3=+残差归一化; 4=+var 走 log1p


def obs_config():
    """观测参数——**逐字取自 4dvarnet_enkf/config.yaml**，不在这里另设默认值。"""
    return dict(
        sensing_range=cfg4d.get("observation", "sensing_range"),
        num_agents=cfg4d.get("observation", "num_agents"),
        obs_std=np.asarray(cfg4d.get("observation", "obs_std"), dtype=np.float32),
        add_noise=cfg4d.get("observation", "add_noise"),
        obs_every_k=cfg4d.get("observation", "obs_every_k", default=1),
        seed=0,                       # 与 4dvarnet_enkf 的 build_windows 一致（固定种子）
    )


def encode_day(fp, stats: StateStats, stride=4, max_frames=0):
    """一天 -> (inputs, target)，第 0 维是抽出来的目标帧。

    inputs (n, N_IN, H, W) float32 ; target (n, 2*NCH, H, W) float32
    """
    oc = obs_config()
    with h5py.File(fp, "r") as f:
        T_all = f["grid"].shape[0]
        T = min(T_all, max_frames) if max_frames else T_all
        X = f["grid"][:T]
        t_unix = f["time"][:T]

    obs = om.generate_observations(
        X, sensing_range=oc["sensing_range"], num_agents=oc["num_agents"],
        add_noise=oc["add_noise"], seed=oc["seed"], valid_mask=stats.valid,
        obs_std=oc["obs_std"], obs_every_k=oc["obs_every_k"])
    Y, M = obs["Y"][:, :NCH], obs["Omega_c"][:, :NCH]

    scaled, invvar = observed_pair(Y, M, stats.mean, stats.std)

    # 目标帧：抽帧，并留出相邻窗需要的边界
    lo, hi = -min(FRESH_OFFSETS), T - max(FRESH_OFFSETS)
    idx = np.arange(lo, hi, stride, dtype=np.int64)
    H, W = X.shape[2], X.shape[3]

    out = np.empty((len(idx), N_IN, H, W), dtype=np.float32)
    out[:, :N_STATIC] = static_channels(t_unix[idx], H, W)
    o = N_STATIC
    for dt in FRESH_OFFSETS:
        j = idx + dt
        out[:, o:o + NCH] = scaled[j]
        out[:, o + NCH:o + 2 * NCH] = invvar[j]
        o += 2 * NCH
    assert o == N_IN, (o, N_IN)

    tgt = encode_target(X[idx], stats.mean, stats.std, stats.valid)
    return out, tgt


def cache_key(stats: StateStats, stride):
    """缓存键——把所有会改变编码结果的东西都编进去，改了任何一项就自动重建。"""
    oc = obs_config()
    return (f"v{CACHE_VER}_s{stride}_nd{stats.n_train_days}_c{NCH}"
            f"_na{oc['num_agents']}_sr{oc['sensing_range']}"
            f"_k{oc['obs_every_k']}_sd{oc['seed']}")


def encode_day_cached(fp, stats, stride, max_frames, cache_dir):
    """带磁盘缓存的 `encode_day`。观测种子固定 => 编码确定性 => 只需算一次。

    输入存 float16（两片都是 O(1)，无溢出风险），目标存 float32。
    """
    if not cache_dir or max_frames:                 # max_frames 是调试用的截断，不缓存
        return encode_day(fp, stats, stride, max_frames)
    os.makedirs(cache_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(fp))[0]
    key = cache_key(stats, stride)
    xp = os.path.join(cache_dir, f"{stem}_{key}_x.npy")
    yp = os.path.join(cache_dir, f"{stem}_{key}_y.npy")
    if not (os.path.exists(xp) and os.path.exists(yp)):
        x, y = encode_day(fp, stats, stride, max_frames)
        # 先写临时文件再 rename：中断/并发时不会留下半个文件被当成有效缓存
        np.save(xp + ".tmp.npy", x.astype(np.float16)); os.replace(xp + ".tmp.npy", xp)
        np.save(yp + ".tmp.npy", y); os.replace(yp + ".tmp.npy", yp)
        print(f"  [cache] wrote {stem} x{x.shape}", flush=True)
    return np.load(xp, mmap_mode="r"), np.load(yp, mmap_mode="r")


class ChunkedDays:
    """按 chunk 载入若干天，组内跨天打乱，产出 minibatch。

    编码是按天的（观测掩膜沿时间连续），所以不做全数据集打乱。每个 epoch：
    打乱天的顺序 -> 每 `days_per_chunk` 天读一次 -> 组内打乱帧 -> 迭代 minibatch。
    """

    def __init__(self, files, stats, batch_size=64, stride=4, days_per_chunk=4,
                 max_frames=0, shuffle=True, seed=0, cache_dir=None):
        self.files, self.stats = list(files), stats
        self.batch_size, self.stride = batch_size, stride
        self.days_per_chunk, self.max_frames = days_per_chunk, max_frames
        self.shuffle, self.cache_dir = shuffle, cache_dir
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        order = list(range(len(self.files)))
        if self.shuffle:
            self.rng.shuffle(order)
        for s in range(0, len(order), self.days_per_chunk):
            xs, ts = [], []
            for di in order[s:s + self.days_per_chunk]:
                x, t = encode_day_cached(self.files[di], self.stats, self.stride,
                                         self.max_frames, self.cache_dir)
                # 缓存是 mmap 的 float16 -> 整块读进内存并升到 float32
                xs.append(np.asarray(x, dtype=np.float32))
                ts.append(np.asarray(t, dtype=np.float32))
            X = np.concatenate(xs); Tg = np.concatenate(ts)
            del xs, ts
            perm = self.rng.permutation(len(X)) if self.shuffle else np.arange(len(X))
            for b in range(0, len(perm), self.batch_size):
                j = perm[b:b + self.batch_size]
                yield (torch.from_numpy(X[j]), torch.from_numpy(Tg[j]))
            del X, Tg
