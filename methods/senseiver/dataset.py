"""
dataset.py — ATC 的一天 -> Senseiver 的训练样本
================================================

**观测场景完全沿用 `4dvarnet_enkf/config.yaml`**（3 个机器人 / 感知半径 7 /
视线遮挡 / 同一套 obs_std / 同一个 obs_every_k / 同一个 walkable 规则），并且直接调用
它的 `observation_model.generate_observations` 和 `navigation.build_valid_mask_from_config`。
三种方法面对的观测完全一致，差异只来自方法本身——这是三方对比能成立的前提。

本目录不在任何地方给观测参数设默认值；要改就去改 `4dvarnet_enkf/config.yaml`。

样本的粒度
----------
参考实现的一个训练样本 = (一帧的传感器读数, 该帧的若干查询点)。我们照搬：
**一帧一个样本**，查询点是整张 432 格网格（见 `sensors.query_all` 的说明）。

相邻帧（1 秒间隔）高度冗余，用 `--stride` 抽帧，默认 4。这不是对论文的改动——
参考实现同样是从全部帧里随机抽 `training_frames` 帧来训练。
"""
from __future__ import annotations

import os
import sys

import numpy as np

# append 而不是 insert(0)：4dvarnet_enkf 里也有 losses.py，插到最前会把本目录的顶掉。
from crowdcore import config as cfg4d                                          # noqa: E402
from crowdcore import navigation as nav                                        # noqa: E402
from crowdcore import observation_model as om                                  # noqa: E402


def obs_config():
    """观测参数——逐字取自 4dvarnet_enkf/config.yaml。"""
    return dict(
        sensing_range=cfg4d.get("observation", "sensing_range"),
        num_agents=cfg4d.get("observation", "num_agents"),
        add_noise=cfg4d.get("observation", "add_noise"),
        obs_every_k=cfg4d.get("observation", "obs_every_k"),
        init_method=cfg4d.get("observation", "init_method"),
    )


def state_shape():
    """(C, H, W) —— 取自 config.yaml 的 grid.state_shape。"""
    return tuple(cfg4d.get("grid", "state_shape"))


def channels():
    return list(cfg4d.get("grid", "channels"))


# --------------------------------------------------------------------------- #
def load_day(path, stride=4, seed=0, frames=0, obs_every_k=None, add_noise=None):
    """一天 -> (X, Y, Omega)，都已抽帧并展平成 (N, C, HW) / (N, HW)。

    frames>0 时只取该天的前 frames 帧（评估时用来对齐别的方法跑过的帧数）。
    """
    oc = obs_config()
    X, _ = om.load_state(path)
    X = np.asarray(X, dtype=np.float32)
    if frames > 0:
        X = X[:frames]
    valid = nav.build_valid_mask_from_config(X)
    out = om.generate_observations(
        X, sensing_range=oc["sensing_range"], num_agents=oc["num_agents"],
        add_noise=oc["add_noise"] if add_noise is None else add_noise,
        seed=seed, valid_mask=valid,
        obs_every_k=oc["obs_every_k"] if obs_every_k is None else obs_every_k)

    idx = np.arange(0, X.shape[0], stride)
    T, C, H, W = X.shape
    Xs = X[idx].reshape(len(idx), C, H * W)
    Ys = out["Y"][idx].reshape(len(idx), C, H * W).astype(np.float32)
    Om = out["Omega"][idx].reshape(len(idx), H * W)
    return Xs, Ys, Om


class DayBank:
    """把若干天拼成一个可随机取批的样本池（参考实现也是全量驻留内存后随机抽帧）。

    内存：每天抽帧后约 2 x N x C x HW x 4B。stride=4 时一天约 138 MB，
    32 天约 4.4 GB —— sbatch 里申请 64G 足够。
    """

    def __init__(self, files, stride=4, seed=0, max_days=0, frames=0,
                 obs_every_k=None, add_noise=None, verbose=True):
        files = files[:max_days] if max_days else files
        Xs, Ys, Os = [], [], []
        for i, f in enumerate(files):
            x, y, o = load_day(f, stride, seed, frames, obs_every_k, add_noise)
            Xs.append(x); Ys.append(y); Os.append(o)
            if verbose:
                print(f"  [{i+1}/{len(files)}] {os.path.basename(f).split('_')[0]}: "
                      f"{x.shape[0]} 帧, 观测格/帧 {o.sum(1).mean():.1f}", flush=True)
        self.X = np.concatenate(Xs, 0)
        self.Y = np.concatenate(Ys, 0)
        self.Omega = np.concatenate(Os, 0)
        self.n = self.X.shape[0]

    def input_stats(self):
        """编码器输入的逐通道标准化统计量：在**被观测到的格子**上统计，
        因为进编码器的只有这些格子的读数。"""
        m = self.Omega                                        # (N, HW)
        vals = [self.Y[:, c][m] for c in range(self.Y.shape[1])]
        mean = np.array([v.mean() for v in vals], np.float32)
        std = np.array([max(v.std(), 1e-6) for v in vals], np.float32)
        return mean, std

    def batches(self, batch, rng, drop_last=True):
        order = rng.permutation(self.n)
        stop = (self.n // batch) * batch if drop_last else self.n
        for i in range(0, stop, batch):
            yield order[i:i + batch]
