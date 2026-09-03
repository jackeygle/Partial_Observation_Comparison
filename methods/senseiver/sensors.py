"""
sensors.py — 观测 -> 变长传感器 token 集
==========================================

参考实现里这一步只有两行（`dataloaders.py:164-165`）：

    sensor_values = self.indexed_sensors[frames,]                     # (B, Ns, C)
    sensor_values = torch.cat([sensor_values, self.sensor_positions], -1)

因为它的传感器是**固定的一组格子**，Ns 是常数，位置编码可以预先 repeat 好。
我们的传感器是**移动机器人**，每一帧看到的格子集合和数量都不同，所以这一步
必须重做：逐帧取观测格 -> 拼位置编码 -> padding 到批内最大长度 -> 生成 pad_mask。

一个 "传感器" = 一个被观测到的格子；它的读数是该格的 C 维向量，这正是论文
`s_i ∈ R^{N_c}`（多通道传感器）的设定。token = [C 个通道值, P 维位置编码]，
与参考实现的拼接顺序一致。

两处必须自己定的事（参考实现给不出答案）：

 1. **输入标准化**。参考实现的 5 个数据集全是单通道，`datasets.py` 里只做了一个
    全局标量除法（`sea /= sea.max()`、`cyl / 11.0960`）。我们有 4 个尺度差一个
    量级以上的通道（std 分别约 0.17 / 0.46 / 0.16 / 0.11）。这里做**逐通道标准化，
    且只作用于编码器输入**——目标与损失一律留在原始场上。理由：对比口径是原始场
    上的四通道无权重 MSE，任何对目标的逐通道缩放都等价于偷偷给损失加权重。
    输入侧的标准化只是特征调理，不改变被优化的量。

 2. **空传感器集**。`obs_every_k > 1` 时有的帧一个观测都没有，此时 cross-attention
    的 K/V 为空，softmax 会产生 NaN。这里放一个全零的哑 token 并把它标为有效，
    模型对这种帧只能输出一个常数场——这是信息上的事实，不是实现缺陷。
"""
from __future__ import annotations

import numpy as np
import torch


def build_batch(Y_flat, Omega_flat, pos_enc, in_mean, in_std):
    """把一批帧的观测打成 padding 好的传感器 token 批。

    输入
        Y_flat     (B, C, HW) float32   带噪的部分观测（未观测处为 0）
        Omega_flat (B, HW)    bool      该帧哪些格子被观测到
        pos_enc    (HW, P)    float32   全网格的位置编码（positional.PositionalEncoder）
        in_mean/in_std (C,)   float32   编码器输入的逐通道标准化统计量

    输出
        tokens   (B, Nmax, C+P) float32 torch
        pad_mask (B, Nmax)      bool    torch，True = padding 位（送给
                                        nn.MultiheadAttention 的 key_padding_mask）
        n_sens   (B,)           int     每帧真实的传感器数（诊断用）
    """
    B, C, _ = Y_flat.shape
    P = pos_enc.shape[1]
    idx = [np.flatnonzero(Omega_flat[b]) for b in range(B)]
    n_sens = np.array([len(i) for i in idx], dtype=np.int64)
    n_max = max(1, int(n_sens.max()))

    tokens = np.zeros((B, n_max, C + P), dtype=np.float32)
    pad_mask = np.ones((B, n_max), dtype=bool)              # 先全标为 padding
    for b, ii in enumerate(idx):
        if len(ii) == 0:
            pad_mask[b, 0] = False                          # 空集合：留一个全零哑 token
            continue
        v = Y_flat[b][:, ii].T                              # (n_b, C)
        tokens[b, :len(ii), :C] = (v - in_mean) / in_std    # 只标准化输入
        tokens[b, :len(ii), C:] = pos_enc[ii]
        pad_mask[b, :len(ii)] = False

    return (torch.from_numpy(tokens), torch.from_numpy(pad_mask),
            torch.from_numpy(n_sens))


def query_all(pos_enc, batch):
    """查询整张网格。返回 (batch, HW, P)。

    参考实现每步只随机查 `batch_pixels` 个像素，因为它的域大到无法整张查
    （3D 孔隙是 128x128x512）。ATC 只有 36x12=432 格，整张查一次的代价可以忽略，
    所以不做像素抽样——这减少了一个与论文无关的随机性来源。

    参考实现的 `pix_avail`(值为 0 的格子不参与) 在这里**直接取消**：它的用途是
    跳过"没有值可重建"的格子（海温的大陆、孔隙的固体）。ATC 网格上不存在这样的
    格子——非 walkable 区同样有真值，评估口径也会给它打分——所以全部 432 格都要查。
    """
    return pos_enc[None].expand(batch, -1, -1)
