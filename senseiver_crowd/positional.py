"""
positional.py — sin-cos 位置编码  a = PE(χ)
============================================

逐字移植自 `reference/Senseiver/positional.py`（论文 §2 的第一个组件，
Appendix 说明用的是 [23] Vaswani et al. 的 sine-cosine embeddings）。

相对参考实现的两处改动：

 1. `torch.meshgrid` 显式传 `indexing="ij"`。参考实现没传，新版 torch 会告警；
    `"ij"` 正是旧版的默认行为，语义完全不变。
 2. 增加 `encoding_channels()`，让调用方去问通道数，而不是在别处重复
    `2 * D * bands` 这个式子（参考实现在 `network_light.py` 里手写了一遍）。

输出约定（与参考实现一致）：
    PositionalEncoder(image_shape, bands) -> (prod(spatial_shape), 2*D*bands)
    行序 = 空间维度按 C 序展平，可直接用 `enc[flat_index]` 取任意格子的编码。
    通道序 = [sin(dim0), sin(dim1), ..., cos(dim0), cos(dim1), ...]
"""
from __future__ import annotations

import math

import torch
from einops import rearrange


def PositionalEncoder(image_shape, num_frequency_bands, max_frequencies=None):
    """image_shape 末位是通道数（沿用参考实现的签名：传 data.shape[1:]）。"""
    *spatial_shape, _ = image_shape

    coords = [torch.linspace(-1, 1, steps=s) for s in spatial_shape]
    pos = torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=len(spatial_shape))

    encodings = []
    if max_frequencies is None:
        max_frequencies = pos.shape[:-1]

    # 每个空间维度一组频率：linspace(1, dim/2, bands)，上界是该维度的 Nyquist 频率
    frequencies = [torch.linspace(1.0, max_freq / 2.0, num_frequency_bands)
                   for max_freq in max_frequencies]

    frequency_grids = []
    for i, frequencies_i in enumerate(frequencies):
        frequency_grids.append(pos[..., i:i + 1] * frequencies_i[None, ...])

    encodings.extend([torch.sin(math.pi * g) for g in frequency_grids])
    encodings.extend([torch.cos(math.pi * g) for g in frequency_grids])
    enc = torch.cat(encodings, dim=-1)
    enc = rearrange(enc, "... c -> (...) c")
    return enc


def encoding_channels(spatial_ndim, num_frequency_bands):
    """位置编码的通道数：每个空间维度 bands 个 sin + bands 个 cos。"""
    return 2 * spatial_ndim * num_frequency_bands
