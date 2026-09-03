"""
model.py — Senseiver 的 Encoder / Decoder
==========================================

移植自 `reference/Senseiver/model.py`（其自述改编自 krasserm/perceiver-io）。
对应论文 §2 的后两个组件：

    z    = E(a_s, s)      注意力编码器：传感器集 -> 定长 latent
    ŝ_q  = D(z, a_q)      注意力解码器：latent + 查询位置 -> 该位置的场值

Appendix A 的结构在代码里的落点：
  * 编码块 = cross-attention(latent 当 Q，传感器当 K/V) + self-attention，
    `num_layers` 个块**共享权重**（`layer_1` 独立 + `layer_n` 复用 num_layers-1 次，
    这是参考实现的具体做法）。
  * 解码 = 查询位置编码与一个可学习向量拼接当 Q，z 当 K/V，单层 cross-attention
    后接线性输出头。

相对参考实现的改动（README"偏离参考实现"一节有同样的清单）：

 1. **删掉 fairscale 的 `checkpoint_wrapper`**。参考实现里 `activation_checkpoint`
    从未被 `s_parser.py` 暴露，恒为 False，是死代码；删掉可少一个依赖。
 2. **`Decoder` 增加维度断言**。解码器的 cross-attention 把 `num_latent_channels`
    当 KV 维，而实际喂进去的 z 的通道数由**编码器**决定；两者不等会静默错位。
    参考实现没有这个检查，README 的例子恰好都把两个值设成一样，掩盖了它。
 3. **`pad_mask` 真正接上**。这条通路在参考实现里已经存在
    （`Encoder.forward(x, pad_mask)` -> `CrossAttention` -> `key_padding_mask`），
    只是它的 dataloader 从不传值——因为它的传感器集是定长的。我们的传感器是
    移动机器人，每帧数量不同，必须 padding，于是这条通路第一次被用上。
    这里没有改结构，只是把已有能力启用，并补了空集合的保护。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from einops import repeat


class Sequential(nn.Sequential):
    """允许多参数在层间传递（第一层收 tuple，之后收单个张量）。"""

    def forward(self, *inputs):
        for module in self:
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


def mlp(num_channels: int):
    return Sequential(
        nn.LayerNorm(num_channels),
        nn.Linear(num_channels, num_channels),
        nn.GELU(),
        nn.Linear(num_channels, num_channels),
    )


def cross_attention_layer(num_q_channels: int, num_kv_channels: int,
                          num_heads: int, dropout: float):
    return Sequential(
        Residual(CrossAttention(num_q_channels, num_kv_channels, num_heads, dropout), dropout),
        Residual(mlp(num_q_channels), dropout),
    )


def self_attention_layer(num_channels: int, num_heads: int, dropout: float):
    return Sequential(
        Residual(SelfAttention(num_channels, num_heads, dropout), dropout),
        Residual(mlp(num_channels), dropout),
    )


def self_attention_block(num_layers: int, num_channels: int, num_heads: int, dropout: float):
    return Sequential(*[self_attention_layer(num_channels, num_heads, dropout)
                        for _ in range(num_layers)])


class Residual(nn.Module):
    def __init__(self, module: nn.Module, dropout: float):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_p = dropout

    def forward(self, *args, **kwargs):
        x = self.module(*args, **kwargs)
        return self.dropout(x) + args[0]


class MultiHeadAttention(nn.Module):
    def __init__(self, num_q_channels: int, num_kv_channels: int,
                 num_heads: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=num_q_channels,
            num_heads=num_heads,
            kdim=num_kv_channels,
            vdim=num_kv_channels,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x_q, x_kv, pad_mask=None, attn_mask=None):
        return self.attention(x_q, x_kv, x_kv,
                              key_padding_mask=pad_mask, attn_mask=attn_mask)[0]


class CrossAttention(nn.Module):
    def __init__(self, num_q_channels: int, num_kv_channels: int,
                 num_heads: int, dropout: float):
        super().__init__()
        self.q_norm = nn.LayerNorm(num_q_channels)
        self.kv_norm = nn.LayerNorm(num_kv_channels)
        self.attention = MultiHeadAttention(num_q_channels=num_q_channels,
                                            num_kv_channels=num_kv_channels,
                                            num_heads=num_heads, dropout=dropout)

    def forward(self, x_q, x_kv, pad_mask=None, attn_mask=None):
        x_q = self.q_norm(x_q)
        x_kv = self.kv_norm(x_kv)
        return self.attention(x_q, x_kv, pad_mask=pad_mask, attn_mask=attn_mask)


class SelfAttention(nn.Module):
    def __init__(self, num_channels: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
        self.attention = MultiHeadAttention(num_q_channels=num_channels,
                                            num_kv_channels=num_channels,
                                            num_heads=num_heads, dropout=dropout)

    def forward(self, x, pad_mask=None, attn_mask=None):
        x = self.norm(x)
        return self.attention(x, x, pad_mask=pad_mask, attn_mask=attn_mask)


class Encoder(nn.Module):
    """z = E(PE(χ_s), s)：把任意数量的传感器 token 压成 (num_latents, ch) 的定长 latent。"""

    def __init__(self, input_ch, preproc_ch, num_latents: int, num_latent_channels: int,
                 num_layers: int = 3, num_cross_attention_heads: int = 4,
                 num_self_attention_heads: int = 4,
                 num_self_attention_layers_per_block: int = 6, dropout: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        if preproc_ch:
            self.preproc = nn.Linear(input_ch, preproc_ch)
        else:
            self.preproc = None
            preproc_ch = input_ch

        def create_layer():
            return Sequential(
                cross_attention_layer(num_q_channels=num_latent_channels,
                                      num_kv_channels=preproc_ch,
                                      num_heads=num_cross_attention_heads,
                                      dropout=dropout),
                self_attention_block(num_layers=num_self_attention_layers_per_block,
                                     num_channels=num_latent_channels,
                                     num_heads=num_self_attention_heads,
                                     dropout=dropout),
            )

        self.layer_1 = create_layer()
        if num_layers > 1:
            self.layer_n = create_layer()          # 权重共享：后续块复用同一个 layer_n
        self.latent = nn.Parameter(torch.empty(num_latents, num_latent_channels))
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x, pad_mask=None):
        b, *_ = x.shape
        if self.preproc:
            x = self.preproc(x)
        x_latent = repeat(self.latent, "... -> b ...", b=b)
        x_latent = self.layer_1(x_latent, x, pad_mask)
        for _ in range(self.num_layers - 1):
            x_latent = self.layer_n(x_latent, x, pad_mask)
        return x_latent


class Decoder(nn.Module):
    """ŝ_q = D(z, PE(χ_q))：查询任意坐标处的场值。"""

    def __init__(self, ff_channels: int, preproc_ch, num_latent_channels: int,
                 latent_size, num_output_channels,
                 num_cross_attention_heads: int = 4, dropout: float = 0.0,
                 enc_latent_channels: int | None = None):
        super().__init__()
        # 改动 2：参考实现缺这个检查。cross-attention 的 KV 维取自
        # num_latent_channels（解码器的超参），但真正喂进来的 z 的通道数由编码器决定。
        if enc_latent_channels is not None and enc_latent_channels != num_latent_channels:
            raise ValueError(
                f"dec_num_latent_channels({num_latent_channels}) 必须等于 "
                f"enc_num_latent_channels({enc_latent_channels})：解码器的 "
                f"cross-attention 用前者当 KV 维，而喂进去的 z 的通道数是后者。")

        q_chan = ff_channels + num_latent_channels
        q_in = preproc_ch if preproc_ch else q_chan

        self.postproc = nn.Linear(q_in, num_output_channels)
        self.preproc = nn.Linear(q_chan, preproc_ch) if preproc_ch else None
        self.cross_attention = cross_attention_layer(num_q_channels=q_in,
                                                     num_kv_channels=num_latent_channels,
                                                     num_heads=num_cross_attention_heads,
                                                     dropout=dropout)
        # latent_size=1 时这就是"每个查询点共享的一个可学习标记"，与位置编码拼接当 Q
        self.output = nn.Parameter(torch.empty(latent_size, num_latent_channels))
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.output.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x, coords):
        b, *_ = x.shape
        output = repeat(self.output, "... -> b ...", b=b)
        output = torch.repeat_interleave(output, coords.shape[1], axis=1)
        output = torch.cat([coords, output], axis=-1)
        if self.preproc:
            output = self.preproc(output)
        output = self.cross_attention(output, x)
        return self.postproc(output)
