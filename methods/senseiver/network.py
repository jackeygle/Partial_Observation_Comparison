"""
network.py — Senseiver 整机
============================

对应参考实现的 `network_light.py`，但改成**纯 PyTorch 的 nn.Module**，不用
PyTorch-Lightning。理由有二：
  * 参考实现的 Lightning 封装带着一个 bug —— `train.py` 的 `Trainer()` 根本没传
    `accelerator`/`devices`，`s_parser.py` 辛苦算出来的 `gpu_device` 只在 `--test`
    分支用得上，所以命令行指定卡号在训练时是无效的；
  * 我们要在 SLURM 上自链接续训，自己写循环比迁就 Lightning 的 ckpt 约定更省事。

前向与参考实现完全一致：

    z = encoder(sensor_tokens, pad_mask)          # 论文 z = E(a_s, s)
    ŝ = decoder(z, query_pos_encodings)           # 论文 ŝ_q = D(z, a_q)

位置编码和输入标准化统计量都注册成 buffer，跟着 checkpoint 走，这样评估脚本
不需要重新推导任何东西就能复现训练时的输入约定。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from methods.senseiver.model import Decoder, Encoder
from methods.senseiver.positional import PositionalEncoder, encoding_channels


class Senseiver(nn.Module):
    def __init__(self, *, im_ch, grid, space_bands,
                 enc_preproc_ch, num_latents, enc_num_latent_channels, num_layers,
                 num_cross_attention_heads, enc_num_self_attention_heads,
                 num_self_attention_layers_per_block,
                 dec_preproc_ch, dec_num_latent_channels, dec_num_cross_attention_heads,
                 dropout=0.0, latent_size=1,
                 in_mean=None, in_std=None):
        super().__init__()
        self.hparams = dict(
            im_ch=im_ch, grid=list(grid), space_bands=space_bands,
            enc_preproc_ch=enc_preproc_ch, num_latents=num_latents,
            enc_num_latent_channels=enc_num_latent_channels, num_layers=num_layers,
            num_cross_attention_heads=num_cross_attention_heads,
            enc_num_self_attention_heads=enc_num_self_attention_heads,
            num_self_attention_layers_per_block=num_self_attention_layers_per_block,
            dec_preproc_ch=dec_preproc_ch,
            dec_num_latent_channels=dec_num_latent_channels,
            dec_num_cross_attention_heads=dec_num_cross_attention_heads,
            dropout=dropout, latent_size=latent_size)

        self.im_ch = im_ch
        self.grid = tuple(grid)
        pos_ch = encoding_channels(len(self.grid), space_bands)

        self.encoder = Encoder(
            input_ch=im_ch + pos_ch,
            preproc_ch=enc_preproc_ch,
            num_latents=num_latents,
            num_latent_channels=enc_num_latent_channels,
            num_layers=num_layers,
            num_cross_attention_heads=num_cross_attention_heads,
            num_self_attention_heads=enc_num_self_attention_heads,
            num_self_attention_layers_per_block=num_self_attention_layers_per_block,
            dropout=dropout)

        self.decoder = Decoder(
            ff_channels=pos_ch,
            preproc_ch=dec_preproc_ch,
            num_latent_channels=dec_num_latent_channels,
            latent_size=latent_size,
            num_output_channels=im_ch,
            num_cross_attention_heads=dec_num_cross_attention_heads,
            dropout=dropout,
            enc_latent_channels=enc_num_latent_channels)     # 改动 2 的断言在这里生效

        pe = PositionalEncoder((*self.grid, im_ch), space_bands).float()
        self.register_buffer("pos_enc", pe, persistent=True)
        m = np.ones(im_ch, np.float32) * 0.0 if in_mean is None else np.asarray(in_mean, np.float32)
        s = np.ones(im_ch, np.float32) if in_std is None else np.asarray(in_std, np.float32)
        self.register_buffer("in_mean", torch.from_numpy(m), persistent=True)
        self.register_buffer("in_std", torch.from_numpy(s), persistent=True)

        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.num_params = int(n)

    # ------------------------------------------------------------------ #
    def forward(self, tokens, pad_mask, coords):
        """tokens (B,Ns,C+P) / pad_mask (B,Ns) bool / coords (B,Nq,P) -> (B,Nq,C)"""
        z = self.encoder(tokens, pad_mask)
        return self.decoder(z, coords)

    def reconstruct(self, tokens, pad_mask):
        """查询整张网格，返回 (B, C, H, W)。"""
        b = tokens.shape[0]
        coords = self.pos_enc[None].expand(b, -1, -1)
        out = self.forward(tokens, pad_mask, coords)          # (B, HW, C)
        return out.permute(0, 2, 1).reshape(b, self.im_ch, *self.grid)
