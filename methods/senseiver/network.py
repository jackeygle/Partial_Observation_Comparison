"""
network.py — the whole Senseiver model
============================

Corresponds to the reference implementation's `network_light.py`, but rewritten
as a **plain PyTorch nn.Module**, without PyTorch-Lightning. Two reasons:
  * The reference implementation's Lightning wrapper carries a bug -- `train.py`'s
    `Trainer()` never passes `accelerator`/`devices` at all, so the `gpu_device`
    that `s_parser.py` painstakingly computes is only used in the `--test`
    branch, meaning a GPU index given on the command line has no effect during
    training;
  * We need to self-chain and resume on SLURM, and writing our own loop is
    simpler than working around Lightning's checkpoint conventions.

The forward pass exactly matches the reference implementation:

    z = encoder(sensor_tokens, pad_mask)          # the paper's z = E(a_s, s)
    s_hat = decoder(z, query_pos_encodings)       # the paper's s_hat_q = D(z, a_q)

The positional encoding and the input-standardisation statistics are both
registered as buffers, so they travel with the checkpoint -- the evaluation
script never has to re-derive anything to reproduce the training-time input
convention.
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
            enc_latent_channels=enc_num_latent_channels)     # change 2's assertion takes effect here

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
        """Queries the entire grid, returns (B, C, H, W)."""
        b = tokens.shape[0]
        coords = self.pos_enc[None].expand(b, -1, -1)
        out = self.forward(tokens, pad_mask, coords)          # (B, HW, C)
        return out.permute(0, 2, 1).reshape(b, self.im_ch, *self.grid)
