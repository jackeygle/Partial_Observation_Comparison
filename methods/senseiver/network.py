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

Two switches select variant G (an extension, not the reference implementation --
see model.GridEncoder): `latent_mode='grid'` replaces the 64 abstract latents with
one latent token per grid cell, and `readout='direct'` replaces the query decoder
with a linear head on each cell's own token. Both default to the reference
behaviour, and old checkpoints (whose hparams lack these keys) load as variant A.

`time_window=k > 1` selects the temporal extension (also not the reference): each
sensor token also carries its relative time offset Δ (positional.TemporalEncoding),
concatenated before the encoder's preproc, and callers pass `dt` alongside the
tokens (sensors.build_batch_temporal). The default k=1 adds nothing, so every
earlier checkpoint loads and computes exactly as before.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from methods.senseiver.model import Decoder, Encoder, GridEncoder
from methods.senseiver.positional import PositionalEncoder, TemporalEncoding, encoding_channels


class Senseiver(nn.Module):
    def __init__(self, *, im_ch, grid, space_bands,
                 enc_preproc_ch, num_latents, enc_num_latent_channels, num_layers,
                 num_cross_attention_heads, enc_num_self_attention_heads,
                 num_self_attention_layers_per_block,
                 dec_preproc_ch, dec_num_latent_channels, dec_num_cross_attention_heads,
                 dropout=0.0, latent_size=1,
                 latent_mode="abstract", readout="decoder",
                 time_window=1, time_dim=8, time_scalar=True,
                 share_encoder_blocks=True,
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
            dropout=dropout, latent_size=latent_size,
            latent_mode=latent_mode, readout=readout,
            time_window=time_window, time_dim=time_dim, time_scalar=time_scalar,
            share_encoder_blocks=share_encoder_blocks)

        self.im_ch = im_ch
        self.grid = tuple(grid)
        self.space_bands = space_bands
        pos_ch = encoding_channels(len(self.grid), space_bands)

        if latent_mode not in ("abstract", "grid"):
            raise ValueError(f"latent_mode must be 'abstract' or 'grid', got {latent_mode!r}")
        if readout not in ("decoder", "direct"):
            raise ValueError(f"readout must be 'decoder' or 'direct', got {readout!r}")
        if readout == "direct" and latent_mode != "grid":
            raise ValueError("readout='direct' reads each cell's value off that cell's own "
                             "latent token, so it needs grid latents -- "
                             "abstract latents have no cell to read")
        self.latent_mode, self.readout_mode = latent_mode, readout

        if time_window < 1:
            raise ValueError(f"time_window must be >= 1, got {time_window}")
        self.time_window = time_window
        time_ch = 0                                   # k=1: nothing added, the model is unchanged
        if time_window > 1:
            self.time_enc = TemporalEncoding(time_window, time_dim, time_scalar)
            time_ch = self.time_enc.channels

        pe = PositionalEncoder((*self.grid, im_ch), space_bands).float()
        if latent_mode == "abstract":
            self.encoder = Encoder(
                input_ch=im_ch + pos_ch + time_ch,
                preproc_ch=enc_preproc_ch,
                num_latents=num_latents,
                num_latent_channels=enc_num_latent_channels,
                num_layers=num_layers,
                num_cross_attention_heads=num_cross_attention_heads,
                num_self_attention_heads=enc_num_self_attention_heads,
                num_self_attention_layers_per_block=num_self_attention_layers_per_block,
                dropout=dropout, share_blocks=share_encoder_blocks)
        else:
            self.encoder = GridEncoder(
                input_ch=im_ch + pos_ch + time_ch,
                preproc_ch=enc_preproc_ch,
                pos_ch=pos_ch,
                num_latent_channels=enc_num_latent_channels,
                num_layers=num_layers,
                num_cross_attention_heads=num_cross_attention_heads,
                num_self_attention_heads=enc_num_self_attention_heads,
                num_self_attention_layers_per_block=num_self_attention_layers_per_block,
                dropout=dropout, share_blocks=share_encoder_blocks)

        if readout == "decoder":
            self.decoder = Decoder(
                ff_channels=pos_ch,
                preproc_ch=dec_preproc_ch,
                num_latent_channels=dec_num_latent_channels,
                latent_size=latent_size,
                num_output_channels=im_ch,
                num_cross_attention_heads=dec_num_cross_attention_heads,
                dropout=dropout,
                enc_latent_channels=enc_num_latent_channels)     # change 2's assertion takes effect here
        else:
            # one linear head shared by every cell: that cell's token -> its C field values
            self.head = nn.Linear(enc_num_latent_channels, im_ch)

        self.register_buffer("pos_enc", pe, persistent=True)
        m = np.ones(im_ch, np.float32) * 0.0 if in_mean is None else np.asarray(in_mean, np.float32)
        s = np.ones(im_ch, np.float32) if in_std is None else np.asarray(in_std, np.float32)
        self.register_buffer("in_mean", torch.from_numpy(m), persistent=True)
        self.register_buffer("in_std", torch.from_numpy(s), persistent=True)

        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.num_params = int(n)

    # ------------------------------------------------------------------ #
    def forward(self, tokens, pad_mask, coords, dt=None, cell_idx=None):
        """tokens (B,Ns,C+P) / pad_mask (B,Ns) bool / coords (B,Nq,P) / dt (B,Ns) long -> (B,Nq,C)

        `dt`, each token's relative time offset, is required when time_window > 1 and
        must be None otherwise.

        readout='direct' has no query mechanism -- it returns one value per grid
        cell in pos_enc order, so `coords` must be the full grid (reconstruct()
        passes exactly that)."""
        if self.time_window > 1:
            if dt is None:
                raise ValueError(f"time_window={self.time_window}: pass dt, each token's "
                                 f"relative time offset")
            tokens = torch.cat([tokens, self.time_enc(dt).to(tokens.dtype)], dim=-1)
        elif dt is not None:
            raise ValueError("this model has no time window (time_window=1); dt must be None")
        if self.latent_mode == "abstract":
            z = self.encoder(tokens, pad_mask)
        else:
            z = self.encoder(tokens, self.pos_enc, pad_mask)
        if self.readout_mode == "decoder":
            return self.decoder(z, coords)
        if coords is not None and coords.shape[1] != z.shape[1]:
            raise ValueError(f"readout='direct' can only answer the full grid "
                             f"({z.shape[1]} cells), got {coords.shape[1]} query points")
        return self.head(z)

    def reconstruct(self, tokens, pad_mask, dt=None, cell_idx=None):
        """Queries the entire grid, returns (B, C, H, W)."""
        b = tokens.shape[0]
        coords = self.pos_enc[None].expand(b, -1, -1)
        out = self.forward(tokens, pad_mask, coords, dt, cell_idx)  # (B, HW, C)
        return out.permute(0, 2, 1).reshape(b, self.im_ch, *self.grid)
