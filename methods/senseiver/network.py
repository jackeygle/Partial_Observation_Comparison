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

`latent_mode='hierarchical'` is the H3 capacity experiment: three independent
grid-latent blocks at 1/4, 1/2 and full resolution, with bilinear coarse-to-fine
feature transfer and direct readout at full resolution.

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

from methods.senseiver.model import (Decoder, Encoder, FramewiseTemporalGridEncoder,
                                     GridEncoder, HierarchicalGridEncoder)
from methods.senseiver.positional import PositionalEncoder, TemporalEncoding, encoding_channels


class TemporalCellSSM(nn.Module):
    """A small diagonal SSM scanned independently at every spatial cell.

    Sparse observations are first placed back on their cells.  The state is
    evolved from the oldest frame to the query frame and collapsed to one token
    per cell, so downstream attention performs spatial -- not temporal -- mixing.
    """

    def __init__(self, value_ch, hidden_ch, window):
        super().__init__()
        self.value_ch, self.hidden_ch, self.window = value_ch, hidden_ch, window
        self.in_proj = nn.Linear(value_ch, hidden_ch)
        self.out_proj = nn.Linear(hidden_ch, value_ch)
        # sigmoid(logit_a) is a stable diagonal transition in (0, 1).
        self.logit_a = nn.Parameter(torch.full((hidden_ch,), 2.0))
        self.log_input_scale = nn.Parameter(torch.zeros(()))
        self.log_output_scale = nn.Parameter(torch.zeros(()))

    def forward(self, tokens, pad_mask, dt, cell_idx, pos_enc):
        if cell_idx is None:
            raise ValueError("temporal_mixer='ssm' requires token cell indices")
        b, _, _ = tokens.shape
        hw = pos_enc.shape[0]
        values = tokens[..., :self.value_ch]
        valid = ~pad_mask
        state = values.new_zeros(b, hw, self.hidden_ch)
        last = values.new_zeros(b, hw, self.value_ch)
        seen = torch.zeros(b, hw, dtype=torch.bool, device=tokens.device)
        decay = torch.sigmoid(self.logit_a).to(values.dtype)

        # dt=window-1 is oldest and dt=0 is the target frame.
        for offset in range(self.window - 1, -1, -1):
            use = valid & (dt == offset)
            dense = values.new_zeros(b, hw, self.value_ch)
            if use.any():
                bi, ni = use.nonzero(as_tuple=True)
                ci = cell_idx[bi, ni]
                dense[bi, ci] = values[bi, ni]
                seen[bi, ci] = True
                last[bi, ci] = values[bi, ni]
            state = state * decay + self.log_input_scale.exp() * self.in_proj(dense)

        out_values = self.log_output_scale.exp() * self.out_proj(state) + last
        out = torch.cat([
            out_values,
            pos_enc.to(tokens.dtype)[None].expand(b, -1, -1),
        ], dim=-1)
        out_pad = ~seen
        # MultiheadAttention cannot accept a sample whose every key is masked.
        empty = ~seen.any(dim=1)
        if empty.any():
            out_pad[empty, 0] = False
        out_cell = torch.arange(hw, device=tokens.device)[None].expand(b, -1)
        return out, out_pad, out_cell


class GeodesicAttentionBias(nn.Module):
    """Fixed soft bias from obstacle-aware shortest-path distance."""

    def __init__(self, grid, num_heads, scale=1.0):
        super().__init__()
        if len(grid) != 2:
            raise ValueError("geodesic attention currently requires a 2-D grid")
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra
        from crowdcore import navigation as nav

        h, w = grid
        walkable = nav.build_valid_mask_from_config()
        if walkable.shape != (h, w):
            raise ValueError(f"walkable mask {walkable.shape} != model grid {(h, w)}")
        n = h * w
        graph = np.zeros((n, n), dtype=np.float32)
        steps = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                 (-1, -1, 2 ** 0.5), (-1, 1, 2 ** 0.5),
                 (1, -1, 2 ** 0.5), (1, 1, 2 ** 0.5)]
        for r in range(h):
            for c in range(w):
                if not walkable[r, c]:
                    continue
                a = r * w + c
                for dr, dc, cost in steps:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and walkable[rr, cc]:
                        graph[a, rr * w + cc] = cost
        dist = dijkstra(csr_matrix(graph), directed=False).astype(np.float32)
        finite = np.isfinite(dist)
        max_distance = float(dist[finite].max()) if finite.any() else 1.0
        dist = np.where(finite, dist / max(max_distance, 1.0), 1.0)
        self.register_buffer("distance", torch.from_numpy(dist), persistent=True)
        self.num_heads = num_heads
        self.scale = float(scale)

    def forward(self, source_cell, dtype):
        # distance is [query cell, source cell].  MHA expects [B*heads,Q,S].
        b, ns = source_cell.shape
        q = self.distance.shape[0]
        d = self.distance[None].expand(b, -1, -1).gather(
            2, source_cell[:, None, :].expand(-1, q, -1))
        bias = -self.scale * d.to(dtype)[:, None].expand(-1, self.num_heads, -1, -1)
        return bias.reshape(b * self.num_heads, q, ns)


class Senseiver(nn.Module):
    def __init__(self, *, im_ch, grid, space_bands,
                 enc_preproc_ch, num_latents, enc_num_latent_channels, num_layers,
                 num_cross_attention_heads, enc_num_self_attention_heads,
                 num_self_attention_layers_per_block,
                 dec_preproc_ch, dec_num_latent_channels, dec_num_cross_attention_heads,
                 dropout=0.0, latent_size=1,
                 latent_mode="abstract", readout="decoder",
                 time_window=1, time_dim=8, time_scalar=True,
                 temporal_mixer="token_set", spatial_attention="global",
                 ssm_hidden_ch=41, geodesic_scale=1.0, advection_tau=3.0,
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
            temporal_mixer=temporal_mixer, spatial_attention=spatial_attention,
            ssm_hidden_ch=ssm_hidden_ch, geodesic_scale=geodesic_scale,
            advection_tau=advection_tau,
            share_encoder_blocks=share_encoder_blocks)

        self.im_ch = im_ch
        self.grid = tuple(grid)
        self.space_bands = space_bands
        pos_ch = encoding_channels(len(self.grid), space_bands)

        if latent_mode not in ("abstract", "grid", "hierarchical"):
            raise ValueError(f"latent_mode must be 'abstract', 'grid' or 'hierarchical', "
                             f"got {latent_mode!r}")
        if readout not in ("decoder", "direct"):
            raise ValueError(f"readout must be 'decoder' or 'direct', got {readout!r}")
        if readout == "direct" and latent_mode not in ("grid", "hierarchical"):
            raise ValueError("readout='direct' reads each cell's value off that cell's own "
                             "latent token, so it needs a grid-based latent mode -- "
                             "abstract latents have no cell to read")
        if latent_mode == "hierarchical" and readout != "direct":
            raise ValueError("latent_mode='hierarchical' is a coarse-to-fine direct-readout "
                             "extension; use readout='direct'")
        if latent_mode == "hierarchical" and share_encoder_blocks:
            raise ValueError("hierarchical scales need independent blocks; set "
                             "share_encoder_blocks=False")
        self.latent_mode, self.readout_mode = latent_mode, readout

        if time_window < 1:
            raise ValueError(f"time_window must be >= 1, got {time_window}")
        if temporal_mixer not in ("token_set", "advected_tokens", "ssm", "framewise"):
            raise ValueError(f"unknown temporal_mixer={temporal_mixer!r}")
        if spatial_attention not in ("global", "geodesic"):
            raise ValueError(f"unknown spatial_attention={spatial_attention!r}")
        if temporal_mixer in ("advected_tokens", "ssm", "framewise") and time_window == 1:
            raise ValueError(f"temporal_mixer={temporal_mixer!r} needs time_window > 1")
        if (temporal_mixer in ("ssm", "framewise") or spatial_attention == "geodesic") and latent_mode != "grid":
            raise ValueError("SSM/framewise/geodesic extensions require grid latents")
        if temporal_mixer == "framewise" and spatial_attention != "global":
            raise ValueError("framewise temporal fusion currently uses global spatial attention")
        self.time_window = time_window
        self.temporal_mixer = temporal_mixer
        self.spatial_attention = spatial_attention
        if advection_tau <= 0:
            raise ValueError(f"advection_tau must be positive, got {advection_tau}")
        self.advection_tau = float(advection_tau)
        time_ch = 0                                   # k=1: nothing added, the model is unchanged
        if time_window > 1 and temporal_mixer in ("token_set", "advected_tokens"):
            self.time_enc = TemporalEncoding(time_window, time_dim, time_scalar)
            time_ch = self.time_enc.channels
        elif temporal_mixer == "ssm":
            self.temporal_ssm = TemporalCellSSM(im_ch, ssm_hidden_ch, time_window)

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
        elif latent_mode == "grid" and temporal_mixer == "framewise":
            self.encoder = FramewiseTemporalGridEncoder(
                input_ch=im_ch + pos_ch,
                preproc_ch=enc_preproc_ch, pos_ch=pos_ch,
                num_latent_channels=enc_num_latent_channels,
                window=time_window, num_layers=num_layers,
                num_cross_attention_heads=num_cross_attention_heads,
                num_self_attention_heads=enc_num_self_attention_heads,
                num_self_attention_layers_per_block=num_self_attention_layers_per_block,
                dropout=dropout, share_blocks=share_encoder_blocks)
        elif latent_mode == "grid":
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
        else:
            if len(self.grid) != 2 or any(s % 4 for s in self.grid):
                raise ValueError("hierarchical mode needs a 2-D grid divisible by 4, "
                                 f"got {self.grid}")
            if num_layers != 3:
                raise ValueError("hierarchical mode maps its three encoder blocks to three "
                                 f"spatial scales, so num_layers must be 3, got {num_layers}")
            h, w = self.grid
            self.hier_shapes = ((h // 4, w // 4), (h // 2, w // 2), (h, w))
            self.encoder = HierarchicalGridEncoder(
                input_ch=im_ch + pos_ch + time_ch,
                preproc_ch=enc_preproc_ch,
                pos_ch=pos_ch,
                num_latent_channels=enc_num_latent_channels,
                scale_shapes=self.hier_shapes,
                num_cross_attention_heads=num_cross_attention_heads,
                num_self_attention_heads=enc_num_self_attention_heads,
                num_self_attention_layers_per_block=num_self_attention_layers_per_block,
                dropout=dropout)
            coarse = PositionalEncoder((*self.hier_shapes[0], im_ch), space_bands,
                                       max_frequencies=self.grid).float()
            middle = PositionalEncoder((*self.hier_shapes[1], im_ch), space_bands,
                                       max_frequencies=self.grid).float()
            self.register_buffer("hier_pos_coarse", coarse, persistent=True)
            self.register_buffer("hier_pos_middle", middle, persistent=True)

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

        if spatial_attention == "geodesic":
            self.geodesic_bias = GeodesicAttentionBias(
                grid, num_cross_attention_heads, geodesic_scale)

        self.register_buffer("pos_enc", pe, persistent=True)
        m = np.ones(im_ch, np.float32) * 0.0 if in_mean is None else np.asarray(in_mean, np.float32)
        s = np.ones(im_ch, np.float32) if in_std is None else np.asarray(in_std, np.float32)
        self.register_buffer("in_mean", torch.from_numpy(m), persistent=True)
        self.register_buffer("in_std", torch.from_numpy(s), persistent=True)

        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.num_params = int(n)

    # ------------------------------------------------------------------ #
    def _advect_token_positions(self, tokens, dt, cell_idx):
        """Move historical token coordinates to their velocity-extrapolated location.

        Velocity is stored in local grid metres/second, the grid resolution and
        frame interval are both one, and dt is the number of seconds before the
        query frame.  Only positional channels change; values and time encoding
        remain exactly the baseline token-set representation.
        """
        if cell_idx is None:
            raise ValueError("temporal_mixer='advected_tokens' requires token cell indices")
        h, w = self.grid
        physical = tokens[..., :self.im_ch] * self.in_std + self.in_mean
        row = torch.div(cell_idx, w, rounding_mode="floor").to(tokens.dtype)
        col = (cell_idx % w).to(tokens.dtype)
        age = dt.to(tokens.dtype)
        # Constant velocity is highly predictive at one second but degrades at
        # long horizons.  Decay the ballistic travel distance rather than
        # extrapolating a noisy instantaneous velocity for all 15 seconds.
        travel = age * torch.exp(-age / self.advection_tau)
        row = (row + travel * physical[..., 1]).clamp(0, h - 1)
        col = (col + travel * physical[..., 2]).clamp(0, w - 1)

        row = 2.0 * row / max(h - 1, 1) - 1.0
        col = 2.0 * col / max(w - 1, 1) - 1.0
        fr = torch.linspace(1.0, h / 2.0, self.space_bands,
                            device=tokens.device, dtype=tokens.dtype)
        fc = torch.linspace(1.0, w / 2.0, self.space_bands,
                            device=tokens.device, dtype=tokens.dtype)
        pi = torch.as_tensor(np.pi, device=tokens.device, dtype=tokens.dtype)
        gr, gc = row[..., None] * fr, col[..., None] * fc
        advected_pe = torch.cat([
            torch.sin(pi * gr), torch.sin(pi * gc),
            torch.cos(pi * gr), torch.cos(pi * gc),
        ], dim=-1)
        return torch.cat([tokens[..., :self.im_ch], advected_pe], dim=-1)

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
            if self.temporal_mixer == "advected_tokens":
                tokens = self._advect_token_positions(tokens, dt, cell_idx)
                tokens = torch.cat([tokens, self.time_enc(dt).to(tokens.dtype)], dim=-1)
            elif self.temporal_mixer == "token_set":
                tokens = torch.cat([tokens, self.time_enc(dt).to(tokens.dtype)], dim=-1)
            elif self.temporal_mixer == "ssm":
                tokens, pad_mask, cell_idx = self.temporal_ssm(
                    tokens, pad_mask, dt, cell_idx, self.pos_enc)
        elif dt is not None:
            raise ValueError("this model has no time window (time_window=1); dt must be None")
        cross_attn_bias = None
        if self.spatial_attention == "geodesic":
            if cell_idx is None:
                raise ValueError("spatial_attention='geodesic' requires token cell indices")
            cross_attn_bias = self.geodesic_bias(cell_idx, tokens.dtype)
        if self.latent_mode == "abstract":
            z = (self.encoder(tokens, pad_mask) if cross_attn_bias is None else
                 self.encoder(tokens, pad_mask, cross_attn_bias))
        elif self.latent_mode == "grid":
            if self.temporal_mixer == "framewise":
                z = self.encoder(tokens, self.pos_enc, pad_mask, dt)
            else:
                z = (self.encoder(tokens, self.pos_enc, pad_mask)
                     if cross_attn_bias is None else
                     self.encoder(tokens, self.pos_enc, pad_mask, cross_attn_bias))
        else:
            z = self.encoder(tokens, (self.hier_pos_coarse, self.hier_pos_middle,
                                      self.pos_enc), pad_mask)
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
