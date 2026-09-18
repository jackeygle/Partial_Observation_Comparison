"""
model.py — Senseiver's Encoder / Decoder
==========================================

Ported from `reference/Senseiver/model.py` (its own README says it is adapted
from krasserm/perceiver-io). Corresponds to the paper's sec.2 last two
components:

    z    = E(a_s, s)      attention encoder: sensor set -> fixed-length latent
    s_q  = D(z, a_q)      attention decoder: latent + query position -> the field value there

Where Appendix A's architecture lands in the code:
  * The encoder block = cross-attention (latent as Q, sensors as K/V) +
    self-attention, with `num_layers` blocks **sharing weights** (`layer_1` is
    independent + `layer_n` is reused num_layers-1 times, this is the reference
    implementation's specific approach).
  * Decoding = the query's positional encoding concatenated with a learnable
    vector as Q, z as K/V, a single cross-attention layer followed by a linear
    output head.

Changes relative to the reference implementation (the README's "Deviations from
the reference implementation" section has the same list):

 1. **Removed fairscale's `checkpoint_wrapper`**. In the reference
    implementation `activation_checkpoint` is never exposed by `s_parser.py`,
    is always False, and is dead code; removing it drops one dependency.
 2. **Added a dimension assertion to `Decoder`**. The decoder's cross-attention
    treats `num_latent_channels` as the KV dimension, while the actual channel
    count of the z fed in is decided by the **encoder**; a mismatch between the
    two silently misaligns. The reference implementation has no such check --
    its README's examples happen to always set the two values equal, hiding
    this trap.
 3. **`pad_mask` genuinely wired up**. This path already exists in the
    reference implementation (`Encoder.forward(x, pad_mask)` ->
    `CrossAttention` -> `key_padding_mask`), it is just that its dataloader
    never passes a value -- because its sensor set is fixed-length. Our sensors
    are moving robots, with a different count each frame, requiring padding, so
    this path is used for the first time here. No structural change was made,
    just enabling an existing capability and adding a guard for the empty-set case.

`GridEncoder` at the bottom is **not** part of the reference implementation: it is
variant G, an extension that replaces the abstract latent array with the grid
itself. `Encoder` and `Decoder` above are untouched by it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat


class Sequential(nn.Sequential):
    """Allows multiple arguments to pass between layers (the first layer takes
    a tuple, later ones take a single tensor)."""

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
        if (attn_mask is not None and attn_mask.is_floating_point()
                and pad_mask is not None and pad_mask.dtype == torch.bool):
            float_pad = torch.zeros_like(pad_mask, dtype=attn_mask.dtype)
            pad_mask = float_pad.masked_fill(pad_mask, float("-inf"))
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
    """z = E(PE(chi_s), s): compresses any number of sensor tokens into a
    fixed-length (num_latents, ch) latent."""

    def __init__(self, input_ch, preproc_ch, num_latents: int, num_latent_channels: int,
                 num_layers: int = 3, num_cross_attention_heads: int = 4,
                 num_self_attention_heads: int = 4,
                 num_self_attention_layers_per_block: int = 6, dropout: float = 0.0,
                 share_blocks: bool = True):
        super().__init__()
        self.num_layers = num_layers
        self.share_blocks = share_blocks
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

        if share_blocks:
            self.layer_1 = create_layer()
            if num_layers > 1:
                self.layer_n = create_layer()      # reference: later blocks reuse this same layer_n
        else:
            self.layers = nn.ModuleList(create_layer() for _ in range(num_layers))
        self.latent = nn.Parameter(torch.empty(num_latents, num_latent_channels))
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x, pad_mask=None, cross_attn_bias=None):
        b, *_ = x.shape
        if self.preproc:
            x = self.preproc(x)
        x_latent = repeat(self.latent, "... -> b ...", b=b)
        if self.share_blocks:
            x_latent = (self.layer_1(x_latent, x, pad_mask)
                        if cross_attn_bias is None else
                        self.layer_1(x_latent, x, pad_mask, cross_attn_bias))
            for _ in range(self.num_layers - 1):
                x_latent = (self.layer_n(x_latent, x, pad_mask)
                            if cross_attn_bias is None else
                            self.layer_n(x_latent, x, pad_mask, cross_attn_bias))
        else:
            for layer in self.layers:
                x_latent = (layer(x_latent, x, pad_mask)
                            if cross_attn_bias is None else
                            layer(x_latent, x, pad_mask, cross_attn_bias))
        return x_latent


class Decoder(nn.Module):
    """s_q = D(z, PE(chi_q)): queries the field value at any coordinate."""

    def __init__(self, ff_channels: int, preproc_ch, num_latent_channels: int,
                 latent_size, num_output_channels,
                 num_cross_attention_heads: int = 4, dropout: float = 0.0,
                 enc_latent_channels: int | None = None):
        super().__init__()
        # Change 2: the reference implementation is missing this check. The
        # cross-attention's KV dimension is taken from num_latent_channels
        # (the decoder's hyperparameter), but the actual channel count of the
        # z fed in is decided by the encoder.
        if enc_latent_channels is not None and enc_latent_channels != num_latent_channels:
            raise ValueError(
                f"dec_num_latent_channels({num_latent_channels}) must equal "
                f"enc_num_latent_channels({enc_latent_channels}): the decoder's "
                f"cross-attention uses the former as the KV dimension, while "
                f"the z fed in has the channel count of the latter.")

        q_chan = ff_channels + num_latent_channels
        q_in = preproc_ch if preproc_ch else q_chan

        self.postproc = nn.Linear(q_in, num_output_channels)
        self.preproc = nn.Linear(q_chan, preproc_ch) if preproc_ch else None
        self.cross_attention = cross_attention_layer(num_q_channels=q_in,
                                                     num_kv_channels=num_latent_channels,
                                                     num_heads=num_cross_attention_heads,
                                                     dropout=dropout)
        # When latent_size=1 this is "one learnable token shared across every
        # query point," concatenated with the positional encoding to form Q
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


class GridEncoder(Encoder):
    """Variant G (not part of the reference implementation): the latent array IS the grid.

    `Encoder` keeps a free parameter `latent` of shape (num_latents, ch) -- 64
    abstract vectors with no physical meaning. Our state is tiny (4 x 36 x 12 =
    1,728 numbers, fewer than those 64 x 32 = 2,048 latent numbers), so there is
    nothing to compress; the latent's only real job here is to turn a
    variable-length, unordered sensor set into something fixed-length. The grid
    itself is fixed-length, so this variant uses one latent token per cell.

    Each cell's initial token is its spatial positional encoding through one
    linear map, **not** a free per-cell vector. A free vector per cell (432 x 32
    parameters) could simply memorise that cell's mean field -- the sensor-value
    ablation already showed climatology carries most of the blind-zone skill --
    which would confound "grid structure helps" with "a per-cell bias helps". The
    projection keeps the parameter count within 32 of `Encoder` (2,080 vs 2,048).

    Everything else -- preproc, the cross-attention + self-attention block and
    its weight sharing (`layer_1` + `layer_n` reused) -- is inherited unchanged.
    """

    def __init__(self, input_ch, preproc_ch, pos_ch: int, num_latent_channels: int, **kw):
        super().__init__(input_ch, preproc_ch, num_latents=1,
                         num_latent_channels=num_latent_channels, **kw)
        del self.latent                                   # no abstract latent in this variant
        self.latent_proj = nn.Linear(pos_ch, num_latent_channels)

    def _init_parameters(self):
        pass                                              # nn.Linear initialises itself

    def forward(self, x, cell_pos, pad_mask=None, cross_attn_bias=None):
        """x (B,Ns,input_ch) / cell_pos (HW,P) -> one token per cell, (B,HW,ch)."""
        b, *_ = x.shape
        if self.preproc:
            x = self.preproc(x)
        x_latent = repeat(self.latent_proj(cell_pos), "... -> b ...", b=b)
        if self.share_blocks:
            x_latent = (self.layer_1(x_latent, x, pad_mask)
                        if cross_attn_bias is None else
                        self.layer_1(x_latent, x, pad_mask, cross_attn_bias))
            for _ in range(self.num_layers - 1):
                x_latent = (self.layer_n(x_latent, x, pad_mask)
                            if cross_attn_bias is None else
                            self.layer_n(x_latent, x, pad_mask, cross_attn_bias))
        else:
            for layer in self.layers:
                x_latent = (layer(x_latent, x, pad_mask)
                            if cross_attn_bias is None else
                            layer(x_latent, x, pad_mask, cross_attn_bias))
        return x_latent


class FramewiseTemporalGridEncoder(nn.Module):
    """Encode each sparse frame separately, then learn a residual temporal update.

    Unlike the flattened token-set baseline, observations from different frames
    do not compete in one cross-attention softmax.  A shared spatial
    cross-attention first produces one dense grid latent per frame.  For each
    grid cell, the current latent then attends over its own sequence of frame
    latents.  No coordinate is moved and no motion equation is supplied.
    """

    def __init__(self, input_ch, preproc_ch: int, pos_ch: int,
                 num_latent_channels: int, window: int, num_layers: int = 3,
                 num_cross_attention_heads: int = 2,
                 num_self_attention_heads: int = 2,
                 num_self_attention_layers_per_block: int = 3,
                 dropout: float = 0.0, share_blocks: bool = True):
        super().__init__()
        if window < 2:
            raise ValueError("framewise temporal encoder needs window >= 2")
        self.window = window
        self.num_layers = num_layers
        self.share_blocks = share_blocks
        self.preproc = nn.Linear(input_ch, preproc_ch) if preproc_ch else None
        kv_ch = preproc_ch or input_ch
        self.latent_proj = nn.Linear(pos_ch, num_latent_channels)

        # Shared by every frame: parameters do not scale with the window.
        self.frame_cross = cross_attention_layer(
            num_q_channels=num_latent_channels, num_kv_channels=kv_ch,
            num_heads=num_cross_attention_heads, dropout=dropout)
        self.time_embedding = nn.Embedding(window, num_latent_channels)
        self.temporal_fuse = cross_attention_layer(
            num_q_channels=num_latent_channels,
            num_kv_channels=num_latent_channels,
            num_heads=num_cross_attention_heads, dropout=dropout)
        # Start close to current-frame-only behaviour; learning decides how
        # strongly each latent channel should accept the historical update.
        self.history_gate_logit = nn.Parameter(
            torch.full((num_latent_channels,), -4.0))

        def spatial_block():
            return self_attention_block(
                num_layers=num_self_attention_layers_per_block,
                num_channels=num_latent_channels,
                num_heads=num_self_attention_heads, dropout=dropout)

        if share_blocks:
            self.spatial_1 = spatial_block()
            if num_layers > 1:
                self.spatial_n = spatial_block()
        else:
            self.spatial_layers = nn.ModuleList(
                spatial_block() for _ in range(num_layers))

    def forward(self, x, cell_pos, pad_mask, dt):
        """x (B,N,D), dt (B,N); return current grid latent (B,HW,C)."""
        if dt is None:
            raise ValueError("framewise temporal encoder requires token time offsets")
        if self.preproc:
            x = self.preproc(x)
        b, _, dkv = x.shape
        base = repeat(self.latent_proj(cell_pos), "... -> b ...", b=b)
        frame_latents, frame_valid = [], []

        # build_batch_temporal concatenates all K frames into one padded token
        # array.  Passing that full array to every frame's attention and merely
        # masking K-1 frames is mathematically correct, but makes the attention
        # kernel scan the same irrelevant tokens K times.  Compact each frame
        # first, retaining original token order.  The appended dummy key keeps
        # all-empty historical frames valid for MultiheadAttention; their result
        # is replaced by ``base`` below exactly as in the un-compacted version.
        offsets = torch.arange(self.window, device=x.device)
        valid_by_frame = ((~pad_mask)[:, None, :]
                          & (dt[:, None, :] == offsets[None, :, None]))
        counts_by_frame = valid_by_frame.sum(dim=2)
        max_frame_tokens = int(counts_by_frame.max().item())
        compact_positions = valid_by_frame.cumsum(dim=2) - 1

        # Offset 0 is the current frame; increasing offsets go into the past.
        for offset in range(self.window):
            valid = valid_by_frame[:, offset]
            counts = counts_by_frame[:, offset]
            empty = counts == 0
            keys = x.new_zeros(b, max_frame_tokens + 1, dkv)
            batch_idx, token_idx = valid.nonzero(as_tuple=True)
            dst_idx = compact_positions[:, offset][batch_idx, token_idx]
            keys[batch_idx, dst_idx] = x[batch_idx, token_idx]
            masked = (torch.arange(max_frame_tokens, device=x.device)[None, :]
                      >= counts[:, None])
            dummy_mask = (~empty).unsqueeze(1)
            key_mask = torch.cat([masked, dummy_mask], dim=1)
            z = self.frame_cross(base, keys, key_mask)
            z = torch.where(empty[:, None, None], base, z)
            frame_latents.append(z)
            frame_valid.append(~empty)

        # (B,HW,K,C): each cell gets a short, explicitly ordered time sequence.
        seq = torch.stack(frame_latents, dim=2)
        seq = seq + self.time_embedding(offsets).to(seq.dtype)[None, None]
        hw, ch = seq.shape[1], seq.shape[3]
        seq_flat = seq.reshape(b * hw, self.window, ch)
        valid = torch.stack(frame_valid, dim=1)
        time_pad = (~valid)[:, None].expand(-1, hw, -1).reshape(
            b * hw, self.window)
        current = seq[:, :, 0].reshape(b * hw, 1, ch)
        fused = self.temporal_fuse(current, seq_flat, time_pad).reshape(b, hw, ch)
        current = current.reshape(b, hw, ch)
        gate = torch.sigmoid(self.history_gate_logit).to(fused.dtype)
        z = current + gate * (fused - current)

        if self.share_blocks:
            z = self.spatial_1(z)
            for _ in range(self.num_layers - 1):
                z = self.spatial_n(z)
        else:
            for layer in self.spatial_layers:
                z = layer(z)
        return z


class HierarchicalGridEncoder(nn.Module):
    """Three independent coarse-to-fine grid-latent blocks.

    Sensors are available to every scale.  The latent field is bilinearly
    prolonged between scales and combined with the next scale's positional
    query before that scale attends to the original sensor set.
    """

    def __init__(self, input_ch, preproc_ch, pos_ch: int, num_latent_channels: int,
                 scale_shapes, num_cross_attention_heads: int = 4,
                 num_self_attention_heads: int = 4,
                 num_self_attention_layers_per_block: int = 6,
                 dropout: float = 0.0):
        super().__init__()
        self.scale_shapes = tuple(tuple(s) for s in scale_shapes)
        if len(self.scale_shapes) != 3:
            raise ValueError(f"hierarchical encoder needs three scales, got {self.scale_shapes}")
        self.preproc = nn.Linear(input_ch, preproc_ch) if preproc_ch else None
        kv_ch = preproc_ch or input_ch

        def create_layer():
            return Sequential(
                cross_attention_layer(num_q_channels=num_latent_channels,
                                      num_kv_channels=kv_ch,
                                      num_heads=num_cross_attention_heads,
                                      dropout=dropout),
                self_attention_block(num_layers=num_self_attention_layers_per_block,
                                     num_channels=num_latent_channels,
                                     num_heads=num_self_attention_heads,
                                     dropout=dropout),
            )

        self.layers = nn.ModuleList(create_layer() for _ in self.scale_shapes)
        # Shared across scales: positional channels retain the same frequency
        # meaning because callers encode every scale with the fine-grid maxima.
        self.latent_proj = nn.Linear(pos_ch, num_latent_channels)

    @staticmethod
    def _resize(z, old_shape, new_shape):
        b, _, c = z.shape
        z = z.transpose(1, 2).reshape(b, c, *old_shape)
        # PositionalEncoder samples every scale with linspace(-1, 1), so the
        # endpoints represent the same physical cells at every resolution.
        z = F.interpolate(z, size=new_shape, mode="bilinear", align_corners=True)
        return z.flatten(2).transpose(1, 2)

    def forward(self, x, scale_pos, pad_mask=None):
        if len(scale_pos) != len(self.scale_shapes):
            raise ValueError(f"expected {len(self.scale_shapes)} positional grids, "
                             f"got {len(scale_pos)}")
        if self.preproc:
            x = self.preproc(x)

        b, z, old_shape = x.shape[0], None, None
        for shape, pos, layer in zip(self.scale_shapes, scale_pos, self.layers):
            if pos.shape[0] != shape[0] * shape[1]:
                raise ValueError(f"scale {shape} needs {shape[0] * shape[1]} positions, "
                                 f"got {pos.shape[0]}")
            q = repeat(self.latent_proj(pos), "... -> b ...", b=b)
            if z is not None:
                q = q + self._resize(z, old_shape, shape)
            z = layer(q, x, pad_mask)
            old_shape = shape
        return z
