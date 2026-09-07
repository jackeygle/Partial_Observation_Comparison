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
"""
from __future__ import annotations

import torch
import torch.nn as nn
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
            self.layer_n = create_layer()          # weight sharing: later blocks reuse this same layer_n
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
