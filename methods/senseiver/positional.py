"""
positional.py — sin-cos positional encoding  a = PE(chi)
============================================

Ported verbatim from `reference/Senseiver/positional.py` (the paper's sec.2
first component; the appendix says it uses [23] Vaswani et al.'s sine-cosine
embeddings).

Two changes relative to the reference implementation:

 1. `torch.meshgrid` is passed `indexing="ij"` explicitly. The reference
    implementation doesn't, and newer torch warns about it; `"ij"` is exactly
    the old default behaviour, so the semantics are unchanged.
 2. Added `encoding_channels()`, so callers can ask for the channel count
    instead of repeating the `2 * D * bands` formula elsewhere (the reference
    implementation writes it out by hand in `network_light.py`).

Output convention (matches the reference implementation):
    PositionalEncoder(image_shape, bands) -> (prod(spatial_shape), 2*D*bands)
    Row order = the spatial dimensions flattened in C order, so `enc[flat_index]`
    directly gives any cell's encoding.
    Channel order = [sin(dim0), sin(dim1), ..., cos(dim0), cos(dim1), ...]
"""
from __future__ import annotations

import math

import torch
from einops import rearrange


def PositionalEncoder(image_shape, num_frequency_bands, max_frequencies=None):
    """The last entry of image_shape is the channel count (following the
    reference implementation's signature: pass data.shape[1:])."""
    *spatial_shape, _ = image_shape

    coords = [torch.linspace(-1, 1, steps=s) for s in spatial_shape]
    pos = torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=len(spatial_shape))

    encodings = []
    if max_frequencies is None:
        max_frequencies = pos.shape[:-1]

    # One set of frequencies per spatial dimension: linspace(1, dim/2, bands),
    # the upper bound is that dimension's Nyquist frequency
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
    """Number of positional-encoding channels: bands sin + bands cos per spatial dimension."""
    return 2 * spatial_ndim * num_frequency_bands
