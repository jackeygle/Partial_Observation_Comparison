"""Targets for learning local PedPred3 forecast-error covariance maps.

For every spatial centre ``p`` the supervised target contains the 16 directed
channel pairs ``e[p, a] * e[q, b]`` for neighbours ``q`` in an odd square
window.  Keeping all 4x4 pairs avoids making an invalid within-patch symmetry
assumption; covariance symmetry relates predictions made at two different
centres and is imposed only when a global/local Kalman matrix is assembled.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from methods.enkf.lowrank.model import STATE_SCALE


def _check_patch(patch_size: int) -> int:
    if patch_size < 1 or patch_size % 2 != 1:
        raise ValueError(f"patch_size must be a positive odd integer, got {patch_size}")
    return patch_size // 2


def local_error_outer_products(error: torch.Tensor, patch_size: int = 15):
    """Return directed local outer products and a padding-validity mask.

    Parameters
    ----------
    error:
        Forecast residual ``truth - forecast``, shape ``(B, 4, H, W)``.
    patch_size:
        Odd local window.  ``15`` corresponds to offsets ``[-7, 7]``.

    Returns
    -------
    target:
        Shape ``(B, H, W, 16, P, P)``. Output channel ``4*a+b`` is
        ``error[a, centre] * error[b, neighbour]``.
    valid:
        Boolean shape ``(H, W, P, P)``; false entries are outside the grid.
    """
    radius = _check_patch(patch_size)
    if error.ndim != 4 or error.shape[1] != 4:
        raise ValueError(f"error must have shape (B,4,H,W), got {tuple(error.shape)}")
    batch, channels, height, width = error.shape
    n_cells = height * width
    patches = F.unfold(error, kernel_size=patch_size, padding=radius)
    patches = patches.reshape(batch, channels, patch_size, patch_size, n_cells)
    patches = patches.permute(0, 4, 1, 2, 3)                 # B,N,b,P,P
    centres = error.reshape(batch, channels, n_cells).permute(0, 2, 1)
    target = centres[:, :, :, None, None, None] * patches[:, :, None]
    target = target.reshape(batch, height, width, channels * channels,
                            patch_size, patch_size)

    ones = error.new_ones((1, 1, height, width))
    valid = F.unfold(ones, kernel_size=patch_size, padding=radius)
    valid = valid.squeeze(0).transpose(0, 1).reshape(
        height, width, patch_size, patch_size).bool()
    return target, valid


def normalized_local_targets(error: torch.Tensor, patch_size: int = 15):
    """As :func:`local_error_outer_products`, divided by channel-pair scales."""
    target, valid = local_error_outer_products(error, patch_size)
    scale = error.new_tensor(STATE_SCALE)
    pair_scale = (scale[:, None] * scale[None, :]).reshape(16)
    return target / pair_scale.view(1, 1, 1, 16, 1, 1), valid


def extract_centered_patches(field: torch.Tensor, centres: torch.Tensor,
                             patch_size: int = 15):
    """Extract patches at per-sample flat spatial indices.

    ``field`` is ``(B,C,H,W)``, ``centres`` is ``(B,S)``, and the result is
    ``(B,S,C,P,P)``. Values outside the physical grid are zero padded.
    """
    radius = _check_patch(patch_size)
    if field.ndim != 4 or centres.ndim != 2 or centres.shape[0] != field.shape[0]:
        raise ValueError("field must be (B,C,H,W) and centres must be (B,S)")
    batch, channels, height, width = field.shape
    patches = F.unfold(field, kernel_size=patch_size, padding=radius)
    patches = patches.transpose(1, 2).reshape(
        batch, height * width, channels, patch_size, patch_size)
    index = centres[:, :, None, None, None].expand(
        -1, -1, channels, patch_size, patch_size)
    return patches.gather(1, index)


def sampled_normalized_targets(error: torch.Tensor, centres: torch.Tensor,
                               patch_size: int = 15):
    """Normalized 16-map targets only at requested centres.

    Returns ``targets (B,S,16,P,P)`` and ``valid (B,S,1,P,P)``.
    """
    batch, channels, height, width = error.shape
    if channels != 4:
        raise ValueError("the ATC state must have four channels")
    patches = extract_centered_patches(error, centres, patch_size)
    flat = error.reshape(batch, channels, height * width).transpose(1, 2)
    centre_error = flat.gather(1, centres[:, :, None].expand(-1, -1, channels))
    target = centre_error[:, :, :, None, None, None] * patches[:, :, None]
    target = target.reshape(batch, centres.shape[1], 16, patch_size, patch_size)
    scale = error.new_tensor(STATE_SCALE)
    target = target / (scale[:, None] * scale[None, :]).reshape(1, 1, 16, 1, 1)
    ones = error.new_ones((batch, 1, height, width))
    valid = extract_centered_patches(ones, centres, patch_size).bool()
    return target, valid


def accumulate_static_local_covariance(
    total: torch.Tensor | None,
    error: torch.Tensor,
    patch_size: int = 15,
):
    """Accumulate ``sum_t e_t[p,a] e_t[q,b]`` without materialising targets.

    The result has shape ``(H, W, 4, 4, P, P)``.  This streaming form is used
    for the climatological baseline over many forecast frames.
    """
    radius = _check_patch(patch_size)
    if error.ndim != 4 or error.shape[1] != 4:
        raise ValueError(f"error must have shape (B,4,H,W), got {tuple(error.shape)}")
    batch, channels, height, width = error.shape
    n_cells = height * width
    patches = F.unfold(error, kernel_size=patch_size, padding=radius)
    patches = patches.reshape(batch, channels, patch_size * patch_size, n_cells)
    centres = error.reshape(batch, channels, n_cells)
    chunk = torch.einsum("ban,bckn->nack", centres, patches)
    chunk = chunk.reshape(height, width, channels, channels, patch_size, patch_size)
    return chunk if total is None else total + chunk
