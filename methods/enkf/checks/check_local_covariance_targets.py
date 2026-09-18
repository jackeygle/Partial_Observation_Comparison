"""Small exact checks for paper-style local covariance targets."""
from __future__ import annotations

import torch

from methods.enkf.local_unetkf.targets import (
    accumulate_static_local_covariance,
    extract_centered_patches,
    local_error_outer_products,
    normalized_local_targets,
    sampled_normalized_targets,
)
from methods.enkf.lowrank.model import STATE_SCALE


def main():
    error = torch.arange(2 * 4 * 3 * 4, dtype=torch.float64).reshape(2, 4, 3, 4) / 10
    target, valid = local_error_outer_products(error, 3)
    assert target.shape == (2, 3, 4, 16, 3, 3)
    assert valid.shape == (3, 4, 3, 3)
    # centre=(1,2), pair=(a=2,b=3), neighbour one column to the right
    expected = error[:, 2, 1, 2] * error[:, 3, 1, 3]
    torch.testing.assert_close(target[:, 1, 2, 2 * 4 + 3, 1, 2], expected)
    assert not valid[0, 0, 0, 0]
    assert valid[1, 2, 1, 1]
    assert int(valid[0, 0].sum()) == 4
    assert int(valid[1, 2].sum()) == 9
    assert target[:, 0, 0, :, 0, 0].count_nonzero() == 0

    normalized, _ = normalized_local_targets(error, 3)
    scale = STATE_SCALE[2] * STATE_SCALE[3]
    torch.testing.assert_close(normalized[:, 1, 2, 11, 1, 2], expected / scale)

    accumulated = accumulate_static_local_covariance(None, error, 3)
    torch.testing.assert_close(accumulated[1, 2, 2, 3, 1, 2], expected.sum())
    # The opposite directed entry lives at the other centre and reverse offset.
    torch.testing.assert_close(accumulated[1, 2, 2, 3, 1, 2],
                               accumulated[1, 3, 3, 2, 1, 0])
    centres = torch.tensor([[0, 6], [5, 11]])
    sampled, sampled_valid = sampled_normalized_targets(error, centres, 3)
    for b in range(2):
        for s in range(2):
            row, col = divmod(int(centres[b, s]), 4)
            torch.testing.assert_close(sampled[b, s], normalized[b, row, col])
            torch.testing.assert_close(sampled_valid[b, s, 0], valid[row, col])
    extracted = extract_centered_patches(error, centres, 3)
    assert extracted.shape == (2, 2, 4, 3, 3)
    print("[pass] local target shapes, indexing, normalization, padding, and symmetry")


if __name__ == "__main__":
    main()
