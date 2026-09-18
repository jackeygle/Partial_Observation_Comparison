"""Numerical and gradient checks for the differentiable low-rank Kalman layer."""
from __future__ import annotations

import torch

from methods.enkf.lowrank.kalman import analysis_update, gaussian_nll, posterior_diag
from methods.enkf.lowrank.model import CovarianceUNet


def main():
    torch.manual_seed(7)
    dtype = torch.float64
    b, c, h, w, rank = 2, 2, 3, 2, 3
    n = c * h * w
    mean = torch.randn(b, c, h, w, dtype=dtype, requires_grad=True)
    factor = (0.15 * torch.randn(b, rank, c, h, w, dtype=dtype)).requires_grad_()
    diagonal = (0.1 + torch.rand(b, c, h, w, dtype=dtype)).requires_grad_()
    target = torch.randn_like(mean)
    observation = target + 0.1 * torch.randn_like(target)
    mask = torch.rand(b, h, w) > 0.35
    obs_var = torch.tensor([0.04, 0.09], dtype=dtype)

    got = analysis_update(mean, factor, diagonal, observation, mask, obs_var, jitter=0.0)
    got_var = posterior_diag(mean, factor, diagonal, observation, mask, obs_var, jitter=0.0)
    for j in range(b):
        u = factor[j].permute(1, 2, 3, 0).reshape(n, rank)
        prior = u @ u.T + torch.diag(diagonal[j].reshape(-1))
        keep = mask[j][None].expand(c, h, w).reshape(-1).nonzero().squeeze(1)
        prior_oo = prior[keep][:, keep]
        rv = obs_var[:, None, None].expand(c, h, w).reshape(-1)[keep]
        gain = prior[:, keep] @ torch.linalg.inv(prior_oo + torch.diag(rv))
        expected = mean[j].reshape(-1) + gain @ (
            observation[j].reshape(-1)[keep] - mean[j].reshape(-1)[keep])
        post = prior - gain @ prior[keep]
        torch.testing.assert_close(got[j].reshape(-1), expected, rtol=2e-10, atol=2e-10)
        torch.testing.assert_close(got_var[j].reshape(-1), post.diag(), rtol=2e-10, atol=2e-10)

    loss = (got - target).square().mean() + 0.01 * gaussian_nll(
        target, mean, factor, diagonal, jitter=0.0)
    loss.backward()
    assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in (mean, factor, diagonal))

    net = CovarianceUNet(rank=8, width=8).float()
    x = torch.randn(3, 4, 36, 12)
    u, d = net(x, x + 0.01 * torch.randn_like(x))
    assert u.shape == (3, 8, 4, 36, 12) and d.shape == x.shape
    assert (d > 0).all()
    print("[PASS] dense equivalence, posterior variance, gradients, and U-Net shapes")


if __name__ == "__main__":
    main()
