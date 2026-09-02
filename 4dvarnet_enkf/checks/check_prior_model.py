"""
check_prior_model.py  —  verification for Component 2 (the GENN prior Φ)
========================================================================

Verifies the three things that must hold for Φ to be a valid dynamical prior,
faithful to the paper's zero-centre GENN (§3.2):

  1. Shape       : Φ(x) has the same shape as x  (B,C,T,H,W).
  2. Zero-centre : Φ(x)(s) does NOT depend on x(s) at the SAME space-time point
                   s=(t,i,j) — the paper's defining property (central kernel tap
                   set to zero). Checked two ways:
                     (a) perturbation: changing x at exactly (t,i,j) leaves
                         Φ(·)(t,i,j) bit-identical, while NEIGHBOURS change;
                     (b) gradient: ∂Φ(x)(t,i,j) / ∂x(t,i,j) is exactly zero.
  3. Differentiable: gradients flow back to x (needed by the solver) and to the
                     model parameters (needed for training).

Only the single-scale GENN is the strict paper prior; two-scale (Eq.10) pools
over space and therefore does NOT preserve the clean zero-centre property.

Run (torch compute goes to the GPU debug partition — do not run on the login node CPU):
    srun --partition=gpu-debug --gres=gpu:1 --time=00:15:00 \
         bash -c 'module load scicomp-pytorch-env/2026.1; python3 checks/check_prior_model.py'
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make parent-level modules importable

import torch

from prior_model import GENN, prior_residual


def _rand_state(B=2, C=4, T=7, H=36, W=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, C, T, H, W, generator=g)


def check():
    print("\n=== GENN (single-scale, strict zero-centre) ===")
    torch.manual_seed(0)
    phi = GENN(n_channels=4, hidden=16, two_scale=False)
    x = _rand_state()
    B, C, T, H, W = x.shape
    t, i, j = T // 2, H // 2, W // 2                       # one interior space-time point s

    # 1) Shape
    y = phi(x)
    assert y.shape == x.shape, f"shape mismatch {y.shape} vs {x.shape}"
    print(f"[1] shape OK: phi(x) {tuple(y.shape)} == x")

    # 2a) Perturbation: change x ONLY at the single point (t,i,j); phi at (t,i,j) must
    #     be unchanged (zero-centre), while a neighbour (t,i,j+1) must change.
    x2 = x.clone()
    x2[:, :, t, i, j] += 5.0                               # perturb the exact point s
    with torch.no_grad():
        y2 = phi(x2)
    diff_at_s = (phi(x)[:, :, t, i, j] - y2[:, :, t, i, j]).abs().max().item()
    diff_neigh = (phi(x)[:, :, t, i, j + 1] - y2[:, :, t, i, j + 1]).abs().max().item()
    print(f"[2a] perturb point (t,i,j): change at s = {diff_at_s:.2e} (must be 0); "
          f"at neighbour (t,i,j+1) = {diff_neigh:.2e} (must be >0)")
    assert diff_at_s < 1e-6, "zero-centre broken: phi(x)(s) depends on x(s)!"
    assert diff_neigh > 1e-6, "phi(x)(neighbour) does not use s? prior degenerate"

    # 2b) Gradient: d phi(x)(t,i,j) / d x(t,i,j) must be 0; a neighbour must be non-zero.
    xg = x.clone().requires_grad_(True)
    phi(xg)[:, :, t, i, j].sum().backward()
    grad_at_s = xg.grad[:, :, t, i, j].abs().max().item()
    grad_neigh = xg.grad[:, :, t, i, j + 1].abs().max().item()
    print(f"[2b] d phi(x)(s)/dx: at s = {grad_at_s:.2e} (must be 0); "
          f"at neighbour = {grad_neigh:.2e} (must be >0)")
    assert grad_at_s < 1e-7, "zero-centre broken (gradient)"
    assert grad_neigh > 1e-9, "phi does not depend on neighbours (gradient)"

    # 3) Differentiability + residual term
    xg2 = x.clone().requires_grad_(True)
    loss = (prior_residual(xg2, phi) ** 2).mean()          # ‖x - phi(x)‖²
    loss.backward()
    n_param_grad = sum(int(p.grad is not None and p.grad.abs().sum() > 0)
                       for p in phi.parameters())
    print(f"[3] prior term ‖x-phi(x)‖² = {loss.item():.4f}; differentiable wrt x = "
          f"{xg2.grad is not None}; param tensors with grad = {n_param_grad}")
    assert xg2.grad is not None and n_param_grad > 0

    print("[OK] single-scale GENN: all passed (zero-centre holds, uses neighbours, trainable)")


if __name__ == "__main__":
    n = sum(p.numel() for p in GENN(two_scale=False).parameters())
    print(f"GENN params (single-scale, hidden=32): {n:,}")
    check()
    print("\nall prior-model checks passed")
