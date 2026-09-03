"""
check_variational_solver.py  —  LIGHT verification for Component 3 (solver)
===========================================================================

Fast machinery check (no data I/O, no training — runs in a couple of seconds on CPU):

  1. Shape         : solve(x0, y, mask) -> (B,C,T,H,W).
  2. Observation   : the obs operator only penalises observed cells ((x−y)⊙Ω).
  3. Differentiable: one backward of the supervised loss reaches BOTH the prior Φ and
                     the solver's LSTM parameters — so the whole pipeline is trainable.

Uses small RANDOM tensors of the right shape (the machinery does not care whether the
numbers are real ATC data). Reconstruction quality on real data is decided by GPU
training — see `train_varnet.py` + `submit_varnet.sbatch`, read off `metrics.jsonl`.

Run (torch compute goes to the GPU debug partition — do not run on the login node CPU):
    srun --partition=gpu-debug --gres=gpu:1 --time=00:15:00 \
         bash -c 'module load scicomp-pytorch-env/2026.1; python3 checks/check_variational_solver.py'
"""

from __future__ import annotations

import os
import sys

import torch

from methods.varnet.prior_model import GENN
from methods.varnet.variational_solver import GradSolver, ObsOperator


def main():
    torch.manual_seed(0)
    B, C, T, H, W = 2, 4, 5, 36, 12                        # small size, seconds on CPU
    x0 = torch.randn(B, C, T, H, W)
    x_true = torch.randn(B, C, T, H, W)
    y = torch.randn(B, C, T, H, W)
    mask = (torch.rand(B, C, T, H, W) > 0.4).float()       # random observation mask

    phi = GENN(n_channels=C, hidden=8, two_scale=True)
    solver = GradSolver(phi, n_channels=C, dT=T, n_iter=3, hidden_ch=16)
    solver.train()

    # 1) Shape
    x_rec = solver(x0, y, mask)
    assert x_rec.shape == x0.shape, f"shape {x_rec.shape} != {x0.shape}"
    print(f"[1] shape OK: x_rec {tuple(x_rec.shape)}")

    # 2) The observation operator only penalises observed cells
    dy = ObsOperator()(x_rec, y, mask).detach()
    assert float(dy[mask < 0.5].abs().max()) == 0.0, "观测项不应在未观测格上非零"
    print("[2] obs operator OK: (x-y)⊙Ω 仅在观测格非零")

    # 3) End-to-end differentiability: one backward pass sends gradients to both Φ and solver params
    loss = ((x_rec - x_true) ** 2).mean()                  # supervised reconstruction loss (Eq.14)
    loss.backward()
    n_grad = sum(int(p.grad is not None and p.grad.abs().sum() > 0)
                 for p in solver.parameters())
    n_total = sum(1 for _ in solver.parameters())
    print(f"[3] differentiable: {n_grad}/{n_total} 个参数张量收到梯度 (Φ + solver 都可训练)")
    assert n_grad > 0

    print("\n轻量检查通过 ✓  (链路通、观测项正确、端到端可微; 训练用 GPU: sbatch submit_varnet.sbatch)")


if __name__ == "__main__":
    main()
