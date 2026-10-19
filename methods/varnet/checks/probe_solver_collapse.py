"""Does the learned solver still move the state, or has it collapsed to returning x0?

Diagnoses the training instability of 2026-10-09 in runs/varnet_mse5_h96_pdfold_s3: from
epoch 80 the solver's ConvLSTM stopped receiving usable updates (Adam's eps dominated a
gradient that had fallen to ~1e-22, so the step was ~1e-18, nine orders below the fp32
resolution of the weights). If that is what happened, u_k ~ 0 and the solver returns its
own initial state, which would explain the bit-identical validation scores of epochs 80-83.

Reports ||x^K - x^0|| / ||x^0|| for a few checkpoints on one synthetic window. A healthy
solver moves the state a long way; a collapsed one returns ~0.

  sbatch methods/varnet/sbatch/probe_collapse.sbatch
"""
from __future__ import annotations

import argparse
import os

import torch

from methods.varnet.checks import model_io


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="methods/varnet/runs/varnet_mse5_h96_pdfold_s3")
    ap.add_argument("--ckpts", nargs="+",
                    default=["ckpt_00050.pt", "ckpt_00070.pt", "ckpt_00080.pt", "ckpt_00090.pt"])
    ap.add_argument("--dT", type=int, default=200)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[dev] {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""), flush=True)

    torch.manual_seed(0)
    B, C, T, H, W = 1, 4, a.dT, 36, 12
    x0 = (torch.randn(B, C, T, H, W) * 0.2).to(dev)
    y = (torch.randn(B, C, T, H, W) * 0.2).to(dev)
    mask = (torch.rand(B, 1, T, H, W) < 0.3).float().expand(B, C, T, H, W).contiguous().to(dev)

    print(f"{'checkpoint':<16s} {'||x^K-x^0||/||x^0||':>21s}  {'verdict':>10s}", flush=True)
    for name in a.ckpts:
        path = os.path.join(a.run_dir, name)
        if not os.path.exists(path):
            print(f"{name:<16s} {'missing':>21s}", flush=True)
            continue
        out = model_io.load_solver(path)
        solver = out[0] if isinstance(out, tuple) else out
        solver.to(dev).eval()
        # No torch.no_grad(): the solver differentiates J with respect to the STATE at
        # inference too. Only the weights are frozen.
        xk = solver(x0.clone(), y, mask)
        rel = float(((xk - x0).norm() / x0.norm()).detach())
        print(f"{name:<16s} {rel:21.6e}  {'COLLAPSED' if rel < 1e-3 else 'moves':>10s}", flush=True)


if __name__ == "__main__":
    main()
