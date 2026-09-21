"""Weight-space ensembling between a clean-trained and a fine-tuned checkpoint.

WiSE-FT (Wortsman et al., CVPR 2022): fine-tuning on a shifted input regime buys
robustness there and gives up accuracy on the clean one.  Linearly interpolating
the two weight vectors,

    theta = (1 - alpha) * theta_clean + alpha * theta_finetuned

traces the whole trade-off between them, and costs no training at all -- one
fine-tuning run yields the entire curve.  Only valid when the two checkpoints
share an architecture and the fine-tune started from the clean one, which is
exactly the ``--init-from`` setup.

    python3 -m methods.enkf.lcskf.checks.interpolate_weights \
        --clean runs/base5_clean_s0/best.pt --finetuned runs/ft_A1_s0/best.pt \
        --alphas 0.25,0.5,0.75 --outdir runs/wise_A1
"""
from __future__ import annotations

import argparse
import copy
import os

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clean", required=True, help="the truth-trained checkpoint")
    p.add_argument("--finetuned", required=True, help="the checkpoint fine-tuned from it")
    p.add_argument("--alphas", default="0.25,0.5,0.75",
                   help="0 reproduces --clean, 1 reproduces --finetuned")
    p.add_argument("--outdir", required=True)
    return p.parse_args()


def blend(a, b, alpha):
    """(1-alpha)*a + alpha*b, elementwise, for every tensor the two share."""
    if set(a) != set(b):
        raise SystemExit("the two checkpoints hold different parameter names")
    out = {}
    for k in a:
        x, y = a[k], b[k]
        if x.shape != y.shape:
            raise SystemExit(f"{k}: {tuple(x.shape)} vs {tuple(y.shape)}")
        out[k] = (x.double() * (1 - alpha) + y.double() * alpha).to(x.dtype) \
            if x.is_floating_point() else y.clone()
    return out


def main():
    args = parse_args()
    clean = torch.load(args.clean, map_location="cpu", weights_only=False)
    tuned = torch.load(args.finetuned, map_location="cpu", weights_only=False)
    for key in ("model", "mean_model"):
        if key not in clean or key not in tuned:
            raise SystemExit(f"both checkpoints need '{key}'; are they joint runs?")
    os.makedirs(args.outdir, exist_ok=True)
    for alpha in (float(v) for v in args.alphas.split(",")):
        if not 0.0 <= alpha <= 1.0:
            raise SystemExit("alphas must lie in [0, 1]")
        # Keep the fine-tuned checkpoint's metadata: it carries the args that say
        # how many history frames to feed and whether the covariance net has a
        # mask channel, which is what load_models() reads back.
        payload = {k: v for k, v in tuned.items() if k not in ("model", "mean_model")}
        payload["model"] = blend(clean["model"], tuned["model"], alpha)
        payload["mean_model"] = blend(clean["mean_model"], tuned["mean_model"], alpha)
        payload["wise_ft"] = {"clean": os.path.abspath(args.clean),
                              "finetuned": os.path.abspath(args.finetuned),
                              "alpha": alpha}
        path = os.path.join(args.outdir, f"wise_a{alpha:g}.pt")
        torch.save(payload, path)
        print(f"[wise] alpha={alpha:g} -> {path}", flush=True)


if __name__ == "__main__":
    main()
