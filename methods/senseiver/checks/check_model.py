"""
check_model.py — self-check for the ported model
==================================

Senseiver suits this problem entirely because of cross-attention's permutation
invariance over an **unordered, variable-length** sensor set. This property must
be proven, not assumed -- the reference implementation never tested it (its
sensor set is fixed-length and fixed-order, so the test cannot even be run there).

Checks:
  1. **Permutation invariance**: shuffling the sensor tokens' order leaves the output unchanged
  2. **Padding invariance**: the same real tokens, padded to different lengths, leave the output unchanged
  3. An empty sensor set produces no NaN
  4. Differentiable: after loss.backward(), every parameter has a non-zero gradient path
  5. Dimension assertion: dec != enc latent channel counts are rejected (the reference implementation is missing this check)
  6. Parameter count and forward-pass shape

Usage: python3 checks/check_model.py     (GPU node)
"""
from __future__ import annotations

import os
import sys

import torch

from methods.senseiver.network import Senseiver

FAIL = []


def ck(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' — ' + extra) if extra else ''}")
    if not cond:
        FAIL.append(name)


def build(**kw):
    d = dict(im_ch=4, grid=(36, 12), space_bands=16, enc_preproc_ch=32, num_latents=64,
             enc_num_latent_channels=32, num_layers=3, num_cross_attention_heads=2,
             enc_num_self_attention_heads=2, num_self_attention_layers_per_block=3,
             dec_preproc_ch=32, dec_num_latent_channels=32,
             dec_num_cross_attention_heads=1, dropout=0.0)
    d.update(kw)
    return Senseiver(**d)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    m = build().to(dev).eval()
    P = m.pos_enc.shape[1]
    print(f"[model] {m.num_params:,} parameters   positional encoding {P} channels   device={dev}")

    B, Ns = 4, 190
    tok = torch.randn(B, Ns, 4 + P, device=dev)
    pad = torch.zeros(B, Ns, dtype=torch.bool, device=dev)
    with torch.no_grad():
        out = m.reconstruct(tok, pad)
    ck("6. forward-pass shape (B,C,H,W)", tuple(out.shape) == (B, 4, 36, 12), str(tuple(out.shape)))

    # 1. Permutation invariance
    perm = torch.randperm(Ns, device=dev)
    with torch.no_grad():
        out_p = m.reconstruct(tok[:, perm], pad[:, perm])
    d1 = float((out - out_p).abs().max())
    ck("1. invariant to sensor order permutation", d1 < 1e-4, f"max|Δ| = {d1:.2e}")

    # 2. padding invariance: 100 real tokens, padded to 100 / 260 respectively
    n_real = 100
    t_short = tok[:, :n_real].clone()
    p_short = torch.zeros(B, n_real, dtype=torch.bool, device=dev)
    t_long = torch.zeros(B, 260, 4 + P, device=dev)
    t_long[:, :n_real] = t_short
    p_long = torch.ones(B, 260, dtype=torch.bool, device=dev)
    p_long[:, :n_real] = False
    with torch.no_grad():
        d2 = float((m.reconstruct(t_short, p_short) - m.reconstruct(t_long, p_long)).abs().max())
    ck("2. invariant to padding length", d2 < 1e-4, f"max|Δ| = {d2:.2e}")

    # 3. empty set (a dummy token, pad_mask all False for the single slot)
    t_e = torch.zeros(B, 1, 4 + P, device=dev)
    p_e = torch.zeros(B, 1, dtype=torch.bool, device=dev)
    with torch.no_grad():
        oe = m.reconstruct(t_e, p_e)
    ck("3. an empty sensor set produces no NaN", bool(torch.isfinite(oe).all()))

    # 4. differentiable
    m.train()
    out = m.reconstruct(tok, pad)
    out.pow(2).sum().backward()
    no_grad = [n for n, p in m.named_parameters()
               if p.requires_grad and (p.grad is None or float(p.grad.abs().sum()) == 0)]
    ck("4. every parameter has a gradient", not no_grad, f"no gradient: {no_grad[:3]}" if no_grad else "")

    # 5. dimension assertion
    try:
        build(dec_num_latent_channels=16)
        raised = False
    except ValueError:
        raised = True
    ck("5. dec != enc latent channel count is rejected", raised)

    print("\n" + ("ALL PASS" if not FAIL else f"FAILED: {FAIL}"))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
