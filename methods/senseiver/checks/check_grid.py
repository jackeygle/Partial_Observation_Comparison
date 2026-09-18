"""
check_grid.py — self-checks for variant G (grid latent)
========================================================

Variant G replaces the 64 abstract latents with one latent token per grid cell
(`latent_mode='grid'`), optionally reading each cell's value straight off its own
token (`readout='direct'`). Nothing gets trained before all of these pass.

Checks:
  1. Sensor permutation invariance (G-dec and G-direct) -- the property Senseiver
     rests on must survive the change of latent
  2. Padding invariance (both), and an empty sensor set gives no NaN
  3. Cell permutation: reorder the cell tokens together with their positional
     encodings. G-dec's output must be *invariant* (the decoder reads z as an
     unordered set); G-direct's output must be *equivariant* (it moves with the
     cells). Plus a teeth check: without undoing the permutation G-direct's
     output must differ -- otherwise every cell predicts the same thing and the
     equivariance would hold trivially. Failing this means something identifies
     cells by array index instead of by positional encoding.
  4. Parameter count: G-dec within 1% of A
  5. Variant A is untouched: the real A checkpoint loads strictly, with the same
     state_dict keys, and on a fixed input reproduces the output saved *before*
     any code was changed (check_outputs/grid/ref_A_before_grid_change.pt, CPU)
  6. The invalid combination latent_mode='abstract' + readout='direct' is rejected
  7. Every parameter of both G variants receives a gradient

Then, not pass/fail: per-step training time and peak memory for A / G-dec /
G-direct at the real batch size, to budget the pilot runs.

Usage (repo root, GPU node):
    srun -p gpu-debug --gres=gpu:1 -t 00:15:00 --mem=16G bash -c \\
      'source sbatch/_env.sh; python3 -u -m methods.senseiver.checks.check_grid'
"""
from __future__ import annotations

import os
import sys
import time

import torch

from methods.senseiver.losses import senseiver_loss
from methods.senseiver.network import Senseiver

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(HERE, "check_outputs", "grid", "ref_A_before_grid_change.pt")
CKPT_A = os.path.join(HERE, "runs", "senseiver_A", "best.pt")
TOL = 1e-4
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


VARIANTS = [("G-dec", dict(latent_mode="grid", readout="decoder")),
            ("G-direct", dict(latent_mode="grid", readout="direct"))]


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator().manual_seed(0)
    print(f"[device] {dev}" + (f"  {torch.cuda.get_device_name(0)}" if dev.type == "cuda" else ""))

    for name, kw in VARIANTS:
        print(f"\n[{name}]")
        torch.manual_seed(0)
        m = build(**kw).to(dev).eval()
        P = m.pos_enc.shape[1]
        HW = m.pos_enc.shape[0]
        B, Ns = 4, 190
        tok = torch.randn(B, Ns, 4 + P, generator=gen).to(dev)
        pad = torch.zeros(B, Ns, dtype=torch.bool, device=dev)
        with torch.no_grad():
            out = m.reconstruct(tok, pad)
        ck("   forward shape (B,C,H,W)", tuple(out.shape) == (B, 4, 36, 12), str(tuple(out.shape)))

        # 1. sensor permutation invariance
        perm = torch.randperm(Ns, generator=gen).to(dev)
        with torch.no_grad():
            d1 = float((out - m.reconstruct(tok[:, perm], pad[:, perm])).abs().max())
        ck("1. invariant to sensor order", d1 < TOL, f"max|Δ| = {d1:.2e}")

        # 2. padding invariance + empty set
        n_real = 100
        t_long = torch.zeros(B, 260, 4 + P, device=dev); t_long[:, :n_real] = tok[:, :n_real]
        p_long = torch.ones(B, 260, dtype=torch.bool, device=dev); p_long[:, :n_real] = False
        p_short = torch.zeros(B, n_real, dtype=torch.bool, device=dev)
        with torch.no_grad():
            d2 = float((m.reconstruct(tok[:, :n_real], p_short) - m.reconstruct(t_long, p_long)).abs().max())
            oe = m.reconstruct(torch.zeros(B, 1, 4 + P, device=dev),
                               torch.zeros(B, 1, dtype=torch.bool, device=dev))
        ck("2. invariant to padding length", d2 < TOL, f"max|Δ| = {d2:.2e}")
        ck("2b. empty sensor set gives no NaN", bool(torch.isfinite(oe).all()))

        # 3. cell permutation
        pe = m.pos_enc
        cperm = torch.randperm(HW, generator=gen).to(dev)
        coords = pe[None].expand(B, -1, -1)
        with torch.no_grad():
            z = m.encoder(tok, pe, pad)
            zp = m.encoder(tok, pe[cperm], pad)
            dz = float((z[:, cperm] - zp).abs().max())
            if kw["readout"] == "decoder":
                d3 = float((m.decoder(z, coords) - m.decoder(zp, coords)).abs().max())
                ck("3. cell permutation: latent equivariant", dz < TOL, f"max|Δ| = {dz:.2e}")
                ck("3. cell permutation: G-dec output invariant", d3 < TOL, f"max|Δ| = {d3:.2e}")
            else:
                o, op = m.head(z), m.head(zp)
                d3 = float((o[:, cperm] - op).abs().max())
                teeth = float((o - op).abs().max())
                ck("3. cell permutation: G-direct output equivariant", d3 < TOL, f"max|Δ| = {d3:.2e}")
                ck("3b. teeth: cells are distinguishable (un-permuted output differs)",
                   teeth > 100 * TOL, f"max|Δ| = {teeth:.2e}")

        # 7. gradients
        m.train()
        m.reconstruct(tok, pad).pow(2).sum().backward()
        no_grad = [n for n, p in m.named_parameters()
                   if p.requires_grad and (p.grad is None or float(p.grad.abs().sum()) == 0)]
        ck("7. every parameter has a gradient", not no_grad, f"none: {no_grad[:3]}" if no_grad else "")

    # 4. parameter counts
    print("\n[parameters]")
    a, gd, gx = build(), build(**VARIANTS[0][1]), build(**VARIANTS[1][1])
    rel = abs(gd.num_params - a.num_params) / a.num_params
    ck("4. G-dec parameter count within 1% of A", rel < 0.01,
       f"A {a.num_params:,} | G-dec {gd.num_params:,} ({rel*100:+.2f}%) | G-direct {gx.num_params:,}")

    # 6. invalid combination
    try:
        build(latent_mode="abstract", readout="direct"); raised = False
    except ValueError:
        raised = True
    ck("6. abstract + direct is rejected", raised)

    # 5. variant A untouched (CPU, same as the reference)
    print("\n[variant A untouched]")
    torch.set_num_threads(1)
    ref = torch.load(REF, map_location="cpu", weights_only=False)
    cka = torch.load(CKPT_A, map_location="cpu", weights_only=False)
    ma = Senseiver(**cka["hparams"])
    ma.load_state_dict(cka["model"], strict=True)
    ma.eval()
    ck("5a. A checkpoint loads strictly with the same state_dict keys",
       sorted(ma.state_dict().keys()) == ref["keys"], f"{len(ref['keys'])} keys")
    ck("5b. A defaults: latent_mode/readout", (ma.latent_mode, ma.readout_mode) == ("abstract", "decoder"))
    with torch.no_grad():
        out_a = ma.reconstruct(ref["tok"], ref["pad"])
    d5 = float((out_a - ref["out"]).abs().max())
    ck("5c. A output reproduces the pre-change reference", d5 < 1e-6,
       f"max|Δ| = {d5:.2e}  (sum now {float(out_a.double().sum()):.10f}, "
       f"ref {float(ref['out'].double().sum()):.10f})")

    # timing (informational)
    if dev.type == "cuda":
        torch.set_num_threads(os.cpu_count() or 1)
        print(f"\n[timing] B=64, Ns=265 sensor tokens, AMP, fwd+bwd+step, "
              f"{torch.cuda.get_device_name(0)}")
        res = {}
        for name, kw in [("A", {})] + VARIANTS:
            torch.manual_seed(0)
            m = build(**kw).to(dev).train()
            opt = torch.optim.Adam(m.parameters(), lr=1e-3)
            scaler = torch.amp.GradScaler("cuda")
            P = m.pos_enc.shape[1]
            tok = torch.randn(64, 265, 4 + P, device=dev)
            pad = torch.zeros(64, 265, dtype=torch.bool, device=dev)
            x = torch.randn(64, 4, 36, 12, device=dev)
            steps, warm = 30, 5
            for i in range(warm + steps):
                if i == warm:
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                    t0 = time.perf_counter()
                with torch.amp.autocast("cuda"):
                    loss = senseiver_loss(m.reconstruct(tok, pad), x)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            torch.cuda.synchronize()
            res[name] = ((time.perf_counter() - t0) / steps * 1000,
                         torch.cuda.max_memory_allocated() / 1e9)
            del m, opt
            torch.cuda.empty_cache()
        steps_per_epoch = 315643 // 64          # A's full training bank
        for name, (ms, gb) in res.items():
            extra = steps_per_epoch * (ms - res["A"][0]) / 1000
            print(f"  {name:<9} {ms:7.1f} ms/step  ({ms / res['A'][0]:4.1f}x A)  peak {gb:5.2f} GB"
                  f"  -> epoch ≈ 121 s + {extra:6.0f} s  ≈ {(121 + extra) / 60:5.1f} min")
        print("  (121 s = variant A's measured epoch on A100; the extra model time is "
              "measured on this GPU, so the projection is approximate)")

    print("\n" + ("ALL PASS" if not FAIL else f"FAILED: {FAIL}"))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
