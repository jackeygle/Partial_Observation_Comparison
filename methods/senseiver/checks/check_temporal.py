"""
check_temporal.py — self-checks for the temporal extension (time_window k > 1)
==============================================================================

The temporal extension feeds each sample the sensor tokens of its last k frames,
each token carrying its relative time offset Δ = t_query - t_token
(positional.TemporalEncoding). It is built on variant G-direct. Nothing is trained
before all of these pass.

  1. A one-frame window packs exactly `build_batch`'s tokens, with Δ = 0; a k-frame
     window holds every observed cell of its k frames, and Δ = 0 marks exactly the
     current frame's
  2. Invariance to token order when each token keeps its own Δ
  3. **Teeth**: moving tokens to different Δ must change the output -- otherwise the
     model cannot tell frames apart and "temporal" is only a label
  4. Padding invariance, and an empty window gives no NaN
  5. Every parameter, the time embedding included, receives a gradient
  6. dt is required when k > 1 and refused when k = 1
  7. k = 1 is exactly the old model: the G-direct s123 and variant A checkpoints load
     strictly and reproduce outputs saved before any temporal code existed (CPU, bitwise)
  8. On real data, TemporalDayBank gives the same targets, the same current-frame
     observations and the same input statistics as DayBank (same seed, same stride),
     so a k-run and its k=1 counterpart train on identical targets
Then, not pass/fail: per-step model time and per-batch token-building time for
k = 1, 2, 4, 8, and a projected A100 epoch.

Usage (repo root, GPU node):
    srun -p gpu-debug --gres=gpu:1 -t 00:20:00 --mem=32G bash -c \\
      'source sbatch/_env.sh; python3 -u -m methods.senseiver.checks.check_temporal'
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

from methods.senseiver import dataset as ds
from methods.senseiver import sensors
from methods.senseiver.losses import senseiver_loss
from methods.senseiver.network import Senseiver

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_G = os.path.join(HERE, "check_outputs", "temporal", "ref_Gdirect_s123_before_temporal.pt")
CKPT_G = os.path.join(HERE, "runs", "grid", "Gdirect_s123", "best.pt")
REF_A = os.path.join(HERE, "check_outputs", "grid", "ref_A_before_grid_change.pt")
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
             dec_num_cross_attention_heads=1, dropout=0.0,
             latent_mode="grid", readout="direct")
    d.update(kw)
    return Senseiver(**d)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {dev}" + (f"  {torch.cuda.get_device_name(0)}" if dev.type == "cuda" else ""))
    rng = np.random.default_rng(0)
    C, HW = 4, 432
    pe_np = build().pos_enc.numpy()
    P = pe_np.shape[1]
    mu, sd = np.zeros(C, np.float32), np.ones(C, np.float32)
    Y = rng.standard_normal((20, C, HW)).astype(np.float32)
    Om = rng.random((20, HW)) < 0.45

    # 1. packing
    print("\n[packing]")
    t1, p1, n1 = sensors.build_batch(Y[5:13], Om[5:13], pe_np, mu, sd)
    t2, p2, dt2, n2 = sensors.build_batch_temporal(
        [(Y[j:j + 1], Om[j:j + 1]) for j in range(5, 13)], pe_np, mu, sd)
    ck("1. one-frame window == build_batch, Δ all 0",
       torch.equal(t1, t2) and torch.equal(p1, p2) and torch.equal(n1, n2) and int(dt2.abs().max()) == 0)
    js = (8, 9)
    tw, pw, dtw, nw = sensors.build_batch_temporal([(Y[j - 3:j + 1], Om[j - 3:j + 1]) for j in js],
                                                  pe_np, mu, sd)
    exp = [int(Om[j - 3:j + 1].sum()) for j in js]
    ck("1b. 4-frame window holds every observed cell of its 4 frames, Δ in 0..3",
       nw.tolist() == exp and int(dtw[~pw].min()) == 0 and int(dtw[~pw].max()) == 3, f"n={nw.tolist()}")
    ck("1c. Δ = 0 tokens are exactly the current frame's observations",
       all(int(((dtw[b] == 0) & ~pw[b]).sum()) == int(Om[j].sum()) for b, j in enumerate(js)))

    # 2-6 on a k=4 model
    k, B = 4, 4
    print(f"\n[model k={k}]")
    torch.manual_seed(0)
    m = build(time_window=k).to(dev).eval()
    win = [(Y[j - k + 1:j + 1], Om[j - k + 1:j + 1]) for j in (10, 11, 12, 13)]
    tok, pad, dt, _ = sensors.build_batch_temporal(win, pe_np, mu, sd)
    tok, pad, dt = tok.to(dev), pad.to(dev), dt.to(dev)
    with torch.no_grad():
        out = m.reconstruct(tok, pad, dt)
    ck("   forward shape (B,C,H,W)", tuple(out.shape) == (B, 4, 36, 12), str(tuple(out.shape)))
    Ns = tok.shape[1]
    perm = torch.randperm(Ns, generator=torch.Generator().manual_seed(1)).to(dev)
    with torch.no_grad():
        d2 = float((out - m.reconstruct(tok[:, perm], pad[:, perm], dt[:, perm])).abs().max())
        d3 = float((out - m.reconstruct(tok, pad, (dt + 1) % k)).abs().max())
    ck("2. invariant to token order (each token keeps its Δ)", d2 < TOL, f"max|Δ| = {d2:.2e}")
    ck("3. teeth: moving tokens to other Δ changes the output", d3 > 100 * TOL, f"max|Δ| = {d3:.2e}")
    extra = 50
    tl = torch.zeros(B, Ns + extra, tok.shape[2], device=dev); tl[:, :Ns] = tok
    pl = torch.ones(B, Ns + extra, dtype=torch.bool, device=dev); pl[:, :Ns] = pad
    dl = torch.zeros(B, Ns + extra, dtype=torch.long, device=dev); dl[:, :Ns] = dt
    with torch.no_grad():
        d4 = float((out - m.reconstruct(tl, pl, dl)).abs().max())
    ck("4. invariant to padding length", d4 < TOL, f"max|Δ| = {d4:.2e}")
    te, pe_, de, ne = sensors.build_batch_temporal([(Y[0:k], np.zeros((k, HW), bool))] * B, pe_np, mu, sd)
    with torch.no_grad():
        oe = m.reconstruct(te.to(dev), pe_.to(dev), de.to(dev))
    ck("4b. an empty window gives no NaN", bool(torch.isfinite(oe).all()) and int(ne.max()) == 0)
    m.train()
    m.reconstruct(tok, pad, dt).pow(2).sum().backward()
    ng = [n for n, p in m.named_parameters() if p.requires_grad and (p.grad is None or float(p.grad.abs().sum()) == 0)]
    has_t = any(n.startswith("time_enc.") for n, _ in m.named_parameters())
    ck("5. every parameter has a gradient, time embedding included", not ng and has_t,
       f"none: {ng[:3]}" if ng else "")
    try:
        m.reconstruct(tok, pad); r1 = False
    except ValueError:
        r1 = True
    m1 = build().to(dev)
    try:
        m1.reconstruct(tok, pad, dt); r2 = False
    except ValueError:
        r2 = True
    ck("6. dt required when k > 1, refused when k = 1", r1 and r2)

    print("\n[parameters]")
    base = build().num_params
    line = "  ".join(f"k={kk}: {build(time_window=kk).num_params:,} (+{build(time_window=kk).num_params - base})"
                     for kk in (2, 4, 8))
    print(f"  G-direct (k=1) {base:,}   {line}")

    # 7. k=1 unchanged (CPU bitwise)
    print("\n[k=1 is the old model]")
    torch.set_num_threads(1)
    for label, ref_p, ck_p in (("G-direct s123", REF_G, CKPT_G), ("variant A", REF_A, CKPT_A)):
        ref = torch.load(ref_p, map_location="cpu", weights_only=False)
        c = torch.load(ck_p, map_location="cpu", weights_only=False)
        mm = Senseiver(**c["hparams"]); mm.load_state_dict(c["model"], strict=True); mm.eval()
        with torch.no_grad():
            o = mm.reconstruct(ref["tok"], ref["pad"])
        dd = float((o - ref["out"]).abs().max())
        ck(f"7. {label}: strict load, same keys, output == pre-temporal reference",
           sorted(mm.state_dict().keys()) == ref["keys"] and mm.time_window == 1 and dd == 0.0,
           f"max|Δ| = {dd:.2e}")
    torch.set_num_threads(os.cpu_count() or 1)

    # 8. real data: TemporalDayBank vs DayBank
    print("\n[real data: TemporalDayBank vs DayBank, 1 training day, 600 frames, seed 123]")
    files = ds.om.split_files("train")[:1]
    b1 = ds.DayBank(files, stride=4, seed=123, frames=600, verbose=False)
    bk = ds.TemporalDayBank(files, stride=4, seed=123, frames=600, verbose=False, window=4)
    idx = np.arange(bk.n)
    wins = bk.windows(idx)
    ck("8a. same target frames and values", b1.n == bk.n and np.array_equal(b1.X, bk.X), f"n={bk.n}")
    ck("8b. each window's last frame == DayBank's observation of that frame",
       np.array_equal(np.stack([w[0][-1] for w in wins]), b1.Y)
       and np.array_equal(np.stack([w[1][-1] for w in wins]), b1.Omega))
    ck("8c. target_omega == DayBank's mask", np.array_equal(bk.target_omega(idx), b1.Omega))
    ck("8d. windows are truncated at the start of the day",
       wins[0][0].shape[0] == 1 and wins[1][0].shape[0] == 4, f"lengths {[w[0].shape[0] for w in wins[:3]]}")
    (ma, sa), (mk, sk) = b1.input_stats(), bk.input_stats()
    ck("8e. identical input statistics", np.array_equal(ma, mk) and np.array_equal(sa, sk), f"mean {mk}")

    # timing (informational)
    if dev.type == "cuda":
        print(f"\n[timing] B=64, ~203 observed cells per frame, AMP, {torch.cuda.get_device_name(0)}")
        res = {}
        Ysyn = rng.standard_normal((40, C, HW)).astype(np.float32)
        Osyn = rng.random((40, HW)) < 203 / 432
        for kk in (1, 2, 4, 8):
            t0 = time.perf_counter()
            for r in range(20):
                if kk == 1:
                    sensors.build_batch(Ysyn[r:r + 1].repeat(64, 0), Osyn[r:r + 1].repeat(64, 0), pe_np, mu, sd)
                else:
                    sensors.build_batch_temporal([(Ysyn[r:r + kk], Osyn[r:r + kk])] * 64, pe_np, mu, sd)
            build_ms = (time.perf_counter() - t0) / 20 * 1000
            torch.manual_seed(0)
            mm = build(time_window=kk).to(dev).train()
            opt = torch.optim.Adam(mm.parameters(), lr=1e-3)
            scaler = torch.amp.GradScaler("cuda")
            Ns = 203 * kk
            tk = torch.randn(64, Ns, 4 + P, device=dev)
            pk = torch.zeros(64, Ns, dtype=torch.bool, device=dev)
            dk = torch.randint(0, kk, (64, Ns), device=dev) if kk > 1 else None
            x = torch.randn(64, 4, 36, 12, device=dev)
            for i in range(25):
                if i == 5:
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
                with torch.amp.autocast("cuda"):
                    loss = senseiver_loss(mm.reconstruct(tk, pk, dk), x)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            torch.cuda.synchronize()
            res[kk] = ((time.perf_counter() - t0) / 20 * 1000, build_ms, torch.cuda.max_memory_allocated() / 1e9)
            del mm, opt
            torch.cuda.empty_cache()
        steps = 315643 // 64
        for kk, (mms, bms, gb) in res.items():
            extra = steps * ((mms - res[1][0]) + (bms - res[1][1])) / 1000
            print(f"  k={kk}  model {mms:6.1f} ms/step  build {bms:6.1f} ms/batch  peak {gb:5.2f} GB"
                  f"  -> A100 epoch ≈ 133 s + {extra:5.0f} s ≈ {(133 + extra) / 60:4.1f} min"
                  f"  -> 40 epochs ≈ {(133 + extra) * 40 / 3600:4.1f} h")
        print("  (133 s = G-direct's measured A100 epoch; the model part is measured on this GPU, and the")
        print("   micro-benchmark underestimated real epochs on V100 before -- budget with margin)")

    print("\n" + ("ALL PASS" if not FAIL else f"FAILED: {FAIL}"))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
