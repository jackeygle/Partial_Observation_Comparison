"""
eval_temporal_controls.py — does a temporal model actually use *when* each observation was made?
================================================================================================

Evaluation-only controls for the temporal extension, run on the best window k after the sweep.
Three inference modes on the same checkpoint, same 7 held-out days, same observations:

  normal   the true relative offsets Δ
  shuffle  Δ randomly permuted among each sample's real tokens -- the set of offsets is
           unchanged, but which observation is how old is destroyed
  zero     every Δ set to 0 -- the model still sees all k frames' tokens but cannot tell
           which frame any of them came from

If `shuffle` and `zero` score like `normal`, any gain over k = 1 came from seeing more tokens,
not from knowing when they were seen.

Scoring is compare5's own, imported rather than re-implemented: `Acc`, `clip_np`,
`build_masks`, the frame range [FRAME_LO, T-FRAME_HI_PAD), the walkable mask from DINCAE's
state_stats with the same equality assert, pooled over days, rmse = sqrt(pooled MSE). Pass
`--reference` (compare5's JSON for the same checkpoint) and `normal` is checked against it
bit for bit before the controls are trusted.

Usage (repo root, GPU node):
    python3 -u -m methods.senseiver.checks.eval_temporal_controls \\
        --ckpt methods/senseiver/runs/temporal/k4_s123/best.pt \\
        --reference methods/senseiver/check_outputs/temporal/eval/k4_s123.json \\
        --out methods/senseiver/check_outputs/temporal/controls_k4_s123.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from compare.compare5 import CONVENTIONS, FRAME_HI_PAD, FRAME_LO, Acc, build_masks, clip_np
from crowdcore import navigation as nav
from crowdcore import paths
from methods.dincae.state import StateStats
from methods.senseiver import dataset as ds
from methods.senseiver import sensors
from methods.senseiver.network import Senseiver

MODES = ("normal", "shuffle", "zero")


def infer(model, Y, Om, dev, batch, rng):
    """One day -> {mode: (T,C,H,W) prediction}. Tokens are built once per batch and reused."""
    k = model.time_window
    pe = model.pos_enc.cpu().numpy(); mu = model.in_mean.cpu().numpy(); sd = model.in_std.cpu().numpy()
    outs = {m: [] for m in MODES}
    T = Y.shape[0]
    with torch.no_grad():
        for i in range(0, T, batch):
            win = [(Y[max(0, j - k + 1):j + 1], Om[max(0, j - k + 1):j + 1]) for j in range(i, min(i + batch, T))]
            tok, pad, dt, n, cell_idx = sensors.build_batch_temporal(
                win, pe, mu, sd, return_cell_idx=True)
            dt_sh = dt.clone()
            for b in range(dt.shape[0]):
                nb = int(n[b])
                if nb > 1:
                    dt_sh[b, :nb] = dt[b, :nb][torch.from_numpy(rng.permutation(nb))]
            tok, pad = tok.to(dev), pad.to(dev)
            for mode, d in (("normal", dt), ("shuffle", dt_sh), ("zero", torch.zeros_like(dt))):
                outs[mode].append(model.reconstruct(
                    tok, pad, d.to(dev), cell_idx.to(dev)).cpu().numpy())
    return {m: np.concatenate(v, 0) for m, v in outs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--reference", default="", help="compare5 JSON for the same checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0, help="seed of the Δ shuffle")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C, H, W = ds.state_shape()
    chans = ds.channels()
    walk = StateStats(os.path.join(paths.method(paths.DINCAE), "artifacts", "state_stats.npz")).valid
    walk_4d = nav.build_valid_mask_from_config()
    assert walk.shape == walk_4d.shape and (walk == walk_4d).all(), "walkable masks disagree"

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = Senseiver(**ck["hparams"]).to(dev)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    if model.time_window < 2:
        raise SystemExit(f"{args.ckpt} has time_window={model.time_window}: controls need a temporal model")
    print(f"[model] {args.ckpt}  time_window={model.time_window}  params={model.num_params:,}  device={dev}", flush=True)

    days = ds.om.split_files("test")
    if args.days:
        days = days[:args.days]
    accs = {m: Acc(C) for m in MODES}
    rng = np.random.default_rng(args.seed)
    for d in days:
        X, Y, Om = ds.load_day(d, stride=1, seed=ds.om.day_seed(d))
        n = X.shape[0]
        Xf, Omf = X.reshape(n, C, H, W), Om.reshape(n, H, W)
        lo, hi = FRAME_LO, n - FRAME_HI_PAD
        sel = {k: v[lo:hi] for k, v in build_masks(Xf, Omf, walk, C).items()}
        preds = infer(model, Y, Om, dev, args.batch, rng)
        line = []
        for m in MODES:
            p = preds[m]
            dacc = Acc(C)
            dacc.add(clip_np(p[lo:hi]), p[lo:hi], Xf[lo:hi], sel)
            accs[m].merge(dacc)
            line.append(f"{m} {np.sqrt(dacc.overall('walkable')):.4f}")
        print(f"  {os.path.basename(d).split('_')[0]}: blind-walkable RMSE  " + "  ".join(line), flush=True)
        del preds

    res = {"ckpt": args.ckpt, "time_window": model.time_window, "shuffle_seed": args.seed, "modes": {}}
    for m in MODES:
        a = accs[m]
        res["modes"][m] = {conv: {"per_channel": {chans[i]: float(v) for i, v in enumerate(a.mse(conv))},
                                  "rmse_pooled": float(np.sqrt(a.overall(conv)))} for conv in CONVENTIONS}

    print(f"\n{'mode':<9}{'blind RMSE':>12}{'all RMSE':>10}" + "".join(f"{c:>9}" for c in chans) + "   vs normal")
    base = res["modes"]["normal"]["walkable"]["rmse_pooled"]
    for m in MODES:
        w = res["modes"][m]["walkable"]; wf = res["modes"][m]["walkable_full"]
        print(f"{m:<9}{w['rmse_pooled']:>12.4f}{wf['rmse_pooled']:>10.4f}"
              + "".join(f"{w['per_channel'][c]:>9.4f}" for c in chans)
              + f"   {(w['rmse_pooled'] - base) / base * 100:+.2f}%")

    if args.reference:
        ref = json.load(open(args.reference))["results"]["Senseiver"]
        dmax = max(abs(res["modes"]["normal"][conv]["per_channel"][c] - ref[conv]["per_channel"][c])
                   for conv in ("walkable", "walkable_full") for c in chans)
        res["reference_check"] = {"reference": args.reference, "max_abs_diff": dmax}
        print(f"\nnormal vs compare5 reference: max|Δ MSE| {dmax:.2e}  "
              f"{'IDENTICAL -- controls are scored exactly like compare5' if dmax == 0 else 'DIFFERENT -- do not trust the controls'}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
