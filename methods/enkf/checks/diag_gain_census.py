"""
diag_gain_census.py — why is the Kalman gain inert? Measure it instead of estimating it.

checks/diag_correction_paths.py found that switching the Kalman gain OFF changes nothing:
arm C (gain only) reproduces arm A (pure forecast) to four significant figures, while the
undocumented EMA bias path accounts for the entire ~30% RMSE reduction the filter achieves.
That is the *what*. This is the *why*, measured rather than argued.

The gain is K = P_xy @ pinv(P_yy) with

    P_yy = (Y_ano^T Y_ano)/(N-1)  +  diag(R)  +  regularization * I

Two candidate explanations, which call for different write-ups:

  (a) the ensemble collapsed, so P_xy and the first term of P_yy are both ~0 -- the gain is
      small because the filter believes its forecast. This is the story README tells.
  (b) `regularization=1e-3`, a hard-coded numerical guard, dominates the denominator. With the
      shipped spread the ensemble term is ~1e-5, i.e. ~77x smaller than the guard, and for the
      `var` channel (R = 0.00645^2 = 4.2e-5) the guard exceeds even the observation noise. If
      so, the gain is being suppressed by a constant that has nothing to do with the physics.

`ens_frac` = ens / (ens + R + reg) separates them: near 0 with reg as the biggest term means
(b), near 0 with R as the biggest term means (a).

Uses the shipped configuration of record (ensemble 100, radius 7, inflation 1.02,
proc_noise_scale 0.01, gain_mode pinv). The census hook is read-only and off by default.

    python3 -m methods.enkf.checks.diag_gain_census --frames 400
"""
from __future__ import annotations
import argparse
import json
import os
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from crowdcore import paths                                              # noqa: E402
from methods.enkf.checks.diag_proc_scale_sweep import (                  # noqa: E402
    CH, F, H, INIT_STD, OPT, PROC_STD, STATE_DIM, TOTAL, W, build_C, load_model)
from pedpred.ENKF import LocalizedEnsembleKalmanFilter                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=100, help="census rows dropped before summary")
    ap.add_argument("--ensemble", type=int, default=100)
    ap.add_argument("--radius", type=int, default=7)
    ap.add_argument("--scale", type=float, default=0.01)
    ap.add_argument("--fix-localization", action="store_true")
    ap.add_argument("--day", default="atc-20130811")
    ap.add_argument("--out", default=os.path.join(paths.eval_out(paths.ENKF),
                                                  "gain_census.json"))
    a = ap.parse_args()

    src = paths.enkf_export("enkf_k1_full")
    with np.load(os.path.join(src, f"obs_{a.day}.npz")) as z:
        obs_std = z["obs_std"]
        Omega = z["Omega"][:a.frames]
        Y = z["Y"][:a.frames]
    T = len(Omega)
    model = load_model(os.path.join(OPT, "apt-ibex_train_model_28D.pth"), torch.device("cpu"))

    enkf = LocalizedEnsembleKalmanFilter(
        grid_size=(H, W), state_shape=(F, H, W), ensemble_size=a.ensemble,
        proc_noise_std=PROC_STD, obs_noise_std=tuple(float(s) for s in obs_std),
        init_perturb_std=INIT_STD, inflation=1.02, localization_radius=a.radius,
        proc_noise_scale=a.scale, fix_localization=a.fix_localization)
    rng = np.random.RandomState(0)
    X_init = np.zeros((a.ensemble, STATE_DIM))
    for f, s in enumerate(PROC_STD):
        X_init[:, f * TOTAL:(f + 1) * TOTAL] = rng.normal(0, s, size=(a.ensemble, TOTAL))
    enkf.X = X_init
    enkf.collect_stats = True

    t0 = time.time()
    for t in range(T):
        cells = list(zip(*np.where(Omega[t])))
        if cells:
            C = build_C(cells)
            try:
                enkf.step(C, C @ Y[t].reshape(-1), model=model)
            except np.linalg.LinAlgError:
                pass
        else:
            enkf.X = enkf._clip_bounds(enkf.forecast(model))
    rows = enkf.gain_stats[a.warmup:]
    print(f"[census] {a.day}  {T} frames, {len(rows)} scored assimilations "
          f"({time.time() - t0:.0f}s)", flush=True)

    reg = rows[0]["reg"]
    print(f"\n=== P_yy 的三项(对角线均值),regularization = {reg:g} ===")
    print(f"{'channel':10s}{'ensemble':>12s}{'R':>12s}{'reg':>12s}{'ens_frac':>11s}"
          f"{'谁最大':>10s}")
    summary = {}
    for c in range(F):
        vals = [r["per_channel"][c] for r in rows if c in r["per_channel"]]
        if not vals:
            continue
        e = float(np.mean([v["ens"] for v in vals]))
        r_ = float(np.mean([v["R"] for v in vals]))
        frac = float(np.mean([v["ens_frac"] for v in vals]))
        biggest = max((("ensemble", e), ("R", r_), ("reg", reg)), key=lambda kv: kv[1])[0]
        sg = float(np.mean([v["self_gain_mean"] for v in vals]))
        sgm = float(np.mean([v["self_gain_max"] for v in vals]))
        summary[CH[c]] = {"ens": e, "R": r_, "reg": reg, "ens_frac": frac,
                          "self_gain_mean": sg, "self_gain_max": sgm}
        print(f"{CH[c]:10s}{e:12.3e}{r_:12.3e}{reg:12.3e}{frac:11.2e}{biggest:>10s}")

    print(f"\n=== 自增益 K[cell, 该 cell 的观测] —— 一个格子自己的新息有多少进入状态 ===")
    print(f"{'channel':10s}{'均值':>14s}{'绝对值最大':>14s}")
    for c in range(F):
        if CH[c] in summary:
            s = summary[CH[c]]
            print(f"{CH[c]:10s}{s['self_gain_mean']:14.3e}{s['self_gain_max']:14.3e}")

    inc = np.array([r["incr_norm"] for r in rows])
    st = np.array([r["state_norm"] for r in rows])
    inn = np.array([r["innov_norm"] for r in rows])
    bias = np.array([r["bias_norm"] for r in rows])
    print(f"\n=== 每步 analysis 实际移动了多少 ===")
    print(f"  ||K @ innovation||            均值 {inc.mean():.4e}")
    print(f"  ||innovation||                均值 {inn.mean():.4e}"
          f"   -> 增益整体把新息缩小了 {inn.mean() / max(inc.mean(), 1e-300):.3g} 倍")
    print(f"  ||state||                     均值 {st.mean():.4e}")
    print(f"  相对状态量级 incr/state       均值 {(inc / st).mean():.3e}")
    print(f"  ||bias_estimate||(对照)       均值 {bias.mean():.4e}"
          f"   -> 偏差修正比 analysis 增量大 {bias.mean() / max(inc.mean(), 1e-300):.3g} 倍")

    out = {"day": a.day, "frames": int(T), "n_rows": len(rows), "regularization": reg,
           "ensemble": a.ensemble, "radius": a.radius, "proc_noise_scale": a.scale,
           "fix_localization": a.fix_localization,
           "per_channel": summary,
           "incr_norm_mean": float(inc.mean()), "innov_norm_mean": float(inn.mean()),
           "state_norm_mean": float(st.mean()), "bias_norm_mean": float(bias.mean()),
           "incr_over_state_mean": float((inc / st).mean()),
           "bias_over_incr": float(bias.mean() / max(inc.mean(), 1e-300))}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\n[out] {a.out}")


if __name__ == "__main__":
    main()
