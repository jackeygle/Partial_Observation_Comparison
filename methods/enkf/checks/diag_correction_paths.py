"""
diag_correction_paths.py — which of the filter's TWO correction paths is doing the work?

This EnKF corrects the forecast in two places, not one:

  1. the Kalman gain, `X_a[i] = X_f[i] + K @ (perturbed_y - Y_f[i])` in update() -- textbook,
     and scaled by the ensemble covariance, which has collapsed to ~1% of the true error.
  2. an EMA bias term, `X_f = X_f - self.bias_estimate` in forecast(). update() accumulates
     `bias_estimate <- 0.95*bias_estimate + 0.05*(ensemble mean - observation)` per state index.
     This is NOT textbook EnKF -- the reference's own comments call it "#estimatin NN model
     bias" / "##aded to remove bias" -- and, crucially, it is applied at FULL strength, never
     multiplied by the gain. It keeps correcting the forecast even when K is ~0.

Neither the original paper nor this repo documents path 2, and every statement anyone has made
about "the EnKF's assimilation" so far has silently been about the two of them combined. An
earlier ad-hoc check here (open loop vs the full filter, 32% RMSE apart) has the same flaw: it
switched update() off, which froze bias_estimate at zero, so it moved both paths at once.

Four arms separate them. All four share one RNG stream by construction -- use_gain=False still
draws the observation perturbations and still accumulates bias_estimate, it only discards the
increment -- so this is a paired comparison, not four independent runs.

    A  gain off, bias off   pure forecast: the surrogate rolled forward, nothing assimilated
    B  gain off, bias on    the undocumented path alone
    C  gain on,  bias off   textbook EnKF alone
    D  gain on,  bias on    the shipped configuration

What each outcome would mean:
    C ~= A   the Kalman gain contributes ~nothing; this baseline's assimilation is the hack.
             That has to go in the thesis's description of the baseline.
    B ~= A   the bias path is decorative and the gain is doing the work, as advertised.
    both     they contribute jointly; report the split.

Everything else matches the run of record (ensemble 100, radius 7, inflation 1.02, the shipped
proc_noise_scale=0.01, the same cold start). Scoring is imported from diag_proc_scale_sweep so
the two diagnostics cannot drift apart.

    python3 -m methods.enkf.checks.diag_correction_paths --frames 2000
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
    CH, OPT, PROC_STD, load_model, run, score_arm)

ARMS = {
    "A_forecast_only":  dict(use_gain=False, use_bias=False),
    "B_bias_only":      dict(use_gain=False, use_bias=True),
    "C_gain_only":      dict(use_gain=True,  use_bias=False),
    "D_shipped":        dict(use_gain=True,  use_bias=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--frames", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ensemble", type=int, default=100)
    ap.add_argument("--radius", type=int, default=7)
    ap.add_argument("--scale", type=float, default=0.01, help="shipped process-noise multiplier")
    ap.add_argument("--fix-localization", action="store_true")
    ap.add_argument("--day", default="atc-20130811")
    ap.add_argument("--out", default=os.path.join(paths.eval_out(paths.ENKF),
                                                  "correction_paths.json"))
    a = ap.parse_args()

    src = paths.enkf_export("enkf_k1_full")
    t0 = time.time()
    with np.load(os.path.join(src, f"obs_{a.day}.npz")) as z:
        obs_std = z["obs_std"]
        Omega = z["Omega"]
        T = min(a.frames, Omega.shape[0])
        Omega = Omega[:T]
        X_true = z["X_true"][:T]
        Y = z["Y"][:T]
    data = (X_true, Y, Omega, obs_std)
    print(f"[load] {a.day} -> {T} frames in {time.time() - t0:.1f}s", flush=True)

    model = load_model(os.path.join(OPT, "apt-ibex_train_model_28D.pth"), torch.device("cpu"))
    print(f"[arms] {a.arms}   scale={a.scale} ensemble={a.ensemble} radius={a.radius}", flush=True)

    res = {"day": a.day, "frames": int(T), "warmup": a.warmup, "ensemble": a.ensemble,
           "radius": a.radius, "proc_noise_scale": a.scale, "proc_std": list(PROC_STD),
           "fix_localization": a.fix_localization,
           "arms": {}}
    for name in a.arms:
        kw = ARMS[name]
        t1 = time.time()
        Est, Spread, n_bad = run(a.scale, data, model, a.ensemble, a.radius, T,
                                 fix_localization=a.fix_localization, **kw)
        s = score_arm(Est, Spread, X_true, Omega, a.warmup)
        s["switches"] = kw
        s["n_svd_skips"] = n_bad
        s["seconds"] = round(time.time() - t1, 1)
        res["arms"][name] = s
        al = s["all"]
        print(f"  {name:18s} rmse {al['rmse']:.4f}  sigma {al['sigma_mean']:.5f}  "
              f"spread/skill {al['spread_skill']:.4f}   ({s['seconds']}s)", flush=True)
        print("      " + "  ".join(
            f"{c} {s['per_channel'][c]['rmse']:.4f}" for c in CH), flush=True)

    if len(res["arms"]) == len(ARMS):
        A = res["arms"]["A_forecast_only"]["all"]["rmse"]
        print(f"\n=== RMSE reduction against arm A (pure forecast, {A:.4f}) ===")
        for name in ("B_bias_only", "C_gain_only", "D_shipped"):
            r = res["arms"][name]["all"]["rmse"]
            print(f"  {name:18s} {r:.4f}   {(1 - r / A):+7.1%}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"\n[out] {a.out}")


if __name__ == "__main__":
    main()
