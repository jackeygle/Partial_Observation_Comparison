"""
score_enkf.py  —  score the EnKF estimates with the SAME metric as 4DVarNet
===========================================================================

Loads the EnKF per-frame estimates (check_outputs/enkf/est_<day>.npz) and the
matching observations (obs_<day>.npz), and computes the identical metrics used for
4DVarNet in eval_test_days.py:

    - blind-zone MSE : error on the UNOBSERVED cells (~Omega, all 4 channels)
    - full-state MSE : error over the whole state

Aggregated over the test days (mean +/- std), written to
check_outputs/eval/enkf_metrics.json. Scored on the SAME frames the EnKF processed,
so it is directly comparable to a 4DVarNet run over the same frames.

Run:  python3 checks/score_enkf.py
"""
from __future__ import annotations
import glob
import json
import os
import sys

import numpy as np
from crowdcore import paths

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENKFDIR = paths.enkf_export()   # default: enkf_k1_full, the model of record


def main():
    ests = sorted(glob.glob(os.path.join(ENKFDIR, "est_*.npz")))
    if not ests:
        print("no est_*.npz found — run the EnKF (run_enkf_on_obs.py) first.")
        return
    per_day = []
    print(f"{'day':16s} {'blindMSE':>10} {'fullMSE':>10} {'frames':>8}")
    print("-" * 48)
    for ep in ests:
        stem = os.path.basename(ep)[4:-4]
        est = np.load(ep)["Est"]                                 # (T,4,36,12)
        z = np.load(os.path.join(ENKFDIR, f"obs_{stem}.npz"))
        X = z["X_true"][:est.shape[0]]                           # (T,4,36,12)
        Omega = z["Omega"][:est.shape[0]]                        # (T,36,12) observed
        unobs = ~Omega[:, None, :, :].repeat(4, axis=1)          # (T,4,36,12) unobserved cells
        blind = float(((est - X) ** 2)[unobs].mean())
        full = float(((est - X) ** 2).mean())
        per_day.append({"day": stem, "blind_mse": blind, "full_mse": full,
                        "n_frames": int(est.shape[0])})
        print(f"{stem:16s} {blind:>10.4f} {full:>10.4f} {est.shape[0]:>8}")

    bl = np.array([p["blind_mse"] for p in per_day])
    fu = np.array([p["full_mse"] for p in per_day])
    print("-" * 48)
    print(f"{'MEAN':16s} {bl.mean():>10.4f} {fu.mean():>10.4f}")
    print(f"{'STD':16s} {bl.std():>10.4f} {fu.std():>10.4f}")
    summary = {"method": "LocalizedEnKF + apt-ibex predictor",
               "blind_mse_mean": float(bl.mean()), "blind_mse_std": float(bl.std()),
               "full_mse_mean": float(fu.mean()), "full_mse_std": float(fu.std()),
               "per_day": per_day}
    outp = os.path.join(paths.eval_out(paths.VARNET), "enkf_metrics.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[out] {outp}")


if __name__ == "__main__":
    main()
