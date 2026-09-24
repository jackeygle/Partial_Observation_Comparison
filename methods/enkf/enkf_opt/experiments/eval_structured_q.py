"""Compare independent Gaussian and residual-field process noise in the 100-member EnKF."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
OPT = os.path.join(ROOT, "methods", "enkf", "enkf_opt")
sys.path.insert(0, OPT)

from crowdcore import paths  # noqa: E402
from methods.enkf.checks.diag_proc_scale_sweep import (  # noqa: E402
    CH, F, H, INIT_STD, PROC_STD, STATE_DIM, TOTAL, W, build_C, score_arm,
)
from pedpred.ENKF import LocalizedEnsembleKalmanFilter  # noqa: E402
from pedpred.utils import load_model  # noqa: E402


def apply_rtps(enkf, x_forecast, weight, max_factor=10.0):
    """Relax analysis spread back toward prior spread, preserving its analysis mean."""
    if weight <= 0:
        return
    xa = enkf.X
    mean = xa.mean(axis=0)
    ano = xa - mean
    sf = x_forecast.std(axis=0)
    sa = xa.std(axis=0)
    target = (1.0 - weight) * sa + weight * sf
    factor = np.ones_like(sa)
    good = sa > 1e-10
    factor[good] = np.minimum(target[good] / sa[good], max_factor)
    enkf.X = enkf._clip_bounds(mean + ano * factor)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise-kind", choices=("gaussian", "residual"), required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--rtps", type=float, default=0.0)
    ap.add_argument("--frames", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ensemble", type=int, default=100)
    ap.add_argument("--radius", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gain-mode", choices=("pinv", "ensemble"), default="ensemble")
    ap.add_argument("--day", default="atc-20130811")
    ap.add_argument("--bank", default=os.path.join(OPT, "experiments", "outputs",
                                                    "pedpred3_train_residual_q_8192.npz"))
    ap.add_argument("--residual-marginal", choices=("proc", "raw"), default="proc",
                    help="proc rescales each residual channel to the same marginal std as "
                         "Gaussian PROC_STD, isolating covariance structure from noise size")
    ap.add_argument("--no-center-members", action="store_true",
                    help="do not remove the finite-ensemble mean from each Q draw")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not 0 <= a.rtps <= 1:
        raise ValueError("--rtps must be in [0, 1]")
    if a.warmup >= a.frames:
        raise ValueError("--warmup must be smaller than --frames")

    src = paths.enkf_export("enkf_k1_full")
    obs_path = os.path.join(src, f"obs_{a.day}.npz")
    with np.load(obs_path) as z:
        obs_std = z["obs_std"]
        T = min(a.frames, len(z["Omega"]))
        omega, x_true, y = z["Omega"][:T], z["X_true"][:T], z["Y"][:T]
    bank = None
    if a.noise_kind == "residual":
        with np.load(a.bank) as z:
            bank = z["residuals"].astype(np.float64)
        if bank.ndim != 2 or bank.shape[1] != STATE_DIM:
            raise ValueError(f"bad residual bank shape {bank.shape}")
        if a.residual_marginal == "proc":
            b4 = bank.reshape(len(bank), F, TOTAL)
            std = b4.std(axis=(0, 2))
            b4 *= (np.asarray(PROC_STD) / np.maximum(std, 1e-12))[None, :, None]

    model = load_model(os.path.join(OPT, "apt-ibex_train_model_28D.pth"), torch.device("cpu"))
    enkf = LocalizedEnsembleKalmanFilter(
        grid_size=(H, W), state_shape=(F, H, W), ensemble_size=a.ensemble,
        proc_noise_std=PROC_STD, obs_noise_std=tuple(float(s) for s in obs_std),
        init_perturb_std=INIT_STD, inflation=1.02, localization_radius=a.radius,
        seed=a.seed, gain_mode=a.gain_mode, proc_noise_scale=0.0,
        fix_localization=True, use_gain=True, use_bias=True)
    init_rng = np.random.RandomState(0)
    x_init = np.zeros((a.ensemble, STATE_DIM))
    for f, std in enumerate(PROC_STD):
        x_init[:, f * TOTAL:(f + 1) * TOTAL] = init_rng.normal(
            0, std, size=(a.ensemble, TOTAL))
    enkf.X = x_init
    noise_rng = np.random.RandomState(a.seed + 104729)
    proc_vec = enkf.proc_noise_vec

    est = np.zeros((T, F, H, W), np.float32)
    spread = np.zeros_like(est)
    n_bad = 0
    t0 = time.time()
    for t in range(T):
        if t and t % 200 == 0:
            elapsed = time.time() - t0
            print(f"[{a.noise_kind}, scale={a.scale:g}, rtps={a.rtps:g}] {t}/{T}, "
                  f"{elapsed/t:.3f} s/frame", flush=True)
        # Calling the existing forecast with Q=0 retains its non-standard bias correction and
        # consumes the same EnKF RNG stream in every arm. Q uses an independent paired stream.
        xf0 = enkf.forecast(model)
        if a.noise_kind == "gaussian":
            q = noise_rng.normal(0, proc_vec, size=xf0.shape)
        else:
            q = bank[noise_rng.randint(0, len(bank), size=a.ensemble)].copy()
        q *= a.scale
        if not a.no_center_members:
            q -= q.mean(axis=0, keepdims=True)
        xf = enkf._clip_bounds(xf0 + q)

        cells = list(zip(*np.where(omega[t])))
        ok = False
        if cells:
            c = build_C(cells)
            try:
                enkf.update(xf, c, c @ y[t].reshape(-1))
                apply_rtps(enkf, xf, a.rtps)
                ok = True
            except np.linalg.LinAlgError:
                n_bad += 1
        if not ok:
            enkf.X = xf
        est[t] = enkf.X.mean(axis=0).reshape(F, H, W)
        spread[t] = enkf.get_std()

    scores = score_arm(est, spread, x_true, omega, a.warmup)
    elapsed = time.time() - t0
    result = {
        "config": vars(a), "fix_localization": True, "bias_ema": True,
        "inflation": 1.02, "proc_std": list(PROC_STD), "frames_used": T,
        "n_analysis_failures": n_bad, "seconds": round(elapsed, 2), "scores": scores,
    }
    tag = f"{a.noise_kind}_s{a.scale:g}_rtps{a.rtps:g}_{a.gain_mode}"
    out = a.out or os.path.join(OPT, "experiments", "outputs", f"structured_q_{tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    al, db = scores["all"], scores["defined_blind"]
    print(f"[all] RMSE={al['rmse']:.5f} CRPS={al['crps']:.5f} "
          f"spread/skill={al['spread_skill']:.3f} cov90={al['coverage'][90]:.3f}")
    print(f"[defined_blind] RMSE={db['rmse']:.5f} CRPS={db['crps']:.5f} "
          f"spread/skill={db['spread_skill']:.3f} cov90={db['coverage'][90]:.3f}")
    for name in CH:
        p = scores["per_channel"][name]
        print(f"[{name}] spearman={p['spearman']:+.3f} AUSE={p['ause']:.3f} "
              f"sigma_cv={p['sigma_cv_within_frame']:.3f}")
    print(f"[out] {out}", flush=True)


if __name__ == "__main__":
    main()
