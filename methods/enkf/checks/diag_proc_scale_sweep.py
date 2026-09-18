"""
diag_proc_scale_sweep.py — is the ensemble's collapsed spread a tuning constant or a dead end?

`methods/enkf/README.md` currently concludes, from diag_enkf_spread_growth.py, that the collapse
is "not a tuning problem ... no amount of extra process noise or inflation fixes it". That
experiment propagated a perturbed ensemble with **no injected noise** and found monotone decay
(65% per step, verdict "contractive"). It answers "does the surrogate amplify perturbations on
its own?" (no). It does NOT answer "can the spread be brought to the right scale?", because a
contractive map with per-step injection settles at a non-zero equilibrium proportional to the
injection. Three measurements on the shipped run say that is exactly what happens:

  * forecast() injects `0.01 * proc_noise_vec` (ENKF.py). PROC_STD is not an arbitrary knob --
    estimate_noise_from_data() measured it as the surrogate's OWN one-step error, i.e. precisely
    the Q a textbook EnKF injects at full strength. So the shipped filter injects 1% of Q.
  * on enkf_k1_full, the spread settles within ~5 frames and is then static for the remaining
    ~39,800 frames of the day (density 0.00054 at frame 10, 0.00060 at frame 20,000).
  * that floor equals the injected level: measured spread / injected = 1.93 / 1.16 / 1.26 / 1.02
    for density / vx / vy / var, against ~1.07 predicted by the 65%-per-step decay.

If the equilibrium is set by the injection, raising the injection raises the spread, and
`spread/skill` should move from its published 0.0106 towards 1. This sweeps the multiplier to
find out. Two outcomes are worth distinguishing, and they are NOT the same defect:

  scale     does spread/skill approach 1?   -- a hard-coded constant, fixable
  structure does AUSE / Spearman improve?   -- the spread field is nearly spatially uniform
            (within-frame CV 0.07-0.40, against 0.30-1.47 for the true |error| field), and
            scaling a uniform injection keeps it uniform. Expected NOT to improve.

This is DIAGNOSIS, not retuning: the reported EnKF numbers stay at the shipped 0.01. The point
is to be able to say which half of the defect is configuration and which half is the method.

Everything matches the run of record (run_enkf_baseline.py / enkf_k1_full): ensemble 100,
radius 7, inflation 1.02, the same cold start, the same PROC_STD/INIT_STD. Only the multiplier
moves. Driven through enkf_opt (verified bit-identical to enkf_lab on the default path) because
enkf_lab is read-only and ~24x slower.

    python3 -m methods.enkf.checks.diag_proc_scale_sweep --frames 2000
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from crowdcore import paths                                              # noqa: E402
from compare import score_uncertainty as su                              # noqa: E402

OPT = os.path.join(ROOT, "enkf_opt")
sys.path.insert(0, OPT)
from pedpred.utils import load_model                                     # noqa: E402
from pedpred.ENKF import LocalizedEnsembleKalmanFilter                   # noqa: E402

H, W, F = 36, 12, 4
TOTAL, STATE_DIM = H * W, F * H * W
CH = ("density", "vx", "vy", "var")
# Same constants as run_enkf_baseline.py -- the run of record.
PROC_STD = (0.02829307, 0.31263075, 0.12325809, 0.41680932)
INIT_STD = (0.2290, 1.2660, 0.3429, 0.0259)
FRACS = np.linspace(0.0, 0.9, 19)
# np.trapz was removed in numpy 2.0. checks/diag_sparsification_enkf.py still calls it and will
# break on this environment too -- noted rather than fixed here, so the two are changed together.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def build_C(obs_cells):
    C = np.zeros((F * len(obs_cells), STATE_DIM))
    for i, (r, c) in enumerate(obs_cells):
        for f in range(F):
            C[i * F + f, f * TOTAL + r * W + c] = 1.0
    return C


def run(scale, data, model, ensemble, radius, T, **filter_kw):
    """One arm of the sweep: the filter of record with proc_noise_scale=`scale`.

    `data` is a plain tuple of already-decompressed, already-sliced arrays. The obs_*.npz are
    savez_compressed and ~300 MB; every `z[key]` on an NpzFile re-inflates the whole array
    (measured: 6.5 s for X_true, 8.0 s for Y), so they are read once in main() and passed in.

    `filter_kw` is forwarded to the filter, so diag_correction_paths.py can reuse this to flip
    use_gain / use_bias without a second copy of the driver loop.
    """
    X_true, Y, Omega, obs_std = data
    enkf = LocalizedEnsembleKalmanFilter(
        grid_size=(H, W), state_shape=(F, H, W), ensemble_size=ensemble,
        proc_noise_std=PROC_STD, obs_noise_std=tuple(float(s) for s in obs_std),
        init_perturb_std=INIT_STD, inflation=1.02, localization_radius=radius,
        proc_noise_scale=scale, **filter_kw)
    # ORIGINAL cold start, byte-for-byte the one in run_enkf_baseline.py
    rng = np.random.RandomState(0)
    X_init = np.zeros((ensemble, STATE_DIM))
    for f, s in enumerate(PROC_STD):
        X_init[:, f * TOTAL:(f + 1) * TOTAL] = rng.normal(0, s, size=(ensemble, TOTAL))
    enkf.X = X_init

    Est = np.zeros((T, F, H, W), np.float32)
    Spread = np.zeros((T, F, H, W), np.float32)
    n_bad = 0
    t_start = time.time()
    for t in range(T):
        if t and t % 200 == 0:
            el = time.time() - t_start
            print(f"      [scale={scale:g}] frame {t}/{T}  {el / t:.2f} s/frame  "
                  f"eta {(T - t) * el / t / 60:.0f} min", flush=True)
        cells = list(zip(*np.where(Omega[t])))
        ok = False
        if cells:
            C = build_C(cells)
            try:
                est = enkf.step(C, C @ Y[t].reshape(-1), model=model); ok = True
            except np.linalg.LinAlgError:
                n_bad += 1
        if not ok:
            enkf.X = enkf._clip_bounds(enkf.forecast(model))
            est = enkf.X.mean(axis=0).reshape(F, H, W)
        Est[t] = est
        Spread[t] = enkf.get_std().reshape(F, H, W)
    return Est, Spread, n_bad


def ause_spearman(sig, err):
    """Sparsification: can sigma RANK the errors? Invariant to any monotone rescaling, so the
    scale defect cannot flatter or punish it -- this is the half that scaling cannot fix."""
    n = err.size
    if n < 100:
        return None, None
    curves = {}
    for nm, order in (("sigma", np.argsort(-sig)), ("oracle", np.argsort(-err))):
        c = np.concatenate([[0.0], np.cumsum(err[order] ** 2)])
        tot = c[-1]
        curves[nm] = np.array([np.sqrt((tot - c[int(f * n)]) / max(n - int(f * n), 1))
                               for f in FRACS])
    rng = np.random.RandomState(0)
    c = np.concatenate([[0.0], np.cumsum(err[rng.permutation(n)] ** 2)])
    tot = c[-1]
    rand = np.array([np.sqrt((tot - c[int(f * n)]) / max(n - int(f * n), 1)) for f in FRACS])
    a = _trapz(curves["sigma"] - curves["oracle"], FRACS)
    b = _trapz(rand - curves["oracle"], FRACS)
    sub = rng.choice(n, size=min(n, 200_000), replace=False)   # spearmanr is O(n log n) but heavy
    return (float(a / b) if b > 0 else None,
            float(spearmanr(sig[sub], err[sub]).statistic))


#: Sigma floor, matching eval_uncertainty_enkf.py:77 (`np.maximum(Spread, 1e-12)`). Not
#: cosmetic: 0.79% of the shipped run's Spread entries are EXACTLY 0 (3.17% on density, from
#: members being clipped to the same bound at empty cells), and crps/nll divide by sigma, so
#: without the floor both come out nan. Note the consequence for the published NLL of 1.67e18 --
#: at those cells the term is (x-mu)^2 / (2e-24), so that figure's magnitude is set by this
#: guard constant, not by anything measured. It is a "sigma collapsed" flag, not a score.
SIGMA_FLOOR = 1e-12


def _defined(X):
    """(T,C,H,W) bool -- that channel is defined here ∩ walkable. Same body as
    eval_uncertainty_enkf.py:_defined, and like it the rule itself is imported from DINCAE's
    single definition rather than copied, so the two files cannot disagree about what "defined"
    means. Imported lazily: the census script pulls names from this module and has no reason to
    load navigation."""
    from crowdcore import navigation as nav
    from methods.dincae.state import channel_valid
    cv = np.ascontiguousarray(channel_valid(X).transpose(1, 0, 2, 3))
    return cv & nav.build_valid_mask_from_config()[None, None].astype(bool)


def score_arm(Est, Spread, X_true, Omega, warmup):
    mu, x = Est[warmup:], X_true[warmup:]
    sig = np.maximum(Spread[warmup:].astype(np.float64), SIGMA_FLOOR)
    # Omega is (T,H,W): a robot observes all 4 channels of a cell together, so it has to be
    # broadcast onto the channel axis before it can index a (T,C,H,W) array. Same expansion as
    # eval_uncertainty_enkf.py:78, so the splits mean the same thing in both files.
    om = np.repeat(Omega[warmup:][:, None], mu.shape[1], axis=1).astype(bool)
    # "all" is dominated by the ~82% of cells that are empty, where the truth is a priori 0 and
    # an observation carries only noise -- diag_observed_vs_blind.py measured assimilation
    # HURTING there (density observed/blind = 1.34) while helping in every occupied stratum.
    # So a pooled number cannot answer "does the gain help where the answer is not trivially
    # known". These two splits can:
    #   defined  = that channel is defined here ∩ walkable -- the project's main convention,
    #              imported from its single definition rather than re-derived.
    #   occupied = density > 0, the cells where velocity and variance are real quantities.
    dfn = _defined(x)
    occ = np.repeat((x[:, 0] > 0)[:, None], mu.shape[1], axis=1)
    out = {}
    for split, sel in (("all", None), ("blind", ~om), ("observed", om),
                       ("defined", dfn), ("occupied", occ),
                       ("defined_blind", dfn & ~om), ("defined_observed", dfn & om)):
        acc = su.Accumulator()
        m_, s_, x_ = (mu, sig, x) if sel is None else (mu[sel], sig[sel], x[sel])
        acc.add(m_, s_, x_)
        out[split] = acc.result()
        # Constant-sigma null model on the SAME cells, exactly as eval_uncertainty_enkf.py does
        # it: sigma = this split's own RMSE (su.const_sigma), scored by the same Accumulator.
        # The ratio crps/crps_null is the verdict the method-comparison figure uses, so without
        # it this sweep could say "calibration comes back" but not whether sigma stops losing
        # to a constant -- which is the claim that actually needs checking.
        if split in ("all", "defined", "defined_blind") and out[split]:
            base = su.Accumulator()
            base.add(m_, np.full(np.shape(m_), su.const_sigma(m_, x_)), x_)
            b = base.result()
            out[f"{split}_constant_sigma_baseline"] = b
            out[split]["crps_over_null"] = out[split]["crps"] / b["crps"]
    # per-channel ranking power + how uniform the sigma field is
    out["per_channel"] = {}
    for c in range(F):
        s, e = sig[:, c].ravel(), np.abs(mu[:, c] - x[:, c]).ravel()
        a, sp = ause_spearman(s.astype(np.float64), e.astype(np.float64))
        fr = sig[:, c].reshape(len(sig), -1)
        out["per_channel"][CH[c]] = {
            "ause": a, "spearman": sp,
            "sigma_mean": float(s.mean()),
            "sigma_cv_within_frame": float((fr.std(axis=1) / (fr.mean(axis=1) + 1e-12)).mean()),
            "rmse": float(np.sqrt((e ** 2).mean())),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    ap.add_argument("--frames", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=500,
                    help="frames dropped before scoring; the spread floor is reached in ~5")
    ap.add_argument("--ensemble", type=int, default=100)   # the run of record
    ap.add_argument("--radius", type=int, default=7)
    ap.add_argument("--fix-localization", action="store_true",
                    help="restore the feature-block division the reference dropped, so all "
                         "four fields are assimilated instead of density alone")
    ap.add_argument("--day", default="atc-20130811")
    ap.add_argument("--out", default=os.path.join(paths.eval_out(paths.ENKF)
                                                  if hasattr(paths, "ENKF") else
                                                  os.path.join(ROOT, "check_outputs/eval"),
                                                  "proc_scale_sweep.json"))
    a = ap.parse_args()

    src = paths.enkf_export("enkf_k1_full")
    t0 = time.time()
    with np.load(os.path.join(src, f"obs_{a.day}.npz")) as z:
        # Read each array exactly once, then slice -- see run()'s docstring.
        obs_std = z["obs_std"]
        Omega = z["Omega"]
        T = min(a.frames, Omega.shape[0])
        Omega = Omega[:T]
        X_true = z["X_true"][:T]
        Y = z["Y"][:T]
    data = (X_true, Y, Omega, obs_std)
    print(f"[load] {a.day} -> {T} frames in {time.time() - t0:.1f}s", flush=True)
    model = load_model(os.path.join(OPT, "apt-ibex_train_model_28D.pth"), torch.device("cpu"))
    print(f"[sweep] day={a.day} frames={T} warmup={a.warmup} ensemble={a.ensemble} "
          f"radius={a.radius}  scales={a.scales}", flush=True)
    print(f"        injected per step = scale x PROC_STD = scale x {PROC_STD}", flush=True)

    res = {"day": a.day, "frames": int(T), "warmup": a.warmup, "ensemble": a.ensemble,
           "radius": a.radius, "proc_std": list(PROC_STD),
           "fix_localization": a.fix_localization, "arms": {}}
    for scale in a.scales:
        t0 = time.time()
        Est, Spread, n_bad = run(scale, data, model, a.ensemble, a.radius, T,
                                 fix_localization=a.fix_localization)
        s = score_arm(Est, Spread, X_true, Omega, a.warmup)
        s["n_svd_skips"] = n_bad
        s["seconds"] = round(time.time() - t0, 1)
        res["arms"][str(scale)] = s
        al = s["all"]
        print(f"  scale={scale:<6g} rmse {al['rmse']:.4f}  sigma {al['sigma_mean']:.5f}  "
              f"spread/skill {al['spread_skill']:.4f}  cov@90 {al['coverage'][90]:.3f}  "
              f"crps {al['crps']:.4f}   ({s['seconds']}s, {n_bad} svd-skips)", flush=True)
        for c in CH:
            p = s["per_channel"][c]
            print(f"      {c:8s} ause {str(round(p['ause'], 3)):>6s}  "
                  f"spearman {p['spearman']:+.3f}  sigma_cv {p['sigma_cv_within_frame']:.3f}  "
                  f"rmse {p['rmse']:.4f}", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"\n[out] {a.out}")


if __name__ == "__main__":
    main()
