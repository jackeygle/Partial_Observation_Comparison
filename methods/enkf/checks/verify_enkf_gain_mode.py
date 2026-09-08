"""
verify_enkf_gain_mode.py  —  does the fast gain change the SCIENCE, over a full test stretch?
=============================================================================================

checks/verify_enkf_opt.py proves the default path is bit-identical to enkf_lab and shows the
Woodbury path ("ensemble") differs only at rounding level over a few frames. That is not by
itself enough: the filter is a feedback loop (analysis -> bias EMA -> surrogate forecast ->
clipping -> analysis), so a 1e-15 difference can grow. This script runs BOTH gain modes over a
long stretch of a real test day and reports

  * how the state estimates drift apart, frame by frame (max and RMS), and
  * the metric that is actually reported -- per-frame MSE against ground truth -- for each mode,

so the question "is gain_mode='ensemble' safe to run experiments with" is answered with numbers
instead of an argument about floating point.

Both modes use the same seed, same observations, same initial ensemble, and the same number of
random draws per step, so the comparison isolates the gain arithmetic.

Run (from Thesis_Project/4dvarnet_enkf):
    python3 checks/verify_enkf_gain_mode.py --frames 400 --ensemble 100
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from crowdcore import paths
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "enkf_opt"))
from pedpred.utils import load_model                          # noqa: E402
from pedpred.ENKF import LocalizedEnsembleKalmanFilter         # noqa: E402

H, W, F = 36, 12, 4
TOTAL, STATE_DIM = H * W, F * H * W
PROC_STD = (0.02829307, 0.31263075, 0.12325809, 0.41680932)
INIT_STD = (0.2290, 1.2660, 0.3429, 0.0259)


def build_C(cells):
    C = np.zeros((F * len(cells), STATE_DIM))
    idx = np.empty(F * len(cells), dtype=np.int64)
    for i, (r, c) in enumerate(cells):
        for f in range(F):
            col = f * TOTAL + r * W + c
            C[i * F + f, col] = 1.0
            idx[i * F + f] = col
    return C, idx


def run(mode, model, pre, obs_std, ensemble, radius, pass_obs_idx=True):
    f = LocalizedEnsembleKalmanFilter(
        grid_size=(H, W), state_shape=(F, H, W), ensemble_size=ensemble,
        proc_noise_std=PROC_STD, obs_noise_std=tuple(float(s) for s in obs_std),
        init_perturb_std=INIT_STD, inflation=1.02, localization_radius=radius, gain_mode=mode)
    rng = np.random.RandomState(0)
    X0 = np.zeros((ensemble, STATE_DIM))
    for i, s in enumerate(PROC_STD):
        X0[:, i * TOTAL:(i + 1) * TOTAL] = rng.normal(0, s, size=(ensemble, TOTAL))
    f.X = X0

    est = np.zeros((len(pre), F, H, W))
    n_bad = 0
    t0 = time.perf_counter()
    for t, (C, y, idx) in enumerate(pre):
        ok = False
        if C is not None:
            try:
                est[t] = f.step(C, y, model=model, obs_idx=idx if pass_obs_idx else None)
                ok = True
            except np.linalg.LinAlgError:
                n_bad += 1
        if not ok:
            f.X = f._clip_bounds(f.forecast(model))
            est[t] = f.X.mean(axis=0).reshape(F, H, W)
    return est, time.perf_counter() - t0, n_bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--ensemble", type=int, default=100)
    ap.add_argument("--radius", type=int, default=7)
    ap.add_argument("--day", default="atc-20130811")
    ap.add_argument("--out", default="check_outputs/eval/enkf_gain_mode.json")
    args = ap.parse_args()

    npz = os.path.join(paths.enkf_export(), f"obs_{args.day}.npz")
    if not os.path.exists(npz):
        npz = sorted(glob.glob(os.path.join(paths.enkf_export(), "obs_*.npz")))[0]
    z = np.load(npz)
    X_true, Y, Om, obs_std = z["X_true"], z["Y"], z["Omega"], z["obs_std"]
    T = min(args.frames, X_true.shape[0])
    pre = []
    for t in range(T):
        cells = list(zip(*np.where(Om[t])))
        if not cells:
            pre.append((None, None, None)); continue
        C, idx = build_C(cells)
        pre.append((C, C @ Y[t].reshape(-1), idx))
    n_obs = sum(c is not None for c, _, _ in pre)
    print(f"[data] {os.path.basename(npz)}  {T} frames, {n_obs} with observations, "
          f"ensemble={args.ensemble} radius={args.radius}", flush=True)

    model = load_model(os.path.join(ROOT, "enkf_opt", "apt-ibex_train_model_28D.pth"),
                       torch.device("cpu"))
    out = {}
    for mode in ("pinv", "ensemble"):
        est, wall, n_bad = run(mode, model, pre, obs_std, args.ensemble, args.radius)
        mse = ((est - X_true[:T]) ** 2).mean(axis=(1, 2, 3))
        out[mode] = dict(est=est, wall=wall, n_bad=n_bad, mse=mse)
        print(f"[{mode:<8}] {wall:6.1f}s  {wall/T*1000:6.1f} ms/frame  "
              f"MSE={mse.mean():.6f}  skips={n_bad}", flush=True)

    a, b = out["pinv"]["est"], out["ensemble"]["est"]
    d = np.abs(a - b)
    scale = np.abs(a).max()
    per_frame = d.reshape(T, -1).max(axis=1)
    print(f"\n[drift] estimate difference between the two gain modes over {T} frames")
    print(f"  max|d| overall           {d.max():.3e}   (rel to max|est| = {d.max()/scale:.3e})")
    print(f"  RMS|d| overall           {np.sqrt((d**2).mean()):.3e}")
    print(f"  max|d| first / last frame with obs: "
          f"{per_frame[:20].max():.3e} / {per_frame[-20:].max():.3e}")
    print(f"  worst frame              t={int(per_frame.argmax())} ({per_frame.max():.3e})")
    ma, mb = out["pinv"]["mse"].mean(), out["ensemble"]["mse"].mean()
    print(f"\n[metric] mean MSE vs ground truth: pinv {ma:.8f} | ensemble {mb:.8f} "
          f"| delta {mb-ma:+.2e} ({(mb-ma)/ma*100:+.2e} %)")
    print(f"[speed]  {out['pinv']['wall']/out['ensemble']['wall']:.2f}x faster with "
          f"gain_mode='ensemble' on this stretch")

    path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"day": os.path.basename(npz), "frames": int(T), "frames_with_obs": int(n_obs),
               "ensemble": args.ensemble, "radius": args.radius,
               "node": os.environ.get("SLURMD_NODENAME", ""),
               "cores": len(os.sched_getaffinity(0)),
               "wall_s": {k: out[k]["wall"] for k in out},
               "per_frame_ms": {k: out[k]["wall"] / T * 1000 for k in out},
               "mean_mse": {k: float(out[k]["mse"].mean()) for k in out},
               "skips": {k: out[k]["n_bad"] for k in out},
               "max_abs_drift": float(d.max()), "rms_drift": float(np.sqrt((d ** 2).mean())),
               "max_rel_drift": float(d.max() / scale),
               "drift_per_frame_max": per_frame.tolist()},
              open(path, "w"), indent=2)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
