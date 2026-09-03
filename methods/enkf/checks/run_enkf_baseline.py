"""
run_enkf_baseline.py  —  run the ORIGINAL Localized EnKF as the baseline (lives in OUR repo)
============================================================================================

Drives the UNMODIFIED EnKF from the Partial_observation project (imported read-only via
sys.path; nothing is written into that project) on the observations we exported
(check_outputs/enkf/obs_<day>.npz), and saves the per-frame full-state estimate +
ensemble spread to check_outputs/enkf/est_<day>.npz.

Fairness:
  - initialisation is the ORIGINAL project's own cold start — a random Gaussian ensemble
    (mean 0, std = PROC_STD) per channel, exactly as ENKF.py main()/evaluate_enkf. It does
    NOT use the ground truth (no leakage). 4DVarNet likewise starts from an observation-based
    fill, never the truth.
  - same observations, same test frames, same 4-channel MSE metric as 4DVarNet.

Run (from Thesis_Project/4dvarnet_enkf):
    python3 checks/run_enkf_baseline.py --frames 400 --ensemble 100
"""
from __future__ import annotations
import argparse
import glob
import os
import sys
import time

import numpy as np
from crowdcore import paths
import torch

H, W, F = 36, 12, 4
TOTAL = H * W
STATE_DIM = F * TOTAL
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENKFDIR = paths.enkf_export("enkf")
# EnKF process/init noise: the calibrated values from the original ENKF.py main().
PROC_STD = (0.02829307, 0.31263075, 0.12325809, 0.41680932)
INIT_STD = (0.2290, 1.2660, 0.3429, 0.0259)

# Which copy of the EnKF to drive. All three expose the same package layout (an in-package
# `pedpred -> .` symlink, so the filter's own relative imports resolve) and, by construction,
# the same numerics — `opt` is verified bit-identical to `lab` before use.
#   orig : /scratch/work/zhangx29/Partial_observation — the untouched original project
#   lab  : enkf_lab  — pristine vendored copy of it, read-only, the correctness reference
#   opt  : enkf_opt  — same numerics, optimised: _localization_matrix vectorised (round 1, 18x;
#                      the original spent 96% of runtime in that one Python quadruple loop) plus
#                      the dead recomputation and allocations removed from the rest of the step
#                      (round 2). Default gain path is verified bit-identical to lab by
#                      checks/verify_enkf_opt.py. Setting ENKF_GAIN_MODE=ensemble additionally
#                      switches the gain to its Woodbury form: ~2.4x faster again, and the same
#                      gain, but a different rounding — see enkf_opt/README.md before using it
#                      for anything that gets reported.
SRC = {"orig": "/scratch/work/zhangx29/Partial_observation",
       "lab": os.path.join(ROOT, "enkf_lab"),
       "opt": os.path.join(ROOT, "enkf_opt")}
# Resolved before argparse because the import has to happen at module level; argparse still
# declares --enkf-src so it shows up in --help and is validated there.
_src = "orig"
for _i, _a in enumerate(sys.argv):
    if _a == "--enkf-src" and _i + 1 < len(sys.argv):
        _src = sys.argv[_i + 1]
    elif _a.startswith("--enkf-src="):
        _src = _a.split("=", 1)[1]
if _src not in SRC:
    raise SystemExit(f"--enkf-src must be one of {sorted(SRC)}, got {_src!r}")
PO = SRC[_src]
sys.path.insert(0, PO)
from pedpred.utils import load_model
from pedpred.ENKF import LocalizedEnsembleKalmanFilter

MODEL = os.path.join(PO, "apt-ibex_train_model_28D.pth")


def build_C(obs_cells):
    C = np.zeros((F * len(obs_cells), STATE_DIM))
    for i, (r, c) in enumerate(obs_cells):
        for f in range(F):
            C[i * F + f, f * TOTAL + r * W + c] = 1.0
    return C


def run_day(npz, model, ensemble, radius, frames):
    z = np.load(npz)
    X_true, Y, Omega, obs_std = z["X_true"], z["Y"], z["Omega"], z["obs_std"]
    T = min(frames, X_true.shape[0]) if frames > 0 else X_true.shape[0]
    enkf = LocalizedEnsembleKalmanFilter(
        grid_size=(H, W), state_shape=(F, H, W), ensemble_size=ensemble,
        proc_noise_std=PROC_STD, obs_noise_std=tuple(float(s) for s in obs_std),
        init_perturb_std=INIT_STD, inflation=1.02, localization_radius=radius)
    # ORIGINAL cold start: random Gaussian ensemble per channel (no truth, no observations)
    rng = np.random.RandomState(0)
    X_init = np.zeros((ensemble, STATE_DIM))
    for f, s in enumerate(PROC_STD):
        X_init[:, f * TOTAL:(f + 1) * TOTAL] = rng.normal(0, s, size=(ensemble, TOTAL))
    enkf.X = X_init

    Est = np.zeros((T, F, H, W), np.float32)
    Spread = np.zeros((T, F, H, W), np.float32)
    n_bad = 0
    cum_t = np.zeros(T, np.float64)                      # cumulative wall time after each frame
    t0 = time.time()
    for t in range(T):
        cells = list(zip(*np.where(Omega[t])))
        ok = False
        if cells:
            C = build_C(cells)
            y_obs = C @ Y[t].reshape(-1)
            try:
                est = enkf.step(C, y_obs, model=model); ok = True
            except np.linalg.LinAlgError:
                n_bad += 1
        if not ok:
            Xf = enkf.forecast(model); enkf.X = enkf._clip_bounds(Xf)
            est = enkf.X.mean(axis=0).reshape(F, H, W)
        Est[t] = est
        Spread[t] = enkf.get_std().reshape(F, H, W)
        cum_t[t] = time.time() - t0                       # online filter: time grows frame by frame
    return Est, Spread, T, cum_t[-1] if T else 0.0, n_bad, cum_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--ensemble", type=int, default=100)
    ap.add_argument("--radius", type=int, default=7)
    ap.add_argument("--only", default="")
    ap.add_argument("--dir", default=ENKFDIR, help="dir holding obs_*.npz to read and est_*.npz to write")
    ap.add_argument("--enkf-src", default="orig", choices=sorted(SRC),
                    help="drive which copy of the EnKF: orig=Partial_observation (untouched), "
                         "lab=enkf_lab (pristine vendored copy), opt=enkf_opt (optimised, "
                         "verified bit-identical to lab; ENKF_GAIN_MODE=ensemble makes it "
                         "faster still at a different rounding)")
    ap.add_argument("--timing", default="", help="if set, also save per-frame cumulative wall time to this JSON path")
    args = ap.parse_args()
    # CPU on purpose: this is the baseline the published numbers came from. (enkf_opt's _f_model
    # now takes the device from the model, so loading it on a GPU would move the surrogate's
    # convolutions there; the original hard-coded a CPU input tensor.)
    model = load_model(MODEL, torch.device("cpu"))
    print(f"[enkf] src={_src} ({PO}); ORIGINAL random init; ensemble={args.ensemble} "
          f"radius={args.radius} frames={args.frames} dir={args.dir}", flush=True)
    files = sorted(glob.glob(os.path.join(args.dir, "obs_*.npz")))
    if args.only:
        files = [f for f in files if args.only in f]
    for npz in files:
        stem = os.path.basename(npz)[4:-4]
        Est, Spread, T, dt, n_bad, cum_t = run_day(npz, model, args.ensemble, args.radius, args.frames)
        np.savez_compressed(os.path.join(args.dir, f"est_{stem}.npz"), Est=Est, Spread=Spread)
        print(f"  {stem}: {T} frames in {dt:.1f}s, {n_bad} SVD-skips -> est_{stem}.npz", flush=True)
        # Timing is written ALWAYS, not only on --timing: a run that forgot the flag cannot be
        # re-timed later without repeating the whole filter pass (hours per day), and the
        # hardware string has to be captured on the node that actually did the work — a
        # timing comparison quoted against the wrong CPU is worse than none.
        import json
        cpu_model = "CPU"
        try:
            for l in open("/proc/cpuinfo"):
                if l.startswith("model name"):
                    cpu_model = l.split(":", 1)[1].strip(); break
        except Exception:
            pass
        hw = f"{cpu_model} · {len(os.sched_getaffinity(0))} cores"
        tpath = args.timing or os.path.join(args.dir, f"timing_{stem}.json")
        json.dump({"method": "enkf", "device": "cpu", "hw": hw, "enkf_src": _src,
                   "node": os.environ.get("SLURMD_NODENAME", ""), "day": stem,
                   "ensemble": args.ensemble, "radius": args.radius, "n_frames": int(T),
                   "cum_s": cum_t.tolist(), "total_s": float(dt),
                   "per_frame_s": float(dt / T) if T else 0.0},
                  open(tpath, "w"), indent=2)
        print(f"  timing -> {tpath}  ({hw})", flush=True)
    print("[done]")


if __name__ == "__main__":
    main()
