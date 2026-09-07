"""
export_obs_for_enkf.py  —  export our partial observations for the EnKF baseline
================================================================================

For each held-out TEST day, regenerate EXACTLY the same multi-robot partial
observations that 4DVarNet is evaluated on (same config: robots, range,
line-of-sight, temporal sparsity, noise) and save the raw arrays to disk so the
EnKF (which lives in the Partial_observation project and has a colliding `config`
module) can consume them in a separate process — a fair, identical-input compare.

Saves, per test day, check_outputs/enkf/obs_<day>.npz with:
    X_true (T,4,36,12)  ground-truth state
    Y      (T,4,36,12)  noisy partial observation (0 where unobserved)
    Omega  (T,36,12)    observed-cell mask (shared across the 4 channels)
    obs_std (4,)        per-channel observation noise std (for the EnKF's R)

Only the first N frames per day are exported (N = --frames), because the EnKF is
an expensive per-frame filter; N is chosen to cover several dT windows so the
comparison is representative but tractable.

Run:  python3 checks/export_obs_for_enkf.py --frames 1000
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np

from crowdcore import config
from crowdcore import navigation as nav
from crowdcore import observation_model as om


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--frames", type=int, default=1000,
                    help="frames per day to export (EnKF is slow). 0 = the WHOLE day, matching "
                         "run_enkf_baseline.py's --frames 0")
    ap.add_argument("--start", type=int, default=0, help="start frame offset (default 0 = start of day)")
    ap.add_argument("--only", default="", help="only export days whose stem contains this string")
    ap.add_argument("--outdir", default="check_outputs/enkf")
    ap.add_argument("--obs-every-k", type=int, default=None,
                    help="overrides config's observation.obs_every_k. The EnKF's "
                         "estimate is tied to its observation sequence, so k=1 "
                         "and k=4 each need their own export and their own run")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    obs_std = np.asarray(config.get("observation", "obs_std"), dtype=np.float32)
    days = om.split_files(args.split)
    if args.only:
        days = [d for d in days if args.only in d]
    span = "whole day" if args.frames <= 0 else f"[{args.start}, {args.start+args.frames})"
    print(f"[export] {len(days)} {args.split} days, frames {span}, obs_std={obs_std}")
    for d in days:
        stem = os.path.basename(d).split("_")[0]
        X, _ = om.load_state(d)
        # frames=0 means the whole day (same convention as run_enkf_baseline.py); a plain
        # start:start+0 slice would silently export an EMPTY array.
        X = np.asarray(X)[args.start:] if args.frames <= 0 else \
            np.asarray(X)[args.start:args.start + args.frames]
        valid = nav.build_valid_mask_from_config(X)
        out = om.generate_observations(X, add_noise=True, valid_mask=valid,
                                       obs_every_k=args.obs_every_k)
        # observation-based initial fill (same X0 the 4DVarNet solver starts from);
        # the EnKF is initialised from this too, so NEITHER method sees ground truth at t=0.
        X0 = om.fill_missing_state(out["Y"], out["Omega_c"], method=config.get("observation", "init_method"))
        np.savez_compressed(
            os.path.join(args.outdir, f"obs_{stem}.npz"),
            X_true=X.astype(np.float32),
            Y=out["Y"].astype(np.float32),
            Omega=out["Omega"].astype(bool),
            X0=X0.astype(np.float32),
            obs_std=obs_std)
        print(f"  {stem}: exported {X.shape[0]} frames, "
              f"observed cells/frame mean {out['Omega'].reshape(X.shape[0],-1).sum(1).mean():.1f}")
    print(f"[done] -> {args.outdir}/obs_*.npz")


if __name__ == "__main__":
    main()
