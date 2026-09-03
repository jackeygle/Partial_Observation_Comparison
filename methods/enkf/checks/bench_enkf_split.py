"""
bench_enkf_split.py  —  split the EnKF's per-frame cost into forecast and analysis
=================================================================================

Answers the one question the stacked bars in results_speed.png make: WHY does the EnKF get
~2.5x cheaper when observations drop from every frame to every 4th? The filter does two
different things per frame and only one of them depends on observation density:

    forecast   advance all 100 ensemble members through the neural surrogate.
               Runs on EVERY frame, observed or not.
    analysis   the Kalman update — localization weights, gain, member perturbations.
               Runs ONLY on frames that carry an observation.

The split is EXACT rather than apportioned: LocalizedEnsembleKalmanFilter.step() is literally

    X_f = self.forecast(model)
    X_a = self.update(X_f, C, y_obs)

so calling those two directly, with a timer between them, changes nothing about the numerics
and charges every microsecond to one side or the other. `update` also does the bound clipping
(it ends with self.X = self._clip_bounds(X_a)), so clipping counts as analysis on observed
frames; on unobserved frames the driver clips the forecast itself, so it counts as forecast
there. Same convention as checks/bench_speed.py's driver.

Why this file exists at all: the split reported in the presentation was originally measured
with a throwaway script, so check_outputs/eval/enkf_time_split.json could be read but not
reproduced. Anything that ends up on a slide has to be re-runnable.

Method follows checks/bench_speed.py, which is the method-vs-method benchmark and the source of
the totals those shares are applied to:
  * measurements INTERLEAVED (k1 k4, k1 k4, ...) so drift in the node's background load spreads
    over both settings instead of landing on one and reading as a difference between them;
  * one warm-up pass before the timed repeats (lazy init, allocator, first-touch faults);
  * repeats reported as mean, with the spread kept in the json;
  * hardware, node and core count recorded next to every number.

Run on a node, NOT the login node, and with the SAME core count as bench_speed.py so the
shares can be applied to its totals:
    srun -p batch-csl -c 4 --mem=24G -t 0:30:00 \
        python3 -u checks/bench_enkf_split.py --frames 300 --repeats 3
"""
from __future__ import annotations
import argparse
import importlib
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "checks"))

# Reuse the benchmark's own constants and hardware probe rather than restating them: if the
# ensemble size, noise or localization radius ever changes there, this must follow, and the
# shares are only meaningful applied to that script's totals.
from bench_speed import (F, H, INIT_STD, PROC_STD, SRC, STATE_DIM, TOTAL, W,  # noqa: E402
                         build_C, hw_info)


def load_enkf(src):
    """Import one copy of the filter in isolation and return (module, surrogate model)."""
    for m in [k for k in list(sys.modules) if k.startswith(("pedpred", "tools"))]:
        del sys.modules[m]
    root = SRC[src]
    sys.path.insert(0, root)
    E = importlib.import_module("pedpred.ENKF")
    U = importlib.import_module("pedpred.utils")
    model = U.load_model(os.path.join(root, "apt-ibex_train_model_28D.pth"), torch.device("cpu"))
    sys.path.remove(root)
    return E, model


def prepare(obs_npz, n_frames):
    """Per-frame (C, y) pairs. Built OUTSIDE the timed region: assembling C is our driver's
    bookkeeping, not the filter's work, and it would otherwise be charged to the analysis."""
    z = np.load(obs_npz)
    Y, Om, obs_std = z["Y"], z["Omega"], z["obs_std"]
    if n_frames > Y.shape[0]:
        raise SystemExit(f"{obs_npz} has {Y.shape[0]} frames, --frames asked for {n_frames}")
    pre = []
    for t in range(n_frames):
        cells = list(zip(*np.where(Om[t])))
        C = build_C(cells) if cells else None
        pre.append((C, C @ Y[t].reshape(-1) if C is not None else None))
    return pre, tuple(float(s) for s in obs_std)


def run_once(E, model, pre, obs_std):
    """One full pass over `pre`, timing forecast and analysis separately.

    Returns (forecast_s, analysis_s, n_observed, n_failed). A frame whose analysis raises
    LinAlgError is charged to the forecast only, matching how the drivers treat it: the
    forecast is kept and clipped, the update is discarded."""
    f = E.LocalizedEnsembleKalmanFilter(
        grid_size=(H, W), state_shape=(F, H, W), ensemble_size=100,
        proc_noise_std=PROC_STD, obs_noise_std=obs_std,
        init_perturb_std=INIT_STD, inflation=1.02, localization_radius=7)
    rng = np.random.RandomState(0)
    X0 = np.zeros((100, STATE_DIM))
    for i, s in enumerate(PROC_STD):
        X0[:, i * TOTAL:(i + 1) * TOTAL] = rng.normal(0, s, size=(100, TOTAL))
    f.X = X0

    fore = anal = 0.0
    n_obs = n_bad = 0
    for C, y in pre:
        t0 = time.perf_counter()
        X_f = f.forecast(model)
        t1 = time.perf_counter()
        fore += t1 - t0
        if C is None:
            # No observation this frame: keep the forecast, clip it, no analysis to pay for.
            t2 = time.perf_counter()
            f.X = f._clip_bounds(X_f)
            fore += time.perf_counter() - t2
            continue
        try:
            f.update(X_f, C, y)                  # assigns self.X = _clip_bounds(X_a) internally
            anal += time.perf_counter() - t1
            n_obs += 1
        except np.linalg.LinAlgError:
            t2 = time.perf_counter()
            f.X = f._clip_bounds(X_f)
            fore += time.perf_counter() - t2
            n_bad += 1
    return fore, anal, n_obs, n_bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--day", default="atc-20130811")
    ap.add_argument("--enkf-src", default="opt", choices=sorted(SRC),
                    help="which copy of the filter to time; 'opt' is what the deck reports")
    ap.add_argument("--out", default="check_outputs/eval/enkf_time_split.json")
    args = ap.parse_args()

    E, model = load_enkf(args.enkf_src)
    settings = {}
    for k in (1, 4):
        npz = f"{ROOT}/check_outputs/enkf_k{k}_full/obs_{args.day}.npz"
        settings[k] = prepare(npz, args.frames)

    hw = hw_info()
    print(f"[hw] {hw['cpu']} · {hw['cores_avail']} cores · node {hw['node'] or '?'}")
    print(f"[setup] {args.frames} frames of {args.day}, enkf_src={args.enkf_src}, "
          f"{args.repeats} interleaved repeats\n", flush=True)

    # Warm-up: one short pass per setting, untimed.
    for k in (1, 4):
        pre, std = settings[k]
        run_once(E, model, pre[:10], std)

    acc = {k: {"fore": [], "anal": []} for k in (1, 4)}
    n_obs_of = {}
    for r in range(args.repeats):
        for k in (1, 4):                                  # interleaved, not k1 x3 then k4 x3
            pre, std = settings[k]
            fore, anal, n_obs, n_bad = run_once(E, model, pre, std)
            acc[k]["fore"].append(fore)
            acc[k]["anal"].append(anal)
            n_obs_of[k] = n_obs
            print(f"  [{r + 1}/{args.repeats}] k={k}  forecast {fore / args.frames * 1000:7.2f} "
                  f"analysis {anal / args.frames * 1000:7.2f} ms/frame"
                  + (f"   ({n_bad} analyses failed -> forecast only)" if n_bad else ""),
                  flush=True)

    runs = {}
    for k in (1, 4):
        fo = np.array(acc[k]["fore"]) / args.frames * 1000               # ms per frame
        an = np.array(acc[k]["anal"]) / args.frames * 1000
        tot = fo.mean() + an.mean()
        runs[f"k{k}"] = {
            "k": k, "n_frames": args.frames, "n_observed": n_obs_of[k],
            "forecast_ms": float(fo.mean()), "analysis_ms": float(an.mean()),
            "total_ms": float(tot),
            "forecast_share": float(fo.mean() / tot), "analysis_share": float(an.mean() / tot),
            # Spread kept so a reader can see whether the shares are stable, not just their mean.
            "forecast_ms_std": float(fo.std()), "analysis_ms_std": float(an.std()),
            "forecast_ms_all": fo.tolist(), "analysis_ms_all": an.tolist(),
        }

    rec = {"hw": hw["cpu"], "cores": hw["cores_avail"], "node": hw["node"],
           "enkf_src": args.enkf_src, "day": args.day, "repeats": args.repeats, "runs": runs}
    p = os.path.join(ROOT, args.out)
    json.dump(rec, open(p, "w"), indent=2)

    print(f"\n{'':6s}{'forecast':>12}{'analysis':>12}{'total':>10}{'obs frames':>12}")
    for k in (1, 4):
        d = runs[f"k{k}"]
        print(f"k={k:<4}{d['forecast_ms']:>10.2f}ms{d['analysis_ms']:>10.2f}ms"
              f"{d['total_ms']:>8.2f}ms{d['n_observed']:>9d}/{d['n_frames']}"
              f"   ({d['forecast_share']:.1%} / {d['analysis_share']:.1%})")
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
