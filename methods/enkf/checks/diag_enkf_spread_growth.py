"""
diag_enkf_spread_growth.py — does the EnKF's forecast model grow or damp ensemble spread?

The EnKF's uncertainty IS its ensemble spread, and we measured that spread collapsed to ~0.011
of the actual error. Two independent mechanisms could cause that and they call for different
conclusions:

  (a) too little injected noise -- forecast() adds rng.normal(0, 0.01 * proc_noise_vec), i.e.
      1% of the configured process noise. If this is the binding constraint, the collapse is a
      configuration issue and more noise / higher inflation would fix it.
  (b) a contractive forecast model -- PedPred3 is deterministic, and a deterministic model can
      still sustain an ensemble IF its dynamics amplify differences (that is exactly how
      operational weather ensembles work). But if it DAMPS differences, members converge back
      together no matter what is injected, and neither knob helps much.

This measures (b) directly and in isolation: perturb an ensemble, then propagate it through
PedPred3 with NO added noise and NO analysis step, and watch the spread. Growth means the model
sustains an ensemble and (a) is the problem; decay means the model itself is the limit.

Read-only: enkf_lab is imported, never modified.

    python3 checks/diag_enkf_spread_growth.py --steps 30 --ensemble 100
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
LAB = os.path.join(ROOT, "enkf_lab")
sys.path.insert(0, LAB)
from pedpred.utils import load_model            # noqa: E402
from pedpred.ENKF import LocalizedEnsembleKalmanFilter  # noqa: E402

H, W, F = 36, 12, 4
PROC_STD = (0.02829307, 0.31263075, 0.12325809, 0.41680932)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--ensemble", type=int, default=100)
    ap.add_argument("--day", default="atc-20130811")
    ap.add_argument("--out", default="check_outputs/eval/enkf_spread_growth.json")
    args = ap.parse_args()

    model = load_model(os.path.join(LAB, "apt-ibex_train_model_28D.pth"), torch.device("cpu"))
    z = np.load(os.path.join(ROOT, f"check_outputs/enkf_k1_full/obs_{args.day}.npz"))
    x0 = z["X_true"][500].astype(np.float64).reshape(-1)      # a real mid-day state

    f = LocalizedEnsembleKalmanFilter(
        grid_size=(H, W), state_shape=(F, H, W), ensemble_size=args.ensemble,
        proc_noise_std=PROC_STD, obs_noise_std=tuple(float(s) for s in z["obs_std"]),
        inflation=1.0, localization_radius=7)

    rng = np.random.RandomState(0)
    out = {"steps": [], "day": args.day, "ensemble": args.ensemble}
    # Start from a spread of the FULL configured process noise -- 100x what forecast() injects.
    # If the model sustained perturbations, this is where it would show.
    per_cell = np.repeat(np.array(PROC_STD), H * W)
    X = x0[None, :] + rng.normal(0, per_cell, size=(args.ensemble, F * H * W))
    print(f"{'step':>5} {'spread':>10} {'vs step0':>9}")
    s0 = None
    for k in range(args.steps + 1):
        sp = float(X.std(axis=0).mean())
        if s0 is None:
            s0 = sp
        out["steps"].append({"step": k, "spread": sp, "ratio_to_start": sp / s0})
        print(f"{k:>5} {sp:>10.6f} {sp / s0:>8.3f}x", flush=True)
        if k == args.steps:
            break
        X = f._f_model(X, model)          # forecast ONLY: no noise, no analysis, no inflation

    r = out["steps"][-1]["ratio_to_start"]
    out["verdict"] = ("amplifying" if r > 1.05 else "neutral" if r > 0.95 else "contractive")
    out["injected_per_step"] = float(np.mean(PROC_STD) * 0.01)
    p = os.path.join(ROOT, args.out)
    json.dump(out, open(p, "w"), indent=2)
    print(f"\n{args.steps} 步后离散度是初始的 {r:.3f}x  ->  模型是 {out['verdict']} 的")
    print(f"对照: forecast() 每步注入 {out['injected_per_step']:.5f}")
    print(f"      实测 EnKF 运行时的平衡离散度 0.0025")
    print(f"[out] {p}")


if __name__ == "__main__":
    main()
