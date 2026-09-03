"""
eval_comprehensive.py  —  one evaluation pass covering every method
===================================================================

Replaces the piecemeal scripts (eval_test_days / score_enkf / compare_channels) with a
single pass, so every method is scored on identical frames under an identical protocol.

Protocol:
  * held-out test days, frames read from <dir>/obs_<day>.npz so every method sees the same
    frames and the same observations;
  * EnKF-consistent physical clip applied to EVERY method;
  * scored on the BLIND cells (and, separately, the observed cells).

Metric: plain UNWEIGHTED squared error, mean of (pred-true)^2. This is the convention the
EnKF project itself uses to score its filter (ENKF.py evaluate_enkf takes
sqrt(mean((est-true)^2))), so the two projects' numbers are directly comparable; figures
report its square root, RMSE, which carries the state's own units.

The density-weighted variants this script used to compute are gone. They were added on the
mistaken belief that a density-weighted 'total weighted square error' was the teacher
project's headline metric — it is not; that project scores the EnKF with plain unweighted
RMSE, and its figures use the same. Weighting also made the comparison harder to read
rather than fairer, since it silently drops the empty cells that hold most of the error.

Aggregation: BOTH are reported, because the two existing conventions disagree slightly —
  per-day mean +- std   (test_metrics_matched_clip.json convention, the deck headline)
  pooled over all cells (channel_metrics.json convention)

Run on a GPU node:
    python3 checks/eval_comprehensive.py
    python3 checks/eval_comprehensive.py --add a2=runs/varnet_a2_k1/varnet_best.pt
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np, torch
from crowdcore import config
from crowdcore import navigation as nav
from crowdcore import observation_model as om
from methods.varnet.checks.model_io import load_solver

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENKFDIR = os.path.join(ROOT, "check_outputs", "enkf")
CH = ["density", "vx", "vy", "var"]


def build_solver(ckpt, dev):
    s, a, ck = load_solver(ckpt, dev)
    return s, a, int(ck.get("epoch", -1))


def clip_bounds(x):
    """EnKF-consistent physical bounds, applied to EVERY method (fair)."""
    x = x.copy()
    x[:, 0] = np.clip(x[:, 0], 0, 5); x[:, 1] = np.clip(x[:, 1], -5, 5)
    x[:, 2] = np.clip(x[:, 2], -5, 5); x[:, 3] = np.clip(x[:, 3], 0, 2)
    return x


def reconstruct(solver, dT, x0, Y, Omega_c, batch=32):
    n = (x0.shape[0] // dT) * dT
    win = lambda arr: om.to_windows(arr, dT, as_torch=True)
    x0b, yb, mb = win(x0), win(Y), win(Omega_c.astype(np.float32))
    dev = next(solver.parameters()).device
    outs = []
    with torch.enable_grad():
        for i in range(0, x0b.shape[0], batch):
            outs.append(solver(x0b[i:i+batch].to(dev), yb[i:i+batch].to(dev),
                               mb[i:i+batch].to(dev)).detach().cpu())
    xr = torch.cat(outs, 0).numpy()
    return clip_bounds(om.from_windows(xr)[:n])


def day_scores(pred, true, unobs):
    """Metrics for ONE day: unweighted MSE per region, per channel, plus the ALL pooling.

    _se_sum / _n carry the raw squared-error sum and cell count so the caller can pool
    across days exactly (a mean of per-day means is not the pooled mean when days differ
    in length, and the test days range from 37,964 to 42,890 frames).
    """
    out = {}
    for region, m in (("blind", unobs), ("obs", ~unobs)):
        unw, se_sum, n_sum = {}, 0.0, 0
        for c in range(4):
            u = m[:, c]
            sq = (pred[:, c] - true[:, c]) ** 2
            unw[CH[c]] = float(sq[u].mean()) if u.any() else float("nan")
            se_sum += sq[u].sum(); n_sum += u.sum()
        unw["ALL"] = float(se_sum / max(n_sum, 1))
        out[region] = {"unweighted": unw, "_se_sum": float(se_sum), "_n": int(n_sum)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mse-ckpt", default="runs/varnet_b0_k1/varnet_best.pt")
    ap.add_argument("--add", action="append", default=[],
                    help="extra model as name=path/to/varnet_last.pt (repeatable)")
    ap.add_argument("--dir", default=ENKFDIR,
                    help="dir holding obs_<day>.npz (and est_<day>.npz for the EnKF). Use "
                         "check_outputs/enkf_full for the full-day evaluation")
    ap.add_argument("--out", default="check_outputs/eval/comprehensive_metrics.json")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {}
    for spec in [f"4DVarNet-MSE={args.mse_ckpt}"] + args.add:
        name, path = spec.split("=", 1)
        if os.path.exists(path):
            s, a, ep = build_solver(path, dev)
            models[name] = (s, a["dT"])
            print(f"[model] {name}: {path} (epoch {ep}, dT={a['dT']}, hidden={a['hidden']})", flush=True)
        else:
            print(f"[skip ] {name}: {path} not found", flush=True)

    per_day = {}                                          # method -> day -> scores
    for npz in sorted(glob.glob(os.path.join(args.dir, "obs_*.npz"))):
        stem = os.path.basename(npz)[4:-4]
        z = np.load(npz)
        day = [d for d in om.split_files("test") if stem in d][0]
        X = np.asarray(om.load_state(day)[0])[:z["X_true"].shape[0]]
        out = om.generate_observations(X, add_noise=True,
                                       valid_mask=nav.build_valid_mask_from_config(X))
        x0 = om.fill_missing_state(out["Y"], out["Omega_c"],
                                   method=config.get("observation", "init_method"))

        preds = {}
        est_p = os.path.join(args.dir, f"est_{stem}.npz")  # EnKF may still be running
        if os.path.exists(est_p):
            preds["EnKF"] = clip_bounds(np.asarray(np.load(est_p)["Est"]))
        for name, (solver, dT) in models.items():
            preds[name] = reconstruct(solver, dT, x0, out["Y"], out["Omega_c"])

        for name, p in preds.items():
            k = min(p.shape[0], X.shape[0])
            per_day.setdefault(name, {})[stem] = day_scores(
                p[:k], X[:k], ~out["Omega_c"][:k].astype(bool))
        print(f"  {stem}: done", flush=True)

    # ---- aggregate: per-day mean+-std (deck convention) AND pooled (channel_metrics) ----
    res = {}
    for name, days in per_day.items():
        res[name] = {"per_day": days}
        for region in ("blind", "obs"):
            agg = {}
            for c in CH + ["ALL"]:
                v = np.array([days[d][region]["unweighted"][c] for d in days], float)
                agg[f"unweighted.{c}.mean"] = float(np.nanmean(v))
                agg[f"unweighted.{c}.std"] = float(np.nanstd(v))
                # RMSE per day FIRST, then mean/std over days — sqrt of a mean is not the
                # mean of sqrts, and RMSE is what the figures and the EnKF project report.
                r = np.sqrt(v)
                agg[f"rmse.{c}.mean"] = float(np.nanmean(r))
                agg[f"rmse.{c}.std"] = float(np.nanstd(r))
            agg["unweighted.ALL.pooled"] = float(
                sum(days[d][region]["_se_sum"] for d in days) / sum(days[d][region]["_n"] for d in days))
            agg["rmse.ALL.pooled"] = float(np.sqrt(agg["unweighted.ALL.pooled"]))
            res[name][region] = agg

    order = (["EnKF"] if "EnKF" in res else []) + [n for n in res if n != "EnKF"]
    for region, title in (("blind", "BLIND cells (the reconstruction task)"),
                          ("obs", "OBSERVED cells")):
        for conv, ctitle in (("unweighted", "MSE"), ("rmse", "RMSE  (what the figures show)")):
            print(f"\n=== {title} — {ctitle} ===")
            print(f"{'channel':10s}" + "".join(f"{n:>22s}" for n in order))
            for c in CH + ["ALL"]:
                row = "".join(f"{res[n][region][f'{conv}.{c}.mean']:>13.4f} ±{res[n][region][f'{conv}.{c}.std']:>7.4f}"
                              for n in order)
                print(f"{c:10s}{row}")
        print(f"\n--- pooled ALL over every cell (channel_metrics.json convention) ---")
        for n in order:
            print(f"  {n:16s} MSE {res[n][region]['unweighted.ALL.pooled']:.4f}   "
                  f"RMSE {res[n][region]['rmse.ALL.pooled']:.4f}")

    p = os.path.join(ROOT, args.out)
    json.dump(res, open(p, "w"), indent=2)
    print(f"\n[out] {p}")


if __name__ == "__main__":
    main()
