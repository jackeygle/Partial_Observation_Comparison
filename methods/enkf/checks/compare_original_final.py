"""
compare_original_final.py — the original EnKF against the final EnKF, scored identically.

Both versions run through the same GPU filter on the same seven test days, the same robot
observations and the same frames; the only differences are the forecast model and the
process noise (see methods/enkf/sbatch/run_original_enkf.sbatch). This script scores both
exports exactly as supervisor_evaluation/evaluate.py scores the final EnKF:

  * frames [warmup, min(n-1, (n // 200) * 200)) of each day -- the common support used
    in the final comparison (DINCAE needs t+1, 4DVarNet complete 200-frame windows);
  * cells walkable and not observed at that frame, all four channels pooled (and per
    channel);
  * Gaussian predictive N(mean, spread^2), CRPS skill against N(mean, RMSE^2) of the same
    version and cells, spread/RMSE, coverage of the central 90% interval.

The final version's numbers must reproduce supervisor_evaluation's (checked below).

    python3 -m methods.enkf.checks.compare_original_final
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from compare import score_uncertainty as su

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CHANNELS = ("density", "vx", "vy", "variance")
DAYS = ("atc-20130811", "atc-20130818", "atc-20130825", "atc-20130901",
        "atc-20130915", "atc-20130922", "atc-20130929")
DT = 200


def score_version(export_dir: str, warmup: int, days=DAYS) -> dict:
    acc = {"all": su.Accumulator(), **{c: su.Accumulator() for c in CHANNELS}}
    resid = {c: [] for c in CHANNELS}
    for day in days:
        with np.load(os.path.join(export_dir, f"{day}.npz")) as z:
            mean, spread, truth = z["mean"], z["spread"], z["truth"]
            observed, walk = z["observed"].astype(bool), z["walkable"].astype(bool)
        n = len(mean) + warmup                       # the export starts at the warmup frame
        lo, hi = max(1, warmup), min(n - 1, (n // DT) * DT)
        sl = slice(lo - warmup, hi - warmup)
        sel = walk[None] & ~observed[sl]
        for i, c in enumerate(CHANNELS):
            mu, sd, x = mean[sl, i][sel], np.maximum(spread[sl, i][sel], 1e-12), truth[sl, i][sel]
            acc["all"].add(mu, sd, x)
            acc[c].add(mu, sd, x)
            resid[c].append((x - mu).astype(np.float32))
        del mean, spread, truth
    out = {}
    for key, a in acc.items():
        r = a.result()
        d = np.concatenate([np.concatenate(resid[c]) for c in CHANNELS] if key == "all"
                           else resid[key])
        null = su.Accumulator()
        for j in range(0, d.size, 1 << 24):
            chunk = d[j:j + (1 << 24)]
            null.add(np.zeros_like(chunk), np.full(chunk.shape, r["rmse"]), chunk)
        n_res = null.result()
        out[key] = {"rmse": r["rmse"], "spread_mean": r["sigma_mean"],
                    "spread_rmse": r["spread_skill"], "crps": r["crps"],
                    "crps_null": n_res["crps"], "crps_skill": 1 - r["crps"] / n_res["crps"],
                    "coverage90": r["coverage"][90], "n": r["n"]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", default=os.path.join(
        ROOT, "methods/enkf/check_outputs/original_vs_final/exports"))
    ap.add_argument("--final", default=os.path.join(
        ROOT, "supervisor_evaluation/outputs/full/raw/enkf_exports"))
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--days", default=",".join(DAYS), help="comma-separated day stems")
    ap.add_argument("--labels", default="original,final", help="names for --original,--final")
    ap.add_argument("--no-check", action="store_true",
                    help="skip reproducing the published final numbers (other days or configs)")
    ap.add_argument("--out", default=os.path.join(
        ROOT, "methods/enkf/check_outputs/original_vs_final/comparison.json"))
    a = ap.parse_args()

    days = a.days.split(",")
    la, lb = a.labels.split(",")
    res = {la: score_version(a.original, a.warmup, days), lb: score_version(a.final, a.warmup, days)}
    # The final version must match the published unified evaluation.
    if not a.no_check:
        pub = json.load(open(os.path.join(ROOT, "supervisor_evaluation/outputs/full/raw/"
                                                "uncertainty_unified.json")))
        pub = pub["methods"]["enkf"]["results"]["walkable_blind"]
        for k_pub, k_here in (("rmse", "rmse"), ("spread_skill", "spread_rmse")):
            assert abs(pub[k_pub] - res[lb]["all"][k_here]) < 1e-6, (k_pub, pub[k_pub],
                                                                    res[lb]["all"][k_here])
        res["check"] = "final version reproduces supervisor_evaluation (rmse, spread/rmse)"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)

    print(f"{'':12}{'':10}{'RMSE':>8}{'spread':>9}{'spread/RMSE':>13}{'CRPS skill':>12}{'cov90':>8}")
    for key in ("all",) + CHANNELS:
        for v in (la, lb):
            r = res[v][key]
            print(f"{key:12}{v:10}{r['rmse']:8.4f}{r['spread_mean']:9.4f}{r['spread_rmse']:13.3f}"
                  f"{r['crps_skill']:12.3f}{r['coverage90']:8.3f}")
    print(f"[out] {a.out}")


if __name__ == "__main__":
    main()
