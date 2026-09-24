"""Aggregate the seven-day structured-Q EnKF evaluation."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_DAYS = (
    "atc-20130811", "atc-20130818", "atc-20130825", "atc-20130901",
    "atc-20130915", "atc-20130922", "atc-20130929",
)
ARMS = ("gaussian_s1", "residual_s1", "residual_s15")


def result_path(root, day, arm):
    direct = os.path.join(root, f"{day}_{arm}.json")
    if os.path.exists(direct):
        return direct
    if day == "atc-20130811":
        old = {"gaussian_s1": "formal_gaussian_s1_r0_pinv.json",
               "residual_s1": "formal_residual_s1_r0_pinv.json",
               "residual_s15": "formal_residual_s15_r0_pinv.json"}
        return os.path.join(os.path.dirname(root), old[arm])
    return direct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(HERE, "outputs", "full7"))
    ap.add_argument("--out", default=os.path.join(HERE, "outputs", "full7_summary.json"))
    a = ap.parse_args()
    raw = {}
    missing = []
    for day in TEST_DAYS:
        raw[day] = {}
        for arm in ARMS:
            path = result_path(a.root, day, arm)
            if not os.path.exists(path):
                missing.append(path)
                continue
            with open(path) as f:
                raw[day][arm] = json.load(f)
    if missing:
        raise FileNotFoundError("missing results:\n" + "\n".join(missing))

    summary = {"days": list(TEST_DAYS), "arms": {}, "paired_vs_gaussian": {}, "raw": raw}
    scalar = (("all", "rmse"), ("all", "crps"), ("all", "spread_skill"),
              ("defined", "rmse"), ("defined", "crps"),
              ("defined_blind", "rmse"), ("defined_blind", "crps"),
              ("defined_blind", "spread_skill"))
    for arm in ARMS:
        row = {"analysis_failures_total": int(sum(raw[d][arm]["n_analysis_failures"]
                                                   for d in TEST_DAYS))}
        for split, metric in scalar:
            vals = np.asarray([raw[d][arm]["scores"][split][metric] for d in TEST_DAYS])
            row[f"{split}_{metric}"] = {"mean": float(vals.mean()),
                                         "std_across_days": float(vals.std(ddof=1)),
                                         "per_day": vals.tolist()}
        cov = np.asarray([raw[d][arm]["scores"]["defined_blind"]["coverage"]["90"]
                          for d in TEST_DAYS])
        row["defined_blind_coverage90"] = {"mean": float(cov.mean()),
                                            "std_across_days": float(cov.std(ddof=1)),
                                            "per_day": cov.tolist()}
        for channel in ("density", "vx", "vy", "var"):
            vals = np.asarray([raw[d][arm]["scores"]["per_channel"][channel]["spearman"]
                               for d in TEST_DAYS])
            row[f"{channel}_spearman"] = {"mean": float(vals.mean()),
                                           "std_across_days": float(vals.std(ddof=1)),
                                           "per_day": vals.tolist()}
        summary["arms"][arm] = row

    for arm in ("residual_s1", "residual_s15"):
        paired = {}
        for split, metric in (("all", "rmse"), ("all", "crps"),
                              ("defined", "crps"), ("defined_blind", "crps")):
            base = np.asarray([raw[d]["gaussian_s1"]["scores"][split][metric]
                               for d in TEST_DAYS])
            vals = np.asarray([raw[d][arm]["scores"][split][metric] for d in TEST_DAYS])
            delta = 100.0 * (vals / base - 1.0)
            paired[f"{split}_{metric}"] = {
                "mean_percent_change": float(delta.mean()),
                "wins_lower_is_better": int((vals < base).sum()),
                "per_day_percent_change": delta.tolist(),
            }
        summary["paired_vs_gaussian"][arm] = paired

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(summary, f, indent=2)
    print("arm             all RMSE       all CRPS       DB CRPS        DB cov90   failures")
    for arm in ARMS:
        r = summary["arms"][arm]
        def fmt(k):
            x = r[k]
            return f"{x['mean']:.5f}+/-{x['std_across_days']:.5f}"
        print(f"{arm:15s} {fmt('all_rmse'):14s} {fmt('all_crps'):14s} "
              f"{fmt('defined_blind_crps'):14s} {r['defined_blind_coverage90']['mean']:.3f} "
              f"{r['analysis_failures_total']:5d}")
    for arm, p in summary["paired_vs_gaussian"].items():
        print(f"{arm} vs gaussian:")
        for key, value in p.items():
            print(f"  {key}: {value['mean_percent_change']:+.1f}% "
                  f"({value['wins_lower_is_better']}/7 days better)")
    print(f"[out] {a.out}")


if __name__ == "__main__":
    main()
