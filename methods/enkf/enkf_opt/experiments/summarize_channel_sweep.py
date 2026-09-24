"""Aggregate the channel-wise Q validation sweep and select test candidates.

Selection is deliberately validation-only.  The primary winner minimizes CRPS in
defined/blind cells while allowing at most a 2% all-cell RMSE increase over the
unmodified channel-scale baseline.  A calibration-oriented diagnostic winner is
also reported, but is not allowed to replace the proper-score winner.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
CHANNELS = ("density", "vx", "vy", "var")
AGES = ("blind_1_10", "blind_11_50", "blind_51_plus")


def mean_std(values):
    x = np.asarray(list(values), dtype=float)
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(HERE, "outputs", "channel_sweep"))
    ap.add_argument("--out", default=os.path.join(HERE, "outputs", "channel_sweep_summary.json"))
    ap.add_argument("--expected-days", type=int, default=7)
    ap.add_argument("--rmse-tolerance", type=float, default=0.02)
    ap.add_argument("--baseline-root", default=None,
                    help="optional directory containing external baseline JSON files")
    ap.add_argument("--baseline-suffix", default="r15_rho50",
                    help="filename suffix for external baseline (stored as config 'base')")
    args = ap.parse_args()

    grouped = {}
    for path in glob.glob(os.path.join(args.root, "atc-*_*.json")):
        base = os.path.basename(path)[:-5]
        day, config = base[:12], base[13:]
        grouped.setdefault(config, {})[day] = json.load(open(path))
    if args.baseline_root:
        baseline = {}
        pattern = f"atc-*_{args.baseline_suffix}.json"
        for path in glob.glob(os.path.join(args.baseline_root, pattern)):
            day = os.path.basename(path)[:12]
            baseline[day] = json.load(open(path))
        grouped["base"] = baseline
    if not grouped:
        raise FileNotFoundError(f"no channel-sweep JSON files under {args.root}")
    incomplete = {key: len(rows) for key, rows in grouped.items()
                  if len(rows) != args.expected_days}
    if incomplete:
        raise RuntimeError(f"incomplete configs: {incomplete}")
    if "base" not in grouped:
        raise RuntimeError("the channel sweep must include a 'base' config")

    out = {"configs": {}}
    for config, days in sorted(grouped.items()):
        rows = list(days.values())
        result = {
            "days": sorted(days),
            "failures": sum(r["n_analysis_failures"] for r in rows),
        }
        for split in ("all", "defined", "defined_blind"):
            for metric in ("rmse", "crps", "spread_skill"):
                result[f"{split}_{metric}"] = mean_std(
                    r["scores"][split][metric] for r in rows)
            result[f"{split}_coverage90"] = mean_std(
                r["scores"][split]["coverage"]["90"] for r in rows)
        for channel in CHANNELS:
            for metric in ("rmse", "crps", "spread_skill"):
                result[f"db_{channel}_{metric}"] = mean_std(
                    r["diagnostics"]["per_channel"][channel]["defined_blind"][metric]
                    for r in rows)
            result[f"db_{channel}_coverage90"] = mean_std(
                r["diagnostics"]["per_channel"][channel]["defined_blind"]
                ["coverage"]["90"] for r in rows)
        for age in AGES:
            for metric in ("rmse", "crps", "spread_skill"):
                result[f"age_{age}_{metric}"] = mean_std(
                    r["diagnostics"]["observation_age"][age][metric] for r in rows)
        # Dimensionless diagnostic only: zero is ideal, but CRPS remains the
        # selection objective because spread/skill alone can be gamed by inflation.
        result["channel_log_calibration_error"] = float(sum(
            abs(math.log(max(result[f"db_{channel}_spread_skill"]["mean"], 1e-12)))
            for channel in CHANNELS))
        out["configs"][config] = result

    configs = out["configs"]
    base_rmse = configs["base"]["all_rmse"]["mean"]
    threshold = (1.0 + args.rmse_tolerance) * base_rmse
    eligible = {key: row for key, row in configs.items()
                if row["all_rmse"]["mean"] <= threshold}
    out["selection"] = {
        "baseline": "base",
        "all_rmse_threshold": threshold,
        "primary_min_db_crps_within_rmse_tolerance": min(
            eligible, key=lambda key: eligible[key]["defined_blind_crps"]["mean"]),
        "minimum_defined_blind_crps": min(
            configs, key=lambda key: configs[key]["defined_blind_crps"]["mean"]),
        "minimum_all_crps": min(
            configs, key=lambda key: configs[key]["all_crps"]["mean"]),
        "best_channel_calibration": min(
            configs, key=lambda key: configs[key]["channel_log_calibration_error"]),
        "minimum_long_blind_crps": min(
            configs, key=lambda key: configs[key]["age_blind_51_plus_crps"]["mean"]),
    }

    with open(args.out, "w") as handle:
        json.dump(out, handle, indent=2)

    print("config  all_rmse all_crps db_crps db_ss db_cov90 "
          "den_ss vx_ss vy_ss var_ss cal_error long_crps")
    for config, row in sorted(configs.items(),
                              key=lambda item: item[1]["defined_blind_crps"]["mean"]):
        m = lambda key: row[key]["mean"]
        print(f"{config:7s} {m('all_rmse'):.5f} {m('all_crps'):.5f} "
              f"{m('defined_blind_crps'):.5f} {m('defined_blind_spread_skill'):.3f} "
              f"{m('defined_blind_coverage90'):.3f} "
              f"{m('db_density_spread_skill'):.2f} {m('db_vx_spread_skill'):.2f} "
              f"{m('db_vy_spread_skill'):.2f} {m('db_var_spread_skill'):.2f} "
              f"{row['channel_log_calibration_error']:.3f} "
              f"{m('age_blind_51_plus_crps'):.5f}")
    print("selection", out["selection"])
    print("[out]", args.out)


if __name__ == "__main__":
    main()
