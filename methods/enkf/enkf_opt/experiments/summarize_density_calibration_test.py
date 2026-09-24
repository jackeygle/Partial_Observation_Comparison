"""Aggregate the locked posterior density calibration on held-out test days."""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))


def stat(values):
    x = np.asarray(list(values), dtype=float)
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(
        HERE, "outputs", "density_calibration_test"))
    ap.add_argument("--out", default=os.path.join(
        HERE, "outputs", "density_calibration_test_summary.json"))
    ap.add_argument("--expected-days", type=int, default=7)
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(args.root, "atc-*_sp_b2_dcal.json")))
    if len(paths) != args.expected_days:
        raise RuntimeError(f"expected {args.expected_days} days, found {len(paths)}")
    rows = [json.load(open(path)) for path in paths]
    if any(not row.get("report_calibration_only") for row in rows):
        raise RuntimeError("one or more rows are not report-only calibration runs")

    out = {"days": [row["config"]["day"] for row in rows], "calibrated": {}, "raw": {}}
    for split in ("all", "defined", "defined_blind"):
        for metric in ("rmse", "crps", "spread_skill"):
            out["calibrated"][f"{split}_{metric}"] = stat(
                row["scores"][split][metric] for row in rows)
            out["raw"][f"{split}_{metric}"] = stat(
                row["raw_scores_before_report_calibration"][split][metric]
                for row in rows)
        out["calibrated"][f"{split}_coverage90"] = stat(
            row["scores"][split]["coverage"]["90"] for row in rows)
        out["raw"][f"{split}_coverage90"] = stat(
            row["raw_scores_before_report_calibration"][split]["coverage"]["90"]
            for row in rows)
    density = out["calibrated"]["density_defined_blind"] = {}
    for metric in ("rmse", "crps", "spread_skill"):
        density[metric] = stat(
            row["diagnostics"]["per_channel"]["density"]["defined_blind"][metric]
            for row in rows)
    density["coverage90"] = stat(
        row["diagnostics"]["per_channel"]["density"]["defined_blind"]
        ["coverage"]["90"] for row in rows)
    out["paired_improvement_days"] = {
        "all_crps": sum(row["scores"]["all"]["crps"]
                        < row["raw_scores_before_report_calibration"]["all"]["crps"]
                        for row in rows),
        "defined_blind_crps": sum(
            row["scores"]["defined_blind"]["crps"]
            < row["raw_scores_before_report_calibration"]["defined_blind"]["crps"]
            for row in rows),
    }
    out["config"] = {key: rows[0]["config"][key] for key in (
        "density_report_gain", "density_report_threshold", "density_report_radius")}
    with open(args.out, "w") as handle:
        json.dump(out, handle, indent=2)

    print("metric                         raw       calibrated    delta")
    for key in ("all_rmse", "all_crps", "defined_blind_rmse",
                "defined_blind_crps", "defined_blind_spread_skill",
                "defined_blind_coverage90"):
        raw, cal = out["raw"][key]["mean"], out["calibrated"][key]["mean"]
        print(f"{key:30s} {raw:.6f}  {cal:.6f}  {cal - raw:+.6f}")
    d = out["calibrated"]["density_defined_blind"]
    print("density calibrated", {key: value["mean"] for key, value in d.items()})
    print("paired improvement days", out["paired_improvement_days"])
    print("[out]", args.out)


if __name__ == "__main__":
    main()
