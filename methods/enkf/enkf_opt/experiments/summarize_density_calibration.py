"""Aggregate the validation-only posterior density-sigma calibration grid."""
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
        HERE, "outputs", "density_calibration_validation"))
    ap.add_argument("--out", default=os.path.join(
        HERE, "outputs", "density_calibration_validation_summary.json"))
    ap.add_argument("--expected-days", type=int, default=7)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.root, "atc-*_grid.json")))
    if len(paths) != args.expected_days:
        raise RuntimeError(f"expected {args.expected_days} days, found {len(paths)}")
    rows = [json.load(open(path)) for path in paths]
    keys = set(rows[0]["density_calibration_grid"])
    if any(set(row["density_calibration_grid"]) != keys for row in rows[1:]):
        raise RuntimeError("calibration grids do not have identical candidates")

    metrics = ("density_db_crps", "density_db_sigma_mean",
               "density_db_spread_skill", "density_db_coverage90",
               "defined_blind_crps", "defined_blind_coverage90", "all_crps")
    out = {"days": [row["config"]["day"] for row in rows], "candidates": {}}
    for key in sorted(keys):
        first = rows[0]["density_calibration_grid"][key]
        result = {name: first[name] for name in ("radius", "threshold", "gain")}
        for metric in metrics:
            result[metric] = stat(
                row["density_calibration_grid"][key][metric] for row in rows)
        result["db_crps_better_days"] = sum(
            row["density_calibration_grid"][key]["defined_blind_crps"]
            < row["scores"]["defined_blind"]["crps"] for row in rows)
        result["all_crps_better_days"] = sum(
            row["density_calibration_grid"][key]["all_crps"]
            < row["scores"]["all"]["crps"] for row in rows)
        out["candidates"][key] = result

    out["baseline"] = {
        "density_db_crps": stat(
            row["diagnostics"]["per_channel"]["density"]["defined_blind"]["crps"]
            for row in rows),
        "density_db_spread_skill": stat(
            row["diagnostics"]["per_channel"]["density"]["defined_blind"]
            ["spread_skill"] for row in rows),
        "density_db_coverage90": stat(
            row["diagnostics"]["per_channel"]["density"]["defined_blind"]
            ["coverage"]["90"] for row in rows),
        "defined_blind_crps": stat(
            row["scores"]["defined_blind"]["crps"] for row in rows),
        "defined_blind_coverage90": stat(
            row["scores"]["defined_blind"]["coverage"]["90"] for row in rows),
        "all_crps": stat(row["scores"]["all"]["crps"] for row in rows),
    }
    candidates = out["candidates"]
    out["selection"] = {
        "minimum_defined_blind_crps": min(
            candidates, key=lambda key: candidates[key]["defined_blind_crps"]["mean"]),
        "minimum_density_db_crps": min(
            candidates, key=lambda key: candidates[key]["density_db_crps"]["mean"]),
        "minimum_all_crps": min(
            candidates, key=lambda key: candidates[key]["all_crps"]["mean"]),
    }
    with open(args.out, "w") as handle:
        json.dump(out, handle, indent=2)

    print("candidate       density_crps density_ss density_cov90 db_crps db_cov90 all_crps wins")
    for key, row in sorted(candidates.items(),
                           key=lambda item: item[1]["defined_blind_crps"]["mean"])[:15]:
        m = lambda name: row[name]["mean"]
        print(f"{key:15s} {m('density_db_crps'):.6f} {m('density_db_spread_skill'):.3f} "
              f"{m('density_db_coverage90'):.3f} {m('defined_blind_crps'):.6f} "
              f"{m('defined_blind_coverage90'):.3f} {m('all_crps'):.6f} "
              f"{row['db_crps_better_days']}/{row['all_crps_better_days']}")
    print("baseline", {key: value["mean"] for key, value in out["baseline"].items()})
    print("selection", out["selection"])
    print("[out]", args.out)


if __name__ == "__main__":
    main()
