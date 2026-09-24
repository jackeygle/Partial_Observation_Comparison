"""Write the overnight validation/held-out-test decision report."""
from __future__ import annotations
import argparse, json, os

HERE = os.path.dirname(os.path.abspath(__file__))


def m(row, key):
    return row[key]["mean"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation", default=os.path.join(HERE, "outputs", "validation_sweep_summary.json"))
    ap.add_argument("--test", default=os.path.join(HERE, "outputs", "selected_test_summary.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "outputs", "optimization_report.md"))
    a = ap.parse_args()
    val, test = json.load(open(a.validation)), json.load(open(a.test))
    primary = val["selection"]["minimum_db_crps_within_5pct_r1_rmse"]
    lines = ["# Structured-Q EnKF optimization report", "",
             "Hyperparameters were selected exclusively on seven complete validation days. "
             "The selected candidates were then evaluated on seven complete held-out test days.", "",
             "## Validation sweep", "",
             "| config | all RMSE | all CRPS | defined-blind CRPS | DB spread/skill | DB coverage90 |",
             "|---|---:|---:|---:|---:|---:|"]
    for cfg, r in sorted(val["configs"].items(), key=lambda kv: m(kv[1], "defined_blind_crps")):
        lines.append(f"| {cfg} | {m(r,'all_rmse'):.5f} | {m(r,'all_crps'):.5f} | "
                     f"{m(r,'defined_blind_crps'):.5f} | {m(r,'defined_blind_spread_skill'):.3f} | "
                     f"{m(r,'defined_blind_coverage90'):.3f} |")
    lines += ["", "Validation-only selections:", "", "```json",
              json.dumps(val["selection"], indent=2), "```", "", "## Locked held-out test", "",
              "| config | all RMSE | all CRPS | defined-blind CRPS | DB spread/skill | DB coverage90 |",
              "|---|---:|---:|---:|---:|---:|"]
    for cfg, r in sorted(test["configs"].items(), key=lambda kv: m(kv[1], "defined_blind_crps")):
        lines.append(f"| {cfg} | {m(r,'all_rmse'):.5f} | {m(r,'all_crps'):.5f} | "
                     f"{m(r,'defined_blind_crps'):.5f} | {m(r,'defined_blind_spread_skill'):.3f} | "
                     f"{m(r,'defined_blind_coverage90'):.3f} |")
    r = test["configs"][primary]
    lines += ["", "## Recommended configuration", "",
              f"Primary: **{primary}**, selected by minimum validation defined-blind CRPS "
              "subject to no more than 5% RMSE degradation relative to residual scale 1.", "",
              f"Held-out test: all RMSE {m(r,'all_rmse'):.5f}, all CRPS {m(r,'all_crps'):.5f}, "
              f"defined-blind CRPS {m(r,'defined_blind_crps'):.5f}, spread/skill "
              f"{m(r,'defined_blind_spread_skill'):.3f}, coverage90 {m(r,'defined_blind_coverage90'):.3f}.", "",
              "## Remaining limitation", "",
              "If defined-blind spread/skill and 90% coverage remain materially below 1.0 and 0.9, "
              "a single global Q scale is insufficient. The next method change should condition Q "
              "on forecast state / observation age or learn channel-wise scales on validation, not "
              "continue increasing a global test-tuned multiplier.", ""]
    with open(a.out, "w") as f:
        f.write("\n".join(lines))
    print("[out]", a.out)


if __name__ == "__main__":
    main()
