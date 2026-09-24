"""Aggregate the full-day validation sweep without consulting test metrics."""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(HERE, "outputs", "validation_sweep"))
    ap.add_argument("--out", default=os.path.join(HERE, "outputs", "validation_sweep_summary.json"))
    a = ap.parse_args()
    files = glob.glob(os.path.join(a.root, "atc-*_*.json"))
    grouped = {}
    for path in files:
        base = os.path.basename(path)[:-5]
        day, config = base[:12], base[13:]
        grouped.setdefault(config, {})[day] = json.load(open(path))
    if not grouped:
        raise FileNotFoundError(f"no sweep JSON files under {a.root}")
    bad = {k: len(v) for k, v in grouped.items() if len(v) != 7}
    if bad:
        raise RuntimeError(f"incomplete configs: {bad}")

    out = {"configs": {}}
    for config, days in sorted(grouped.items()):
        rows = list(days.values())
        result = {"days": sorted(days), "failures": sum(r["n_analysis_failures"] for r in rows)}
        for split in ("all", "defined", "defined_blind"):
            for metric in ("rmse", "crps", "spread_skill"):
                x = np.asarray([r["scores"][split][metric] for r in rows])
                result[f"{split}_{metric}"] = {"mean": float(x.mean()),
                                                 "std": float(x.std(ddof=1))}
            x = np.asarray([r["scores"][split]["coverage"]["90"] for r in rows])
            result[f"{split}_coverage90"] = {"mean": float(x.mean()),
                                               "std": float(x.std(ddof=1))}
        for channel in ("density", "vx", "vy", "var"):
            for metric in ("crps", "spread_skill"):
                x = np.asarray([r["diagnostics"]["per_channel"][channel]
                                ["defined_blind"][metric] for r in rows])
                result[f"db_{channel}_{metric}"] = {"mean": float(x.mean()),
                                                      "std": float(x.std(ddof=1))}
        for age in ("blind_1_10", "blind_11_50", "blind_51_plus"):
            for metric in ("crps", "spread_skill"):
                x = np.asarray([r["diagnostics"]["observation_age"][age][metric]
                                for r in rows])
                result[f"age_{age}_{metric}"] = {"mean": float(x.mean()),
                                                   "std": float(x.std(ddof=1))}
        out["configs"][config] = result

    # Proper score is the primary selection rule.  Also expose a constrained winner that
    # cannot buy uncertainty improvement with more than 5% RMSE over residual scale 1.
    base_rmse = out["configs"]["r1"]["all_rmse"]["mean"]
    eligible = {k: v for k, v in out["configs"].items()
                if v["all_rmse"]["mean"] <= 1.05 * base_rmse}
    out["selection"] = {
        "minimum_defined_blind_crps": min(out["configs"],
                                             key=lambda k: out["configs"][k]["defined_blind_crps"]["mean"]),
        "minimum_all_crps": min(out["configs"],
                                  key=lambda k: out["configs"][k]["all_crps"]["mean"]),
        "minimum_db_crps_within_5pct_r1_rmse": min(
            eligible, key=lambda k: eligible[k]["defined_blind_crps"]["mean"]),
        "closest_db_coverage90": min(out["configs"], key=lambda k: abs(
            out["configs"][k]["defined_blind_coverage90"]["mean"] - .9)),
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("config       all_rmse all_crps  db_crps db_ss  db_cov90 longblind_crps")
    for config, r in sorted(out["configs"].items(),
                            key=lambda kv: kv[1]["defined_blind_crps"]["mean"]):
        g = lambda key: r[key]["mean"]
        print(f"{config:12s} {g('all_rmse'):.5f}  {g('all_crps'):.5f}  "
              f"{g('defined_blind_crps'):.5f} {g('defined_blind_spread_skill'):.3f}  "
              f"{g('defined_blind_coverage90'):.3f}   {g('age_blind_51_plus_crps'):.5f}")
    print("selection", out["selection"])
    print("[out]", a.out)


if __name__ == "__main__":
    main()
