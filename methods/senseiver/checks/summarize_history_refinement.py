"""Summarise the matched baseline/history-loss/framewise held-out evaluations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EVAL = ROOT / "methods/senseiver/check_outputs/history_refinement/eval"
RUNS = {
    "Baseline": ROOT / "methods/senseiver/runs/capacity/base32_k16_s123",
    "History loss (w=0.5)": ROOT / "methods/senseiver/runs/history_loss/w05_k16_s123",
    "Framewise": ROOT / "methods/senseiver/runs/framewise/k16_s123",
}
EVAL_FILES = {
    "Baseline": "baseline.json",
    "History loss (w=0.5)": "history_w05.json",
    "Framewise": "framewise.json",
}


def training_best(run_dir: Path) -> dict:
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    rows = [row for row in rows if row.get("valid_mse_blind") is not None]
    best = min(rows, key=lambda row: row["valid_mse_blind"])
    return {
        "epochs": len(rows),
        "best_epoch": best["epoch"],
        "best_valid_blind_mse": best["valid_mse_blind"],
    }


def evaluation(path: Path) -> dict:
    raw = json.loads(path.read_text())
    result = raw["results"]["Senseiver"]
    return {
        "blind_walkable_mse": result["walkable"]["overall_pooled"],
        "blind_walkable_rmse": result["walkable"]["rmse_pooled"],
        "blind_walkable_mse_per_channel": result["walkable"]["per_channel"],
        "blind_allcells_mse": result["allcells"]["overall_pooled"],
        "blind_allcells_rmse": result["allcells"]["rmse_pooled"],
        "full_field_mse": result["full"]["overall_pooled"],
        "full_field_rmse": result["full"]["rmse_pooled"],
        "protocol": raw["protocol"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--out-md", type=Path, default=None)
    args = ap.parse_args()
    out_json = args.out_json or args.eval_dir / "summary.json"
    out_md = args.out_md or args.eval_dir / "summary.md"

    models = {}
    for name, run_dir in RUNS.items():
        models[name] = {
            "run_dir": str(run_dir.relative_to(ROOT)),
            **training_best(run_dir),
            **evaluation(args.eval_dir / EVAL_FILES[name]),
        }

    base = models["Baseline"]["blind_walkable_rmse"]
    for values in models.values():
        values["blind_walkable_rmse_delta_vs_baseline_pct"] = 100.0 * (
            values["blind_walkable_rmse"] / base - 1.0)

    summary = {
        "selection": "best validation blind MSE checkpoint for each 60-epoch run",
        "test_metric": "7-day pooled blind-walkable RMSE, matched per-day trajectories",
        "models": models,
    }
    out_json.write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# History refinement result",
        "",
        "| Model | best epoch | validation blind MSE | test blind-walkable RMSE | vs baseline |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in models.items():
        lines.append(
            f"| {name} | {values['best_epoch']} | "
            f"{values['best_valid_blind_mse']:.8f} | "
            f"{values['blind_walkable_rmse']:.8f} | "
            f"{values['blind_walkable_rmse_delta_vs_baseline_pct']:+.3f}% |"
        )
    lines.extend(["", "Per-channel blind-walkable MSE:", "",
                  "| Model | density | vx | vy | var |", "|---|---:|---:|---:|---:|"])
    for name, values in models.items():
        pc = values["blind_walkable_mse_per_channel"]
        lines.append(f"| {name} | {pc['density']:.8f} | {pc['vx']:.8f} | "
                     f"{pc['vy']:.8f} | {pc['var']:.8f} |")
    out_md.write_text("\n".join(lines) + "\n")
    print(out_json)
    print(out_md)


if __name__ == "__main__":
    main()
