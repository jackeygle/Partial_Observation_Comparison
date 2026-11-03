"""Coverage of the central 50% and 90% prediction intervals (route 2), without touching the
reported results in outputs/full_6m_all92.

    DATA_VARIANT=all92 EVAL_VARIANT=six_allwalk_all92 DINCAE_VAR_TRANSFORM=none \
        python3 -m supervisor_evaluation.checks.coverage_run enkf  --out DIR [--shard k/n]
        python3 -m supervisor_evaluation.checks.coverage_run check --out DIR
        python3 -m supervisor_evaluation.checks.coverage_run unc   --out DIR [--days N --frames N]

enkf   reruns the final EnKF with exactly the arguments of evaluate.py `full`, reading the
       observation exports of the reported run (read-only), and writes only the coverage
       indicators of its members (DIR/coverage/<day>.npz) and its score JSON (DIR/enkf/).
       --smoke-frames N runs the first test day only, up to frame N, and also exports the
       analysis fields so that `check` can score them on the same frames as the reported run.
check  compares the pooled scores of the rerun with the reported run (per-day score JSONs in
       the full run, the smoke export otherwise). The filter is not bit-reproducible, see
       cmd_check, so the check is a tolerance and not an equality.
unc    runs evaluate_uncertainty with the reported EnKF exports (read-only) and the new
       coverage indicators; writes DIR/coverage.json and DIR/raw/uncertainty_unified.json
       and compares the latter's CRPS with the reported one.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate as ev  # noqa: E402

DATA = Path("/scratch/work/zhangx29/data_all92")
REPORTED = ev.HERE / "outputs" / "full_6m_all92"
TOL = 0.02          # full days, per-day scores
TOL_SMOKE = 0.03    # 700 frames of one day


def enkf_args(day: str, out: Path, frames: str, smoke: bool) -> list[str]:
    """The argument list of evaluate.py's EnKF call (mode `full`), outputs redirected."""
    cfg = ev.FINAL_CONFIG["enkf"]
    bank = ev.MODELS / cfg["bank_file"]
    args = [
        "--noise-kind", "residual", "--density-noise-space", "log1p",
        "--log-density-mean-correction", "ensemble",
        "--model-checkpoint", str(ev.MODELS / cfg["file"]),
        "--input-frames", "5", "--bank", str(bank),
        "--scale", str(cfg["q_scale"]), "--temporal-rho", str(cfg["temporal_rho"]),
        "--channel-scales", *map(str, cfg["channel_scales"]),
        "--blind-channel-scales", *map(str, cfg["blind_channel_scales"]),
        "--cross-channel-matrix", *map(str, cfg["cross_channel_matrix"]),
        *(["--bank-native-std"] if cfg.get("bank_native_std") else []),
        "--ensemble", str(cfg["ensemble"]), "--radius", str(cfg["localization_radius"]),
        "--frames", frames, "--warmup", "500",
        "--day", day, "--obs-dir", str(REPORTED / "raw" / "enkf_observations"),
        "--ensemble-crps",
        "--coverage-export", str(out / "coverage" / f"{day}.npz"),
        "--out", str(out / "enkf" / f"{day}.json"),
    ]
    if smoke:          # small enough to keep: the mean, for a bit-for-bit comparison
        args += ["--calibration-export", str(out / "smoke_export" / f"{day}.npz")]
    return args


def cmd_enkf(a) -> None:
    (a.out / "enkf").mkdir(parents=True, exist_ok=True)
    dates = list(ev.TEST_DATES)
    if a.smoke_frames:
        dates = dates[:1]
    elif a.shard:
        k, n = (int(x) for x in a.shard.split("/"))
        dates = dates[k::n]
    frames = str(a.smoke_frames) if a.smoke_frames else "1000000"
    for date in dates:
        ev._python_module("methods.enkf.enkf_opt.experiments.eval_structured_q_gpu",
                          enkf_args(f"atc-{date}", a.out, frames, bool(a.smoke_frames)),
                          ev.ROOT, DATA)
    print(f"[enkf] done {dates}", flush=True)


def _walkable_scores(z, n, w):
    """RMSE, CRPS and mean spread of an export over its first n frames on the walkable cells."""
    e = (z["mean"][:n] - z["truth"][:n])[:, :, w]
    c = (z["ens_mae"][:n] - z["ens_pair"][:n])[:, :, w]
    return {"rmse": float(np.sqrt((e.astype(np.float64) ** 2).mean())),
            "crps": float(c.mean(dtype=np.float64)),
            "spread": float(z["spread"][:n][:, :, w].mean(dtype=np.float64))}


def _flat(d, prefix=""):
    """{"a": {"b": 1.0}} -> {"a.b": 1.0}, numbers only."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[f"{prefix}{k}"] = float(v)
    return out


def cmd_check(a) -> None:
    """The filter is not bit-reproducible: float32 rounding differences on the GPU (3e-7 in the
    first frame) grow through the forecast-analysis loop to visible per-cell differences within
    about 50 frames, also between two runs on one GPU. A rerun is therefore another realisation;
    the pooled scores must agree with the reported run within TOL."""
    ok = True
    smoke = sorted((a.out / "smoke_export").glob("*.npz"))
    if smoke:
        for f in smoke:
            with np.load(f) as s, np.load(REPORTED / "raw" / "enkf_exports" / f.name) as r:
                n, w = len(s["mean"]), s["walkable"]
                ns, nr = _walkable_scores(s, n, w), _walkable_scores(r, n, w)
            for k in ns:
                rel = abs(ns[k] - nr[k]) / abs(nr[k])
                print(f"[check] {f.stem} {n} frames {k:<7} rerun {ns[k]:.5f} reported "
                      f"{nr[k]:.5f}  rel. diff {rel:.2%}", flush=True)
                ok &= rel <= TOL_SMOKE
    else:
        worst = 0.0
        for f in sorted((a.out / "enkf").glob("*.json")):
            s = _flat(json.loads(f.read_text())["scores"])
            r = _flat(json.loads((REPORTED / "raw" / "enkf" / f.name).read_text())["scores"])
            rel = {k: abs(s[k] - r[k]) / abs(r[k])
                   for k in ("all.rmse", "all.crps", "blind.rmse", "blind.crps")}
            k = max(rel, key=rel.get)
            print(f"[check] {f.stem}: all/blind RMSE and CRPS, largest rel. diff "
                  f"{rel[k]:.3%} ({k})", flush=True)
            worst = max(worst, rel[k])
        ok = worst <= TOL
        print(f"[check] largest rel. diff over the days {worst:.3%}", flush=True)
    print(f"[check] {'PASS' if ok else 'FAIL'}", flush=True)
    raise SystemExit(0 if ok else 1)


def cmd_unc(a) -> None:
    os.environ["EVAL_ENKF_RAW"] = str(REPORTED / "raw")
    os.environ["EVAL_COVERAGE_DIR"] = str(a.out / "coverage")
    a.out.mkdir(parents=True, exist_ok=True)
    ev.evaluate_uncertainty(DATA, a.out, a.days, a.frames)
    if not a.days and not a.frames:      # whole test set: must reproduce the reported CRPS
        new = json.loads((a.out / "raw" / "uncertainty_unified.json").read_text())["methods"]
        old = json.loads((REPORTED / "raw" / "uncertainty_unified.json").read_text())["methods"]
        ok = True
        for k in old:
            for sc in ("walkable_blind", "walkable_all"):
                d = abs(new[k]["results"][sc]["crps"] - old[k]["results"][sc]["crps"])
                print(f"[unc] {k:<15} {sc:<15} CRPS {new[k]['results'][sc]['crps']:.6f} "
                      f"reported {old[k]['results'][sc]['crps']:.6f}  |diff| {d:.2e}", flush=True)
                ok &= d < 1e-5
        print(f"[unc] CRPS {'reproduced' if ok else 'DIFFERS FROM THE REPORTED RUN'}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("enkf", "check", "unc"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shard", default=None)
    p.add_argument("--smoke-frames", type=int, default=0)
    p.add_argument("--days", type=int, default=0)
    p.add_argument("--frames", type=int, default=0)
    a = p.parse_args()
    {"enkf": cmd_enkf, "check": cmd_check, "unc": cmd_unc}[a.command](a)


if __name__ == "__main__":
    main()
