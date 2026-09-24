#!/usr/bin/env python3
"""One-command, frozen evaluation entry point for the thesis comparison.

The supervisor-facing interface deliberately has no YAML configuration.  Every
scientific choice is frozen in FINAL_CONFIG below; the command line only selects
the data location, output location, device, methods, and smoke/full extent.

This file is also the packaging authority: ``prepare`` copies exactly the
selected checkpoints into ``models/`` and records their hashes.  Evaluation must
never silently fall back to another training run or checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODELS = HERE / "models"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CHANNELS = ("density", "vx", "vy", "variance")
STATE_SHAPE = (4, 36, 12)
TEST_DATES = (
    "20130811", "20130818", "20130825", "20130901",
    "20130915", "20130922", "20130929",
)

# No experiment choice is read from a second config file.  Keep this small and
# explicit so a reviewer can audit the complete comparison at the top of the
# only executable they need to run.
FINAL_CONFIG: dict[str, dict[str, Any]] = {
    "senseiver_a": {
        "label": "Senseiver-A",
        "source": "methods/senseiver/runs/senseiver_A/best.pt",
        "file": "senseiver_A.pt",
        "temporal_context": "t",
        "causal": True,
        "uncertainty": False,
    },
    "senseiver_g": {
        "label": "Senseiver-G Temporal (ours)",
        "source": "methods/senseiver/runs/capacity/base32_k16_s123/best.pt",
        "file": "senseiver_G_k16.pt",
        "temporal_context": "t-15..t",
        "causal": True,
        "uncertainty": False,
    },
    "dincae": {
        "label": "DINCAE",
        "source": "methods/dincae/runs/dincae_ff/ckpt_00060.pt",
        "file": "dincae_epoch60.pt",
        "temporal_context": "t-1,t,t+1",
        "causal": False,
        "uncertainty": True,
    },
    "varnet_mse": {
        "label": "4DVarNet MSE",
        "source": "methods/varnet/runs/varnet_mse5_h96_s3/ckpt_00080.pt",
        "file": "varnet_mse_s3_epoch80.pt",
        "temporal_context": "200-frame window",
        "causal": False,
        "uncertainty": False,
    },
    "varnet_aughead": {
        "label": "4DVarNet aughead_obs (single)",
        "run_dir": "methods/varnet/runs/varnet_aughead_obs_h96_s0",
        "selection": "select_valid.json",
        "file": "varnet_aughead_obs_s0.pt",
        "temporal_context": "200-frame window",
        "causal": False,
        "uncertainty": True,
    },
    "enkf": {
        "label": "Localized EnKF (100 members)",
        "source": "methods/enkf/runs/pedpred3_5to5_clip_s0/dyn150_best.pt",
        "file": "pedpred3_5to5_epoch38.pt",
        "bank_source": "methods/enkf/enkf_opt/experiments/outputs/pedpred3_5to5_train_residual_q_8192.npz",
        "bank_file": "enkf_residual_q_bank.npz",
        "temporal_context": "t-4..t analysis",
        "causal": True,
        "uncertainty": True,
        "ensemble": 100,
        "warmup": 500,
        "localization_radius": 7,
        "q_scale": 1.5,
        "temporal_rho": 0.5,
        "channel_scales": (2.0, 1.0, 1.0, 0.75),
        "blind_channel_scales": (1.0, 1.25, 1.4, 0.9333333),
        "cross_channel_matrix": (1.0, 0.5, 0.5, 0.1,
                                 0.5, 1.0, 0.5, 0.1,
                                 0.5, 0.5, 1.0, 0.1,
                                 0.1, 0.1, 0.1, 1.0),
        "inflation": 1.02,
    },
}

ACCURACY_METHODS = ("senseiver_a", "senseiver_g", "dincae", "varnet_mse",
                    "varnet_aughead", "enkf")
UNCERTAINTY_METHODS = ("varnet_aughead", "dincae", "enkf")


def sha256(path: Path, block: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def _aughead_source() -> Path:
    cfg = FINAL_CONFIG["varnet_aughead"]
    run = ROOT / cfg["run_dir"]
    selection = run / cfg["selection"]
    if not selection.exists():
        raise FileNotFoundError(
            f"{selection} is missing. Select aughead_obs on the validation split before packaging."
        )
    picked = json.loads(selection.read_text())["selected_ckpt"]
    return run / picked


def source_files() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for name, cfg in FINAL_CONFIG.items():
        if name == "varnet_aughead":
            out[cfg["file"]] = _aughead_source()
        else:
            out[cfg["file"]] = ROOT / cfg["source"]
        if name == "enkf":
            out[cfg["bank_file"]] = ROOT / cfg["bank_source"]
    out["dincae_state_stats.npz"] = ROOT / "methods/dincae/artifacts/state_stats.npz"
    return out


def prepare_models(allow_incomplete: bool = False) -> int:
    MODELS.mkdir(parents=True, exist_ok=True)
    try:
        sources = source_files()
    except FileNotFoundError as exc:
        if not allow_incomplete:
            raise SystemExit(str(exc))
        print(f"[incomplete] {exc}")
        sources = {}
        for name, cfg in FINAL_CONFIG.items():
            if name != "varnet_aughead":
                sources[cfg["file"]] = ROOT / cfg["source"]
            if name == "enkf":
                sources[cfg["bank_file"]] = ROOT / cfg["bank_source"]
        sources["dincae_state_stats.npz"] = ROOT / "methods/dincae/artifacts/state_stats.npz"

    manifest: dict[str, Any] = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "files": {}}
    missing = 0
    for destination, source in sources.items():
        if not source.exists():
            print(f"[missing] {source}")
            missing += 1
            continue
        target = MODELS / destination
        if not target.exists() or sha256(target) != sha256(source):
            print(f"[copy] {source.relative_to(ROOT)} -> models/{destination}")
            shutil.copy2(source, target)
        manifest["files"][destination] = {
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
            "source": str(source.relative_to(ROOT)),
        }
    (MODELS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[out] {MODELS / 'manifest.json'}")
    if missing and not allow_incomplete:
        raise SystemExit(f"{missing} final artifacts are missing")
    return 1 if missing else 0


def _load_manifest() -> dict[str, Any]:
    path = MODELS / "manifest.json"
    if not path.exists():
        raise SystemExit(f"{path} is missing; run `evaluate.py prepare` first")
    return json.loads(path.read_text())


def _resolve_grid_cache(data_root: Path) -> Path:
    direct = data_root / "grid_cache"
    return direct if direct.is_dir() else data_root


def verify(data_root: Path, full_checksum: bool, require_complete: bool = True) -> dict[str, Any]:
    report: dict[str, Any] = {
        "ok": True,
        "repo_root": str(ROOT),
        "data_root": str(data_root.resolve()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {}, "models": {}, "data": {}, "errors": [], "warnings": [],
    }
    for package in ("numpy", "scipy", "h5py", "torch", "matplotlib"):
        spec = importlib.util.find_spec(package)
        report["packages"][package] = bool(spec)
        if spec is None:
            report["errors"].append(f"missing Python package: {package}")

    try:
        manifest = _load_manifest()
    except SystemExit as exc:
        report["errors"].append(str(exc)); manifest = {"files": {}}
    expected = {cfg["file"] for cfg in FINAL_CONFIG.values()}
    expected |= {FINAL_CONFIG["enkf"]["bank_file"], "dincae_state_stats.npz"}
    for filename in sorted(expected):
        path = MODELS / filename
        entry: dict[str, Any] = {"exists": path.exists()}
        if path.exists():
            entry["bytes"] = path.stat().st_size
            recorded = manifest.get("files", {}).get(filename, {}).get("sha256")
            if full_checksum or recorded:
                entry["sha256"] = sha256(path)
                entry["hash_ok"] = recorded is not None and entry["sha256"] == recorded
                if not entry["hash_ok"]:
                    report["errors"].append(f"model hash mismatch or unrecorded: {filename}")
        elif require_complete:
            report["errors"].append(f"missing packaged artifact: {filename}")
        else:
            report["warnings"].append(f"missing unfinished artifact: {filename}")
        report["models"][filename] = entry

    bank_path = MODELS / FINAL_CONFIG["enkf"]["bank_file"]
    dynamics_path = MODELS / FINAL_CONFIG["enkf"]["file"]
    if bank_path.exists() and dynamics_path.exists() and importlib.util.find_spec("numpy"):
        try:
            import numpy as np
            with np.load(bank_path, allow_pickle=False) as bank:
                meta = json.loads(str(bank["metadata"]))
            expected_model_hash = sha256(dynamics_path)
            report["models"][bank_path.name]["input_frames"] = meta.get("input_frames")
            report["models"][bank_path.name]["dynamics_hash_ok"] = (
                meta.get("model_sha256") == expected_model_hash)
            if meta.get("input_frames") != 5 or meta.get("model_sha256") != expected_model_hash:
                report["errors"].append(
                    "EnKF residual bank was not generated by the packaged 5-frame PedPred3")
        except Exception as exc:
            report["errors"].append(f"cannot verify EnKF residual-bank provenance: {exc}")

    cache = _resolve_grid_cache(data_root)
    files = sorted(cache.glob("atc-*_corridor_1.0s.h5"))
    report["data"]["grid_cache"] = str(cache)
    report["data"]["n_h5"] = len(files)
    if not files:
        report["errors"].append(f"no grid-cache HDF5 files under {cache}")
    else:
        try:
            import h5py
            shapes = {}
            for fp in files:
                with h5py.File(fp, "r") as h5:
                    if "grid" not in h5 or "time" not in h5:
                        report["errors"].append(f"{fp.name}: missing grid/time dataset")
                        continue
                    shape = tuple(h5["grid"].shape)
                    shapes[fp.name] = shape
                    if shape[1:] != STATE_SHAPE:
                        report["errors"].append(f"{fp.name}: grid shape {shape}, expected (T,{STATE_SHAPE})")
                    if len(h5["time"]) != shape[0]:
                        report["errors"].append(f"{fp.name}: time/grid length mismatch")
            report["data"]["shapes"] = shapes
            names = " ".join(shapes)
            missing_test = [d for d in TEST_DATES if d not in names]
            if missing_test:
                report["errors"].append(f"missing test dates: {missing_test}")
            # The methods read the test list; the EnKF loop and reporting use TEST_DATES.
            # They must name exactly the same days, or the methods are scored on
            # different days without any error.
            lst = data_root / "sunday_atc_test.lst"
            if lst.exists():
                listed = sorted(ln.strip().rsplit("atc-", 1)[-1].split(".")[0]
                                for ln in lst.read_text().splitlines() if ln.strip())
                report["data"]["test_list"] = listed
                if listed != sorted(TEST_DATES):
                    report["errors"].append(
                        f"{lst.name} lists {listed}, but TEST_DATES in evaluate.py is "
                        f"{sorted(TEST_DATES)}; make them identical")
            else:
                report["errors"].append(f"missing split list {lst}")
        except Exception as exc:  # report cleanly instead of a supervisor-facing traceback
            report["errors"].append(f"HDF5 verification failed: {exc}")

    report["ok"] = not report["errors"]
    return report


def write_verification(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "verification_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for warning in report["warnings"]:
        print(f"[warn] {warning}")
    for error in report["errors"]:
        print(f"[error] {error}")
    print(f"[verify] {'PASS' if report['ok'] else 'FAIL'} -> {path}")


def _python_module(module: str, args: list[str], cwd: Path, data_root: Path) -> None:
    """Run a legacy evaluator after overriding the data root before its imports."""
    code = (
        "from crowdcore import config; "
        f"config.CFG['data']['root']={str(data_root.resolve())!r}; "
        "import runpy,sys; "
        f"sys.argv={[module, *args]!r}; "
        f"runpy.run_module({module!r},run_name='__main__')"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONSAFEPATH"] = "1"
    print(f"[run] {module} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env, check=True)


def run_available(mode: str, data_root: Path, output_dir: Path, methods: set[str]) -> None:
    """Run final neural-model evaluators; EnKF is gated on its new residual bank.

    This orchestration intentionally calls the already-tested method evaluators.
    It does not duplicate their scientific logic in a second implementation.
    """
    days = "1" if mode == "smoke" else "0"
    frames = "512" if mode == "smoke" else "0"
    # Some legacy evaluators need their own project directory as cwd.  Absolute
    # paths prevent their outputs from silently landing under that cwd.
    raw = (output_dir / "raw").resolve()
    raw.mkdir(parents=True, exist_ok=True)

    if "senseiver_a" in methods:
        _python_module("compare.compare5", [
            "--senseiver", str(MODELS / "senseiver_A.pt"), "--arms", "",
            "--no-with-ensemble", "--days", days, "--frames", frames,
            "--out", str(raw / "senseiver_a_accuracy.json"),
        ], ROOT, data_root)
    if "senseiver_g" in methods:
        _python_module("compare.compare5", [
            "--senseiver", str(MODELS / "senseiver_G_k16.pt"), "--arms", "",
            "--no-with-ensemble", "--days", days, "--frames", frames,
            "--out", str(raw / "senseiver_g_accuracy.json"),
        ], ROOT, data_root)
    if "dincae" in methods:
        dargs = ["--run-dir", str(MODELS), "--ckpt-glob", str(MODELS / "dincae_epoch60.pt"),
                 "--split", "test", "--days", days, "--frames", frames,
                 "--out", str(raw / "dincae_accuracy")]
        _python_module("methods.dincae.checks.evaluate", dargs, ROOT, data_root)
    if "varnet_mse" in methods:
        args = ["--senseiver", str(MODELS / "senseiver_A.pt"),
                "--varnet-checkpoints", f"MSE={MODELS / 'varnet_mse_s3_epoch80.pt'}",
                "--arms", "", "--no-with-ensemble",
                "--days", days, "--frames", frames,
                "--out", str(raw / "varnet_mse_accuracy.json")]
        _python_module("compare.compare5", args, ROOT, data_root)
    if "varnet_aughead" in methods:
        # Score the same single checkpoint's mean on the common physical-space
        # reconstruction protocol used by the other deterministic methods.
        args = ["--senseiver", str(MODELS / "senseiver_A.pt"),
                "--varnet-checkpoints", f"aughead_obs={MODELS / 'varnet_aughead_obs_s0.pt'}",
                "--arms", "", "--no-with-ensemble",
                "--days", days, "--frames", frames,
                "--out", str(raw / "varnet_aughead_accuracy.json")]
        _python_module("compare.compare5", args, ROOT, data_root)
        # Its uncertainty is scored by `evaluate_uncertainty`, together with the others.
    if "enkf" in methods:
        bank = MODELS / FINAL_CONFIG["enkf"]["bank_file"]
        if not bank.exists():
            raise SystemExit(
                "final EnKF residual bank is missing. It must be rebuilt with the packaged "
                "5->5 PedPred3; the old apt-ibex residual bank is intentionally rejected."
            )
        obs_dir = raw / "enkf_observations"
        enkf_dir = raw / "enkf"
        export_dir = raw / "enkf_exports"
        obs_dir.mkdir(exist_ok=True); enkf_dir.mkdir(exist_ok=True); export_dir.mkdir(exist_ok=True)
        export_args = [
            "--split", "test", "--frames", "600" if mode == "smoke" else "0",
            "--outdir", str(obs_dir), "--obs-every-k", "1",
        ]
        if mode == "smoke":
            export_args += ["--only", "atc-20130811"]
        _python_module("methods.enkf.checks.export_obs_for_enkf", export_args, ROOT, data_root)
        cfg = FINAL_CONFIG["enkf"]
        dates = TEST_DATES[:1] if mode == "smoke" else TEST_DATES
        for date in dates:
            day = f"atc-{date}"
            args = [
                "--noise-kind", "residual", "--density-noise-space", "log1p",
                "--log-density-mean-correction", "ensemble",
                "--model-checkpoint", str(MODELS / cfg["file"]),
                "--input-frames", "5", "--bank", str(bank),
                "--scale", str(cfg["q_scale"]), "--temporal-rho", str(cfg["temporal_rho"]),
                "--channel-scales", *map(str, cfg["channel_scales"]),
                "--blind-channel-scales", *map(str, cfg["blind_channel_scales"]),
                "--cross-channel-matrix", *map(str, cfg["cross_channel_matrix"]),
                "--ensemble", str(cfg["ensemble"]), "--radius", str(cfg["localization_radius"]),
                "--frames", "600" if mode == "smoke" else "1000000",
                "--warmup", "100" if mode == "smoke" else "500",
                "--day", day, "--obs-dir", str(obs_dir),
                "--calibration-export", str(export_dir / f"{day}.npz"),
                "--out", str(enkf_dir / f"{day}.json"),
            ]
            _python_module("methods.enkf.enkf_opt.experiments.eval_structured_q_gpu",
                           args, ROOT, data_root)


def collect(output_dir: Path) -> None:
    """Collect stable fields from raw evaluator JSONs into supervisor-facing CSVs."""
    raw = output_dir / "raw"
    accuracy_rows: list[dict[str, Any]] = []
    for key, filename in (("senseiver_a", "senseiver_a_accuracy.json"),
                          ("senseiver_g", "senseiver_g_accuracy.json")):
        p = raw / filename
        if not p.exists():
            continue
        x = json.loads(p.read_text()).get("results", {}).get("Senseiver", {})
        if not x:
            continue
        bl, al = x["walkable"], x["walkable_full"]
        pc = bl["per_channel"]
        accuracy_rows.append({
            "method": FINAL_CONFIG[key]["label"],
            "blind_rmse": bl["rmse_pooled"],
            "all_walkable_rmse": al["rmse_pooled"],
            **{f"{c}_rmse": float(pc["var" if c == "variance" else c]) ** 0.5
               for c in CHANNELS},
        })
    dp = raw / "dincae_accuracy" / "dincae_metrics_test.json"
    if dp.exists():
        x = json.loads(dp.read_text())
        pc = x["walkable_blind_mse"]
        accuracy_rows.append({
            "method": FINAL_CONFIG["dincae"]["label"],
            "blind_rmse": float(x["walkable_blind_mse_all_channels"]) ** 0.5,
            "all_walkable_rmse": float(x["walkable_all_mse_all_channels"]) ** 0.5,
            **{f"{c}_rmse": float(pc["var" if c == "variance" else c]) ** 0.5
               for c in CHANNELS},
        })
    p = raw / "varnet_mse_accuracy.json"
    if p.exists():
        x = json.loads(p.read_text())
        row = x.get("results", {}).get("4DVarNet MSE", {})
        if row:
            bl, al = row["walkable"], row["walkable_full"]
            accuracy_rows.append({
                "method": FINAL_CONFIG["varnet_mse"]["label"],
                "blind_rmse": bl["rmse_pooled"], "all_walkable_rmse": al["rmse_pooled"],
                **{f"{c}_rmse": float(bl["per_channel"]["var" if c == "variance" else c]) ** 0.5
                   for c in CHANNELS},
            })
    p = raw / "varnet_aughead_accuracy.json"
    if p.exists():
        x = json.loads(p.read_text()).get("results", {}).get("4DVarNet aughead_obs", {})
        if x:
            bl, al = x["walkable"], x["walkable_full"]
            accuracy_rows.append({
                "method": FINAL_CONFIG["varnet_aughead"]["label"],
                "blind_rmse": bl["rmse_pooled"], "all_walkable_rmse": al["rmse_pooled"],
                **{f"{c}_rmse": float(bl["per_channel"]["var" if c == "variance" else c]) ** 0.5
                   for c in CHANNELS},
            })
    enkf_exports = sorted((raw / "enkf_exports").glob("atc-*.npz"))
    if enkf_exports:
        import numpy as np
        se = np.zeros(4, dtype=np.float64); count = np.zeros(4, dtype=np.int64)
        se_all = 0.0; count_all = 0
        for fp in enkf_exports:
            with np.load(fp) as z:
                mu, truth = z["mean"], z["truth"]
                observed, walk = z["observed"].astype(bool), z["walkable"].astype(bool)
                walk4 = np.broadcast_to(walk[None, None], mu.shape)
                delta = np.square(mu - truth)
                se_all += delta[walk4].sum(dtype=np.float64)
                count_all += int(walk4.sum())
                for c in range(4):
                    sel = (~observed) & walk[None]
                    se[c] += delta[:, c][sel].sum(dtype=np.float64)
                    count[c] += int(sel.sum())
        pc = se / np.maximum(count, 1)
        accuracy_rows.append({
            "method": FINAL_CONFIG["enkf"]["label"],
            "blind_rmse": float(np.sqrt(se.sum() / count.sum())),
            "all_walkable_rmse": float(np.sqrt(se_all / max(count_all, 1))),
            **{f"{c}_rmse": float(np.sqrt(pc[i])) for i, c in enumerate(CHANNELS)},
        })
    if accuracy_rows:
        fields = list(accuracy_rows[0])
        with (output_dir / "accuracy.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(accuracy_rows)

    uncertainty_rows = []
    calibration_rows = []
    up = raw / "uncertainty_unified.json"
    if up.exists():
        doc = json.loads(up.read_text())
        for key in UNCERTAINTY_METHODS:
            r = doc["methods"][key]["results"]
            metric, base = r["walkable_blind"], r["walkable_blind_constant_sigma_baseline"]
            uncertainty_rows.append({
                "method": FINAL_CONFIG[key]["label"], "rmse": metric["rmse"],
                "crps": metric["crps"], "crps_null": base["crps"],
                "crps_skill": 1.0 - metric["crps"] / base["crps"],
                "spread_skill": metric["spread_skill"],
                "coverage90": metric["coverage"]["90"],
            })
            if len(doc["methods"][key].get("per_day") or []) >= 2:   # no interval from one day
                ci = _bootstrap_days(doc["methods"][key]["per_day"])
                for m in ("crps_skill", "spread_skill"):
                    uncertainty_rows[-1][f"{m}_lo"], uncertainty_rows[-1][f"{m}_hi"] = ci[m]
            for nominal in range(10, 100, 10):
                calibration_rows.append({
                    "method": FINAL_CONFIG[key]["label"], "nominal": nominal / 100,
                    "empirical": metric["coverage"][str(nominal)],
                })
    channel_rows = []
    if up.exists():
        for key in UNCERTAINTY_METHODS:
            r = doc["methods"][key]["results"]
            for c in CHANNELS:
                metric, base = r[f"channel_{c}"], r[f"channel_{c}_constant_sigma_baseline"]
                channel_rows.append({
                    "method": FINAL_CONFIG[key]["label"], "channel": c, "rmse": metric["rmse"],
                    "crps": metric["crps"], "crps_null": base["crps"],
                    "crps_skill": 1.0 - metric["crps"] / base["crps"],
                    "spread_skill": metric["spread_skill"],
                    "coverage90": metric["coverage"]["90"],
                })
        with (output_dir / "uncertainty_by_channel.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, list(channel_rows[0])); w.writeheader(); w.writerows(channel_rows)
    if uncertainty_rows:
        with (output_dir / "uncertainty.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, list(uncertainty_rows[0])); w.writeheader(); w.writerows(uncertainty_rows)
    if calibration_rows:
        with (output_dir / "calibration.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, list(calibration_rows[0])); w.writeheader(); w.writerows(calibration_rows)
    timing_rows = []
    import numpy as np
    for key, filename in (("senseiver_a", "senseiver_a_accuracy.json"),
                          ("senseiver_g", "senseiver_g_accuracy.json")):
        p = raw / filename
        if p.exists():
            x = json.loads(p.read_text())
            values = [float(v) for v in x.get("inference", {}).get(
                "Senseiver", {}).get("per_day_ms", [])]
            if values:
                timing_rows.append({"method": FINAL_CONFIG[key]["label"],
                                    "median_ms_per_frame": float(np.median(values)),
                                    "p95_ms_per_frame": float(np.percentile(values, 95)),
                                    "gpu": ""})
    dp = raw / "dincae_accuracy" / "dincae_metrics_test.json"
    if dp.exists():
        x = json.loads(dp.read_text())
        values = [float(d["inference_ms_per_frame"]) for d in x.get("per_day", [])]
        if values:
            timing_rows.append({"method": FINAL_CONFIG["dincae"]["label"],
                                "median_ms_per_frame": float(np.median(values)),
                                "p95_ms_per_frame": float(np.percentile(values, 95)),
                                "gpu": x.get("gpu", "")})
    vp = raw / "varnet_mse_accuracy.json"
    if vp.exists():
        x = json.loads(vp.read_text()).get("inference", {}).get("4DVarNet MSE", {})
        values = [float(v) for v in x.get("per_day_ms", [])]
        if values:
            timing_rows.append({"method": FINAL_CONFIG["varnet_mse"]["label"],
                                "median_ms_per_frame": float(np.median(values)),
                                "p95_ms_per_frame": float(np.percentile(values, 95)),
                                    "gpu": ""})
    vp = raw / "varnet_aughead_accuracy.json"
    if vp.exists():
        x = json.loads(vp.read_text()).get("inference", {}).get("4DVarNet aughead_obs", {})
        values = [float(v) for v in x.get("per_day_ms", [])]
        if values:
            timing_rows.append({"method": FINAL_CONFIG["varnet_aughead"]["label"],
                                "median_ms_per_frame": float(np.median(values)),
                                "p95_ms_per_frame": float(np.percentile(values, 95)),
                                "gpu": ""})
    enkf_jsons = sorted((raw / "enkf").glob("atc-*.json"))
    if enkf_jsons:
        values, gpu = [], ""
        for p in enkf_jsons:
            x = json.loads(p.read_text()); gpu = x.get("gpu", gpu)
            values.append(1000 * float(x["seconds"]) / int(x["frames_used"]))
        timing_rows.append({"method": FINAL_CONFIG["enkf"]["label"],
                            "median_ms_per_frame": float(np.median(values)),
                            "p95_ms_per_frame": float(np.percentile(values, 95)),
                            "gpu": gpu})
    if timing_rows:
        with (output_dir / "inference_time.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, list(timing_rows[0])); w.writeheader(); w.writerows(timing_rows)
    print(f"[collect] {len(accuracy_rows)} accuracy rows, {len(uncertainty_rows)} uncertainty rows")


#: Figure-facing names. Run labels (FINAL_CONFIG[...]["label"]) stay in the CSVs.
SHORT_LABEL = {
    "senseiver_a": "Senseiver-A",
    "senseiver_g": "Senseiver-G (ours)",
    "dincae": "DINCAE",
    "varnet_mse": "4DVarNet",
    "varnet_aughead": "4DVarNet (aug. head)",
    "enkf": "EnKF",
}
CHANNEL_TEX = {"density": "Density", "vx": "$v_x$", "vy": "$v_y$", "variance": "Velocity variance"}


def _bootstrap_days(per_day: list[dict[str, Any]], n_boot: int = 10000,
                    seed: int = 0) -> dict[str, tuple[float, float]]:
    """95% percentile intervals over test days for the pooled uncertainty ratios.

    Days are resampled with replacement and the pooled ratios recomputed from the
    per-day sums, so the interval reflects day-to-day variation (7 days), not the
    millions of correlated cells within a day.
    """
    import numpy as np
    n = np.asarray([d["n"] for d in per_day], dtype=np.float64)
    crps = np.asarray([d["crps"] for d in per_day]) * n
    null = np.asarray([d["crps_null"] for d in per_day]) * n
    sig = np.asarray([d["sigma_mean"] for d in per_day]) * n
    se = np.asarray([d["rmse"] for d in per_day]) ** 2 * n
    idx = np.random.default_rng(seed).integers(0, len(n), (n_boot, len(n)))
    skill = 1.0 - crps[idx].sum(1) / null[idx].sum(1)
    ratio = sig[idx].sum(1) / n[idx].sum(1) / np.sqrt(se[idx].sum(1) / n[idx].sum(1))
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return {"crps_skill": q(skill), "spread_skill": q(ratio)}


def plot_summary(output_dir: Path) -> None:
    """Thesis figures from the collected CSVs, at the paper's text width (PNG, 300 dpi).

    Style comes from compare/plotstyle.py: one colour per method across every
    figure, one colormap per quantity, three font sizes. Figure text carries no
    run metadata; that goes to figures/captions.md.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from compare import plotstyle as ps

    ps.use(dpi=300)
    fig_dir = output_dir / "figures"; fig_dir.mkdir(exist_ok=True)
    # Superseded file names from the previous figure set.
    for old in ("channel_rmse", "crps_skill", "spread_skill", "calibration_curve",
                "enkf_reconstruction_error", "enkf_reconstruction", "accuracy_latency",
                "enkf_uncertainty_maps", "reconstruction_example", "density_uncertainty_example",
                "uncertainty_by_channel"):
        (fig_dir / f"{old}.png").unlink(missing_ok=True)
    short = {FINAL_CONFIG[k]["label"]: SHORT_LABEL[k] for k in FINAL_CONFIG}
    captions: list[str] = []

    def save(fig, stem: str) -> None:
        fig.savefig(fig_dir / f"{stem}.png")
        plt.close(fig)

    def panel(ax, letter: str) -> None:
        ax.set_title(f"({letter})", loc="left", fontweight="bold", fontsize=ps.FS_LABEL)

    # ---- 1. Reconstruction accuracy: grouped bars, pooled + per channel ----------
    acc_path = output_dir / "accuracy.csv"
    if acc_path.exists():
        rows = list(csv.DictReader(acc_path.open()))
        cols = [("blind_rmse", "All channels")] + [(f"{c}_rmse", CHANNEL_TEX[c]) for c in CHANNELS]
        fig, ax = ps.figure(rows_h=2.6, constrained_layout=True)
        n, x = len(rows), np.arange(len(cols))
        w = 0.8 / n
        for i, r in enumerate(rows):
            ax.bar(x + (i - (n - 1) / 2) * w, [float(r[k]) for k, _ in cols], w,
                   color=ps.method_color(r["method"]), edgecolor="white", linewidth=0.6,
                   label=short.get(r["method"], r["method"]))
        ax.set_xticks(x, [t for _, t in cols])
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel("RMSE (lower is better)")
        fig.legend(loc="outside upper center", ncol=3, handlelength=1.2, columnspacing=1.5)
        save(fig, "accuracy_rmse")
        captions.append(
            "**accuracy_rmse** — Reconstruction RMSE on walkable cells that are not observed "
            "at that frame, pooled over the four state channels (left group) and per channel, "
            f"over the {len(TEST_DATES)} held-out test days. Lower is better; exact values in "
            "accuracy_table.tex.")
        # The same numbers as a LaTeX table (best per column in bold).
        head = ["Method", "All", "All walkable"] + [CHANNEL_TEX[c] for c in CHANNELS]
        keys = ["blind_rmse", "all_walkable_rmse"] + [f"{c}_rmse" for c in CHANNELS]
        best = {k: min(float(r[k]) for r in rows) for k in keys}
        lines = [r"\begin{tabular}{l" + "r" * len(keys) + "}", r"\toprule",
                 " & ".join(head) + r" \\", r"\midrule"]
        for r in rows:
            cells = [short.get(r["method"], r["method"])]
            for k in keys:
                v = f"{float(r[k]):.3f}"
                cells.append(r"\textbf{" + v + "}" if float(r[k]) == best[k] else v)
            lines.append(" & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (output_dir / "accuracy_table.tex").write_text("\n".join(lines) + "\n")

    # ---- 2. Uncertainty: pooled ("All") and per channel in one 2x2 figure --------
    unc_path = output_dir / "uncertainty.csv"
    ch_path = output_dir / "uncertainty_by_channel.csv"
    if unc_path.exists() and ch_path.exists():
        urows = list(csv.DictReader(unc_path.open()))
        chrows = list(csv.DictReader(ch_path.open()))
        names = [r["method"] for r in urows]
        pooled = {r["method"]: r for r in urows}
        per_ch = {(r["method"], r["channel"]): r for r in chrows}
        groups = ["All"] + list(CHANNELS)
        glabels = ["All"] + [CHANNEL_TEX[c].replace("Velocity variance", "Vel. var.") for c in CHANNELS]
        has_ci = all(r.get("crps_skill_lo") not in (None, "") for r in urows)

        fig, (ax_c, ax_d) = ps.figure(ncols=2, rows_h=2.6, constrained_layout=True)
        x = np.arange(len(groups)); w = 0.8 / len(names)
        for ax, key, ylabel, ref, letter in (
                (ax_c, "crps_skill", "CRPS skill (higher is better)", 0.0, "a"),
                (ax_d, "spread_skill", "Spread / RMSE (1 = right size)", 1.0, "b")):
            for i, m in enumerate(names):
                v = np.array([float(pooled[m][key])] +
                             [float(per_ch[(m, c)][key]) for c in CHANNELS])
                xs = x + (i - (len(names) - 1) / 2) * w
                ax.bar(xs, v, w, color=ps.method_color(m), edgecolor="white", linewidth=0.6,
                       label=short.get(m, m))
                if has_ci:                              # day-bootstrap CI, pooled only
                    lo, hi = float(pooled[m][f"{key}_lo"]), float(pooled[m][f"{key}_hi"])
                    ax.errorbar(xs[0], v[0], yerr=[[max(v[0] - lo, 0.0)], [max(hi - v[0], 0.0)]],
                                fmt="none",
                                ecolor=ps.INK, elinewidth=0.8, capsize=2)
            ax.axhline(ref, ls="--" if ref else "-", lw=0.8,
                       color=ps.INK_MUTED if ref else ps.AXIS)
            ax.axvline(0.5, color=ps.RULE, lw=0.8)      # separates pooled from channels
            ax.set_xticks(x, glabels, fontsize=ps.FS_TICK)
            ax.tick_params(axis="x", length=0)
            ax.set_ylabel(ylabel)
            panel(ax, letter)
        fig.legend(*ax_c.get_legend_handles_labels(), loc="outside lower center",
                   ncol=len(names), handlelength=1.8)
        save(fig, "uncertainty_summary")
        captions.append(
            "**uncertainty_summary** — Predictive uncertainty on unobserved walkable cells, all "
            "methods scored in physical units on common frames; 'All' pools the four channels. "
            "(a) CRPS skill "
            "against a constant-sigma null N(point, RMSE²) fitted to the same method and cells "
            "(per channel: that channel's own RMSE); below 0 the predicted uncertainty is worse "
            "than a constant. (b) Mean predictive SD / RMSE; dashed line = 1, below it "
            "under-dispersed. " + ("Error bars on 'All': 95% bootstrap intervals over the "
            f"{len(TEST_DATES)} test days (resampling days). " if has_ci else "") +
            "DINCAE's velocity-variance channel is log-normal in physical units and scored "
            "with the closed-form log-normal CRPS; all other channels are Gaussian.")

    # ---- 3. Inference time ------------------------------------------------------
    controlled_timing = output_dir / "inference_time_controlled.csv"
    timing_path = controlled_timing if controlled_timing.exists() else output_dir / "inference_time.csv"
    if timing_path.exists():
        timing = sorted(csv.DictReader(timing_path.open()),
                        key=lambda r: float(r["median_ms_per_frame"]))
        if timing:
            names = [r["method"] for r in timing]
            median = [float(r["median_ms_per_frame"]) for r in timing]
            p95 = [float(r["p95_ms_per_frame"]) for r in timing]
            fig, ax = ps.figure(rows_h=2.4, width=ps.TEXT_W * 0.7, constrained_layout=True)
            xs = np.arange(len(names))
            ax.bar(xs, median, 0.62, color=[ps.method_color(m) for m in names])
            for xi, v in zip(xs, median):
                ax.annotate(f"{v:.2g}", (xi, v), xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=ps.FS_TICK)
            ax.set_yscale("log")
            ax.set_ylim(min(median) / 3, max(median) * 3)
            ax.set_xticks(xs, [short.get(m, m).replace(" (", "\n(") for m in names],
                          fontsize=ps.FS_TICK - 0.5)
            ax.tick_params(axis="x", length=0)
            ax.set_ylabel("GPU time per frame (ms, log scale)")
            save(fig, "inference_latency")
            spread_pct = 100 * max((b - a) / a for a, b in zip(median, p95))
            gpu = timing[0].get("gpu", "")
            captions.append(
                "**inference_latency** — Median GPU compute time per reconstructed frame"
                + (f" on one {gpu}" if gpu else "") + ", prebuilt inputs, same frames for all "
                "methods; 4DVarNet solves 200-frame windows and EnKF runs 100 members. The 95th "
                f"percentile over repeats is within {spread_pct:.1f}% of the median for every "
                "method and is not drawn.")

    (fig_dir / "captions.md").write_text(
        "# Figure captions (generated)\n\n" + "\n\n".join(captions) + "\n")
    print(f"[plot] figures written under {fig_dir}")


def _allocate_samples(lengths: list[int], total: int) -> list[int]:
    """Allocate an exact sample count across days in proportion to valid frames."""
    import numpy as np

    if total < len(lengths):
        raise ValueError(f"n-images={total} is smaller than {len(lengths)} test days")
    weights = np.asarray(lengths, dtype=np.float64)
    ideal = total * weights / weights.sum()
    counts = np.floor(ideal).astype(int)
    for i in np.argsort(-(ideal - counts))[:total - int(counts.sum())]:
        counts[i] += 1
    return counts.tolist()


def _choose_image_samples(files: list[Path], count: int, walk: Any,
                          dT: int, warmup: int) -> list[dict[str, Any]]:
    """Choose frames evenly over time, while covering each day's density range.

    Each time segment contributes one frame nearest a deterministic, evenly
    spaced daily density quantile. Selection uses truth only for stratification;
    it never reads method errors.
    """
    import h5py
    import numpy as np

    plans = []
    valid_lengths = []
    for fp in files:
        with h5py.File(fp, "r") as f:
            n = int(f["grid"].shape[0])
        # Shared supported interval: DINCAE needs t-1/t+1, VarNet uses complete
        # non-overlapping dT windows, and EnKF's evaluated export starts after warmup.
        lo = max(1, warmup)
        hi = min(n - 1, (n // dT) * dT)  # exclusive
        if hi <= lo:
            raise ValueError(f"no common frame range for {fp.name}: [{lo}, {hi})")
        plans.append((fp, lo, hi))
        valid_lengths.append(hi - lo)
    quotas = _allocate_samples(valid_lengths, count)

    samples = []
    for (fp, lo, hi), quota in zip(plans, quotas):
        import h5py
        with h5py.File(fp, "r") as f:
            density = np.asarray(f["grid"][lo:hi, 0], dtype=np.float32)
            times = np.asarray(f["time"][lo:hi]) if "time" in f else np.arange(lo, hi)
        density_mean = density[:, walk].mean(axis=1)
        target_quantiles = np.linspace(0.02, 0.98, quota)
        target_density = np.quantile(density_mean, target_quantiles)
        edges = np.linspace(0, hi - lo, quota + 1, dtype=int)
        day = fp.name.split("_")[0]
        for slot in range(quota):
            left, right = int(edges[slot]), int(edges[slot + 1])
            left, right = max(left, 0), min(max(right, left + 1), hi - lo)
            local = density_mean[left:right]
            chosen_local = left + int(np.argmin(np.abs(local - target_density[slot])))
            frame = lo + chosen_local
            samples.append({
                "day": day, "frame": frame,
                "time": str(times[chosen_local].item() if hasattr(times[chosen_local], "item")
                             else times[chosen_local]),
                "density_mean_walkable": float(density_mean[chosen_local]),
                "density_quantile": float((density_mean <= density_mean[chosen_local]).mean()),
                "selection_quantile": float(target_quantiles[slot]),
                "slot_in_day": slot,
            })
    return samples


def _predict_senseiver_selected(model: Any, Y: Any, observed: Any,
                                frames: list[int], device: Any) -> Any:
    import numpy as np
    import torch
    from methods.senseiver import sensors

    pe = model.pos_enc.detach().cpu().numpy()
    mean = model.in_mean.detach().cpu().numpy()
    std = model.in_std.detach().cpu().numpy()
    if model.time_window > 1:
        windows = [(Y[max(0, t - model.time_window + 1):t + 1],
                    observed[max(0, t - model.time_window + 1):t + 1]) for t in frames]
        tok, pad, dt, _, cell = sensors.build_batch_temporal(
            windows, pe, mean, std, return_cell_idx=True)
        with torch.no_grad():
            out = model.reconstruct(tok.to(device), pad.to(device), dt.to(device),
                                    cell.to(device)).cpu().numpy()
    else:
        tok, pad, _ = sensors.build_batch(Y[frames], observed[frames], pe, mean, std)
        with torch.no_grad():
            out = model.reconstruct(tok.to(device), pad.to(device)).cpu().numpy()
    return out


def _predict_varnet_selected(solver: Any, dT: int, Y: Any, mask: Any, X0: Any,
                             frames: list[int], device: Any,
                             with_spread: bool = False, clip: bool = True) -> dict[int, Any]:
    import numpy as np
    import torch

    starts = sorted({(t // dT) * dT for t in frames})
    yw = np.stack([Y[s:s + dT] for s in starts]).transpose(0, 2, 1, 3, 4)
    mw = np.stack([mask[s:s + dT] for s in starts]).transpose(0, 2, 1, 3, 4)
    xw = np.stack([X0[s:s + dT] for s in starts]).transpose(0, 2, 1, 3, 4)
    result = {}
    # Small batches keep the differentiable unrolled solver comfortably within
    # the debug/V100 memory limit. Only selected 200-frame windows are evaluated.
    with torch.enable_grad():
        for i in range(len(starts)):
            yp = torch.from_numpy(yw[i:i + 1]).float().to(device)
            mp = torch.from_numpy(mw[i:i + 1]).float().to(device)
            xp = torch.from_numpy(xw[i:i + 1]).float().to(device)
            if with_spread:
                rec_t, var_t = solver(xp, yp, mp, return_var=True)
                spread = np.sqrt(np.maximum(var_t.detach().cpu().numpy()[0], 0))
                spread = spread.transpose(1, 0, 2, 3)
            else:
                rec_t = solver(xp, yp, mp)
            rec = rec_t.detach().cpu().numpy()[0].transpose(1, 0, 2, 3)
            start = starts[i]
            if clip:
                rec = np.clip(
                    rec, np.asarray([0, -5, -5, 0], np.float32)[None, :, None, None],
                    np.asarray([5, 5, 5, 2], np.float32)[None, :, None, None])
            result[start] = (rec, spread) if with_spread else rec
    if with_spread:
        return {t: (result[(t // dT) * dT][0][t % dT],
                    result[(t // dT) * dT][1][t % dT]) for t in frames}
    return {t: result[(t // dT) * dT][t % dT] for t in frames}


def _render_image_set(samples: list[dict[str, Any]], arrays: dict[str, Any],
                      out_dir: Path) -> None:
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image, ImageOps, ImageDraw

    methods = tuple(str(x) for x in arrays["method_names"])
    truth, pred = arrays["truth"], arrays["predictions"]
    observed, walk = arrays["observed"], arrays["walkable"]
    observation = arrays["observation"]
    # Match methods/enkf/checks/plot_reconstruction_enkf.py: density in Blues,
    # origin upper, equal aspect, black unit-heading arrows; absolute errors in
    # Reds. Each board uses common scales across methods for fair comparison.
    values = np.concatenate((truth[:, 0].ravel(), pred[:, :, 0].ravel()))
    upper = max(0.3, float(np.percentile(truth[:, 0], 98)))
    errors = np.abs(pred[:, :, 0] - truth[:, None, 0]).ravel()
    error_upper = max(float(np.percentile(errors, 99.5)), 1e-6)
    scales = {"density": [0.0, upper], "density_absolute_error": [0.0, error_upper]}
    (out_dir / "color_scales.json").write_text(json.dumps(scales, indent=2) + "\n")

    wanted = {f"comparison_{i + 1:03d}_{s['day']}_f{s['frame']:05d}.png"
              for i, s in enumerate(samples)}
    for stale in out_dir.glob("comparison_*.png"):
        if stale.name not in wanted:
            stale.unlink()

    files = []
    for i, sample in enumerate(samples):
        fig, axes = plt.subplots(len(methods), 4,
                                 figsize=(16, 3 * len(methods)), constrained_layout=True)
        fig.suptitle(
            f"{sample['day']}  frame {sample['frame']}  "
            f"walkable density={sample['density_mean_walkable']:.3f}  "
            f"observed={100 * observed[i][walk].mean():.1f}%",
            fontsize=15)
        xx, yy = np.meshgrid(np.arange(walk.shape[1]), np.arange(walk.shape[0]))

        def draw_density(ax, state, title, valid=None):
            valid = walk if valid is None else (walk & valid)
            state = np.where(valid[None], state, np.nan)
            density, vx, vy = state[0], state[1], state[2]
            im = ax.imshow(density, cmap="Blues", origin="upper", aspect="equal",
                           vmin=0, vmax=upper)
            heading = np.arctan2(vy, vx)
            u, v = np.cos(heading), np.sin(heading)
            # Same EnKF convention: direction-only arrows, omitting undefined
            # (obstacle) cells but retaining predictions even in low-density cells.
            no_data = np.isnan(density)
            u[no_data] = np.nan
            v[no_data] = np.nan
            ax.quiver(xx, yy, u, v, color="black", scale=30,
                      headwidth=3, headlength=4)
            ax.set_title(title, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            return im

        for row, method in enumerate(methods):
            obs_im = draw_density(axes[row, 0], observation[i], "Partial observation",
                                  valid=observed[i])
            truth_im = draw_density(axes[row, 1], truth[i], "Truth")
            pred_im = draw_density(axes[row, 2], pred[i, row], "Reconstruction")
            err_im = axes[row, 3].imshow(
                np.ma.masked_where(~walk, np.abs(pred[i, row, 0] - truth[i, 0])),
                origin="upper", cmap="Reds", aspect="equal", vmin=0,
                vmax=error_upper)
            axes[row, 0].set_ylabel(method, rotation=0, labelpad=58, fontsize=9, va="center")
            axes[row, 3].set_title("Absolute error", fontsize=10)
            axes[row, 3].set_xticks([]); axes[row, 3].set_yticks([])
        fig.colorbar(obs_im, ax=axes[:, :3].ravel().tolist(), shrink=0.8, pad=0.02,
                     label="Density")
        fig.colorbar(err_im, ax=axes[:, 3].ravel().tolist(), shrink=0.8, pad=0.03,
                     label="Absolute error")
        name = f"comparison_{i + 1:03d}_{sample['day']}_f{sample['frame']:05d}.png"
        path = out_dir / name
        fig.savefig(path, dpi=180)
        plt.close(fig)
        files.append(path)

    # A contact sheet of the 100 complete comparison boards for quick review.
    thumb_w, thumb_h, cols = 360, 198, 10
    rows = int(np.ceil(len(files) / cols))
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), "white")
    for i, path in enumerate(files):
        with Image.open(path) as image:
            image = ImageOps.contain(image.convert("RGB"), (thumb_w, thumb_h - 20))
            x = (i % cols) * thumb_w + (thumb_w - image.width) // 2
            y = (i // cols) * thumb_h + 20
            sheet.paste(image, (x, y))
        draw = ImageDraw.Draw(sheet)
        draw.text(((i % cols) * thumb_w + 6, (i // cols) * thumb_h + 3),
                  f"{i + 1:03d} {samples[i]['day']} f{samples[i]['frame']}", fill="black")
    sheet.save(out_dir / "contact_sheet.png", optimize=True)


def _render_spread_comparison(samples: list[dict[str, Any]], truth: Any,
                              predictions: Any, density_spread: Any,
                              walk: Any, path: Path) -> None:
    """Compare native predictive density SD and density error on one common frame."""
    import numpy as np
    import matplotlib.pyplot as plt

    index = int(np.argmax(np.where(walk[None], truth[:, 0], 0).sum(axis=(1, 2))))
    methods = (("DINCAE", 2), ("4DVarNet aughead_obs (single)", 4), ("EnKF", 5))
    vmax_density = max(0.3, float(np.percentile(truth[index, 0][walk], 98)))
    vmax_error = max(1e-6, float(np.max([
        np.max(np.abs(predictions[index, pred_i, 0][walk] - truth[index, 0][walk]))
        for _, pred_i in methods])))
    vmax_spread = max(1e-6, float(np.max(density_spread[index, :, walk])))
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    for row, (name, pred_i) in enumerate(methods):
        mean = predictions[index, pred_i, 0]
        spread = density_spread[index, row]
        panels = ((mean, "Blues", vmax_density),
                  (np.abs(mean - truth[index, 0]), "Purples", vmax_error),
                  (spread, "Reds", vmax_spread))
        for col, (values, cmap, vmax) in enumerate(panels):
            im = axes[row, col].imshow(np.where(walk, values, np.nan),
                                        cmap=cmap, origin="upper", aspect="equal",
                                        vmin=0, vmax=vmax)
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            fig.colorbar(im, ax=axes[row, col], fraction=.046, pad=.04)
        axes[row, 0].set_ylabel(name, rotation=0, labelpad=72, va="center", fontsize=9)
    for ax, title in zip(axes[0], ("Reconstruction", "Absolute error", "Predicted spread")):
        ax.set_title(title)
    fig.suptitle(f"{samples[index]['day']} frame {samples[index]['frame']}: "
                 "density uncertainty on the same held-out frame (all walkable cells)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def generate_comparison_images(data_root: Path, output_dir: Path, count: int) -> None:
    """Rerun only selected frames and write 100 comparable multi-method image boards."""
    import csv
    import json
    import numpy as np
    import torch

    from crowdcore import config
    config.CFG["data"]["root"] = str(data_root.resolve())
    # Import data/model modules only after setting the root: observation_model
    # resolves its HDF5 cache path at import time.
    from crowdcore import navigation, observation_model as om
    from methods.senseiver import dataset as senseiver_ds
    from methods.senseiver.network import Senseiver
    from methods.varnet.checks.model_io import load_solver
    from methods.dincae.checks import evaluate as dincae_eval

    out_dir = output_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [Path(p) for p in om.split_files("test")]
    if len(files) != len(TEST_DATES):
        raise RuntimeError(f"expected {len(TEST_DATES)} test days, found {len(files)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    walk = navigation.build_valid_mask_from_config().astype(bool)

    # Load the frozen checkpoints from the deliverable directory.
    sense_ck = torch.load(MODELS / FINAL_CONFIG["senseiver_a"]["file"],
                          map_location=device, weights_only=False)
    sense_a = Senseiver(**sense_ck["hparams"]).to(device)
    sense_a.load_state_dict(sense_ck["model"]); sense_a.eval()
    sense_ck = torch.load(MODELS / FINAL_CONFIG["senseiver_g"]["file"],
                          map_location=device, weights_only=False)
    sense_g = Senseiver(**sense_ck["hparams"]).to(device)
    sense_g.load_state_dict(sense_ck["model"]); sense_g.eval()
    var_solver, var_args, _ = load_solver(MODELS / FINAL_CONFIG["varnet_mse"]["file"], device)
    aug_solver, aug_args, _ = load_solver(
        MODELS / FINAL_CONFIG["varnet_aughead"]["file"], device)
    dT = int(var_args["dT"])
    if dT != 200 or int(aug_args["dT"]) != dT or not aug_solver.augmented_var:
        raise RuntimeError("packaged 4DVarNet mean/uncertainty solvers have incompatible layouts")
    dincae_models, _, _ = dincae_eval.load_models(
        str(MODELS), str(MODELS / FINAL_CONFIG["dincae"]["file"]), device)
    dincae_stats = dincae_eval.StateStats()

    warmup = int(FINAL_CONFIG["enkf"].get("warmup", 500))
    samples = _choose_image_samples(files, count, walk, dT, warmup)
    sample_by_day: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for row_id, sample in enumerate(samples):
        sample_by_day.setdefault(sample["day"], []).append((row_id, sample))

    n, c, h, w = count, *STATE_SHAPE
    truth_out = np.empty((n, c, h, w), dtype=np.float32)
    obs_out = np.empty_like(truth_out)
    observed_out = np.empty((n, h, w), dtype=bool)
    pred_out = np.empty((n, 6, c, h, w), dtype=np.float32)
    spread_out = np.empty((n, 3, h, w), dtype=np.float32)
    obs_cfg = senseiver_ds.obs_config()

    for fp in files:
        day = fp.name.split("_")[0]
        rows = sample_by_day.get(day, [])
        if not rows:
            continue
        ids = [sample["frame"] for _, sample in rows]
        X, times = om.load_state(str(fp))
        X = np.asarray(X, dtype=np.float32)
        obs = om.generate_observations(
            X, sensing_range=obs_cfg["sensing_range"], num_agents=obs_cfg["num_agents"],
            add_noise=obs_cfg["add_noise"], seed=om.day_seed(fp),
            valid_mask=navigation.build_valid_mask_from_config(X),
            obs_every_k=obs_cfg["obs_every_k"])
        Y, Omask = obs["Y"].astype(np.float32), obs["Omega"].astype(bool)
        Yflat = Y.reshape(len(X), c, h * w)
        Oflat = Omask.reshape(len(X), h * w)

        # Same per-frame inputs as compare5; Senseiver-G receives its causal history.
        pa = _predict_senseiver_selected(sense_a, Yflat, Oflat, ids, device)
        pg = _predict_senseiver_selected(sense_g, Yflat, Oflat, ids, device)
        pa = np.clip(pa, np.asarray([0, -5, -5, 0])[None, :, None, None],
                     np.asarray([5, 5, 5, 2])[None, :, None, None])
        pg = np.clip(pg, np.asarray([0, -5, -5, 0])[None, :, None, None],
                     np.asarray([5, 5, 5, 2])[None, :, None, None])

        # Reconstruct only the non-overlapping 200-frame windows containing the
        # chosen frames, with the same fill-missing initialization as compare5.
        Omc = np.repeat(Omask[:, None], c, axis=1)
        X0 = om.fill_missing_state(Y, Omc, method=obs_cfg["init_method"])
        pv_by_frame = _predict_varnet_selected(
            var_solver, dT, Y, Omc, X0, ids, device)
        aug_by_frame = _predict_varnet_selected(
            aug_solver, dT, Y, Omc, X0, ids, device, with_spread=True)

        # DINCAE's tested prediction helper includes its exact temporal context.
        # Its observation construction is deterministic; assert it matches ours.
        Xt, rec, _, dincae_sd, dincae_mask = dincae_eval.predict_day(
            dincae_models, dincae_stats, str(fp), device, batch=256)
        if not np.array_equal(Xt[0], X[1]) or any(
                not np.array_equal(dincae_mask[t - 1], obs["Omega_c"][t]) for t in ids):
            raise RuntimeError(f"DINCAE frame/observation alignment check failed on {day}")
        rec = dincae_eval.clip_bounds(rec)

        # Reuse the full EnKF run saved by the completed seven-day evaluation.
        enkf_npz = output_dir / "raw" / "enkf_exports" / f"{day}.npz"
        enkf_json = output_dir / "raw" / "enkf" / f"{day}.json"
        if not enkf_npz.exists() or not enkf_json.exists():
            raise FileNotFoundError(f"full EnKF export is required for image comparison: {enkf_npz}")
        cfg = json.loads(enkf_json.read_text())["config"]
        day_warmup = int(cfg["warmup"])
        with np.load(enkf_npz) as en:
            emean = en["mean"]
            espread = en["spread"]
            eobs = en["observed"]
            etruth = en["truth"]
            for local, (row_id, sample) in enumerate(rows):
                t = ids[local]
                di = t - 1  # DINCAE predicts original frames 1..T-2
                ei = t - day_warmup
                if ei < 0 or ei >= len(emean):
                    raise IndexError(f"EnKF export has no frame {t} ({day})")
                if not np.array_equal(eobs[ei], Omask[t]):
                    raise RuntimeError(f"EnKF observation mask differs at {day} frame {t}")
                if not np.allclose(etruth[ei], X[t], rtol=0, atol=1e-6):
                    raise RuntimeError(f"EnKF truth alignment differs at {day} frame {t}")
                truth_out[row_id] = X[t]
                obs_out[row_id] = Y[t]
                observed_out[row_id] = Omask[t]
                pred_out[row_id, 0] = pa[local]
                pred_out[row_id, 1] = pg[local]
                pred_out[row_id, 2] = rec[di]
                pred_out[row_id, 3] = pv_by_frame[t]
                pred_out[row_id, 4] = aug_by_frame[t][0]
                pred_out[row_id, 5] = emean[ei]
                # DINCAE density has no nonlinear channel transform: its
                # normalised SD converts to physical units with std[density].
                spread_out[row_id, 0] = dincae_sd[di, 0] * dincae_stats.std[0]
                spread_out[row_id, 1] = aug_by_frame[t][1][0]
                spread_out[row_id, 2] = espread[ei, 0]
                sample["observed_fraction_walkable"] = float(Omask[t][walk].mean())
                sample["timestamp"] = str(times[t].item() if hasattr(times[t], "item")
                                           else times[t])
        del X, Y, obs, Xt, rec, pa, pg, pv_by_frame, aug_by_frame, X0
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[images] {day}: {len(rows)} common-frame samples complete", flush=True)

    with (out_dir / "sample_manifest.csv").open("w", newline="") as f:
        fields = list(samples[0])
        writer = csv.DictWriter(f, fields); writer.writeheader(); writer.writerows(samples)
    np.savez_compressed(
        out_dir / "selected_predictions.npz", truth=truth_out, observation=obs_out,
        observed=observed_out, walkable=walk, predictions=pred_out,
        density_spread=spread_out,
        method_names=np.asarray(("Senseiver-A", "Senseiver-G", "DINCAE",
                                 "4DVarNet MSE", "4DVarNet aughead_obs (single)", "EnKF")),
        spread_method_names=np.asarray(("DINCAE", "4DVarNet aughead_obs (single)", "EnKF")),
        day=np.asarray([x["day"] for x in samples]),
        frame=np.asarray([x["frame"] for x in samples], dtype=np.int32))
    _render_image_set(samples, {"truth": truth_out, "observation": obs_out,
                                "observed": observed_out, "walkable": walk,
                                "predictions": pred_out,
                                "method_names": ("Senseiver-A", "Senseiver-G", "DINCAE",
                                                 "4DVarNet MSE", "4DVarNet aughead_obs (single)", "EnKF")}, out_dir)
    _render_spread_comparison(samples, truth_out, pred_out, spread_out, walk,
                              out_dir / "spread_comparison.png")
    print(f"[images] wrote {len(samples)} comparison boards to {out_dir}", flush=True)


def redraw_comparison_images(output_dir: Path) -> None:
    """Re-render saved comparison samples after a style change, without inference."""
    import csv
    import numpy as np

    out_dir = output_dir / "images"
    archive = out_dir / "selected_predictions.npz"
    manifest = out_dir / "sample_manifest.csv"
    if not archive.exists() or not manifest.exists():
        raise SystemExit(f"{archive} and {manifest} are required; run `images` first")
    with manifest.open(newline="") as f:
        samples = list(csv.DictReader(f))
    for row in samples:
        row["frame"] = int(row["frame"])
        for key in ("density_mean_walkable", "density_quantile", "selection_quantile",
                    "observed_fraction_walkable"):
            row[key] = float(row[key])
    with np.load(archive) as z:
        arrays = {key: z[key].copy() for key in
                  ("truth", "observation", "observed", "walkable", "predictions",
                   "method_names")}
        if "density_spread" in z:
            arrays["density_spread"] = z["density_spread"].copy()
    _render_image_set(samples, arrays, out_dir)
    if "density_spread" in arrays:
        _render_spread_comparison(samples, arrays["truth"], arrays["predictions"],
                                  arrays["density_spread"], arrays["walkable"],
                                  out_dir / "spread_comparison.png")
    print(f"[redraw] refreshed {len(samples)} boards under {out_dir}", flush=True)


def evaluate_uncertainty(data_root: Path, output_dir: Path, days: int = 0, frames: int = 0) -> None:
    """Score every uncertainty method with one protocol, in physical units.

    Per test day the observations are generated once and every method is read on
    the same frames (the common support of DINCAE's t-1/t+1 context, 4DVarNet's
    complete dT windows and the EnKF warmup) and the same cells (walkable and not
    observed at that frame, all four channels).  Predictive families:

      4DVarNet aughead_obs   N(mean, sd^2) from the learned variance head
      EnKF                   N(ensemble mean, ensemble spread^2), from the saved exports
      DINCAE                 N in its standardised space mapped back: density/vx/vy
                             are Gaussian with sd * std[c]; var goes through log1p,
                             so its physical predictive is a shifted log-normal

    No physical clipping: the scored distribution is each method's own.  The
    constant-sigma null model is N(point, RMSE^2) of the same slice and method.
    Needs the EnKF exports from `full`/`smoke`; writes raw/uncertainty_unified.json.
    """
    import numpy as np
    import torch

    from crowdcore import config
    config.CFG["data"]["root"] = str(data_root.resolve())
    from crowdcore import navigation, observation_model as om
    from methods.senseiver import dataset as senseiver_ds
    from methods.varnet.checks.model_io import load_solver
    from methods.dincae.checks import evaluate as dincae_eval
    from methods.dincae.state import CHANNEL_TRANSFORM
    from compare import score_uncertainty as su

    raw = output_dir / "raw"
    files = [Path(p) for p in om.split_files("test")]
    if days:
        files = files[:days]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("!! no GPU -- run this through sbatch/gpu-debug", flush=True)
    walk = navigation.build_valid_mask_from_config().astype(bool)

    aug_solver, aug_args, _ = load_solver(MODELS / FINAL_CONFIG["varnet_aughead"]["file"], device)
    dT = int(aug_args["dT"])
    if not aug_solver.augmented_var:
        raise RuntimeError("packaged 4DVarNet aughead solver has no variance head")
    obs_cfg = senseiver_ds.obs_config()
    if aug_args.get("obs_every_k") not in (None, obs_cfg["obs_every_k"]):
        raise RuntimeError(f"4DVarNet was trained with obs_every_k={aug_args['obs_every_k']}, "
                           f"the shared observations use {obs_cfg['obs_every_k']}")
    dincae_models, _, _ = dincae_eval.load_models(
        str(MODELS), str(MODELS / FINAL_CONFIG["dincae"]["file"]), device)
    dstats = dincae_eval.StateStats()
    if not np.array_equal(dstats.valid, walk):
        raise RuntimeError("DINCAE's walkable mask differs from the navigation mask")
    if tuple(CHANNEL_TRANSFORM) != (None, None, None, "log1p"):
        raise RuntimeError(f"unexpected DINCAE channel transforms {CHANNEL_TRANSFORM}")

    keys = ("varnet_aughead", "dincae", "enkf")
    acc = {k: {"walkable_blind": su.Accumulator(),
               **{f"channel_{c}": su.Accumulator() for c in CHANNELS}} for k in keys}
    # Point-estimate residuals of the scored cells, kept per channel for the null
    # model, whose sigma (the slice RMSE) is only known after the last day.
    resid: dict[str, list[list[Any]]] = {k: [[] for _ in CHANNELS] for k in keys}
    per_day = []
    # One accumulator per method and day, for day-level bootstrap intervals.
    day_acc: dict[str, list[Any]] = {k: [] for k in keys}

    for fp in files:
        day = fp.name.split("_")[0]
        X = np.asarray(om.load_state(str(fp))[0], dtype=np.float32)
        if frames:
            X = X[:frames]
        n = len(X)
        obs = om.generate_observations(
            X, sensing_range=obs_cfg["sensing_range"], num_agents=obs_cfg["num_agents"],
            add_noise=obs_cfg["add_noise"], seed=om.day_seed(fp),
            valid_mask=navigation.build_valid_mask_from_config(X),
            obs_every_k=obs_cfg["obs_every_k"])
        Y, Omask = obs["Y"].astype(np.float32), obs["Omega"].astype(bool)

        enkf_npz = raw / "enkf_exports" / f"{day}.npz"
        enkf_json = raw / "enkf" / f"{day}.json"
        if not enkf_npz.exists() or not enkf_json.exists():
            raise FileNotFoundError(f"EnKF export is required (run `full` first): {enkf_npz}")
        warmup = int(json.loads(enkf_json.read_text())["config"]["warmup"])
        lo, hi = max(1, warmup), min(n - 1, (n // dT) * dT)
        if hi <= lo:
            raise ValueError(f"no common frame range for {day}: [{lo}, {hi})")
        t_idx = np.arange(lo, hi)
        truth = X[lo:hi]

        # 4DVarNet: every complete window that touches the common range, unclipped.
        Omc = np.repeat(Omask[:, None], len(CHANNELS), axis=1)
        X0 = om.fill_missing_state(Y, Omc, method=obs_cfg["init_method"])
        vr = _predict_varnet_selected(aug_solver, dT, Y, Omc, X0, list(t_idx), device,
                                      with_spread=True, clip=False)
        v_mean = np.stack([vr[t][0] for t in t_idx]); v_sd = np.stack([vr[t][1] for t in t_idx])
        del vr, X0, Omc

        # DINCAE: predicts original frames 1..n-2, so frame t is row t-1.
        Xt, _rec, mu_n, sd_n, dmask = dincae_eval.predict_day(
            dincae_models, dstats, str(fp), device, frames, 256)
        if not np.array_equal(Xt[lo - 1:hi - 1], truth) or \
                not np.array_equal(dmask[lo - 1:hi - 1, 0], Omask[lo:hi]):
            raise RuntimeError(f"DINCAE frame/observation alignment check failed on {day}")
        std = dstats.std.reshape(1, -1, 1, 1)
        d_loc = mu_n[lo - 1:hi - 1] * std + dstats.mean[None]     # transformed space
        d_scale = np.maximum(sd_n[lo - 1:hi - 1] * std, 1e-12)
        del Xt, _rec, mu_n, sd_n, dmask

        with np.load(enkf_npz) as en:
            e_mean = en["mean"][lo - warmup:hi - warmup]
            e_sd = en["spread"][lo - warmup:hi - warmup]
            if not np.array_equal(en["observed"][lo - warmup:hi - warmup], Omask[lo:hi]) or \
                    not np.allclose(en["truth"][lo - warmup:hi - warmup], truth, rtol=0, atol=1e-6):
                raise RuntimeError(f"EnKF export alignment check failed on {day}")

        sel = walk[None] & ~Omask[lo:hi]                          # (T,H,W), same for all channels
        for k in keys:
            day_acc[k].append(su.Accumulator())
        for c, name in enumerate(CHANNELS):
            x = truth[:, c][sel]
            parts = {
                "varnet_aughead": ("gauss", v_mean[:, c][sel], np.maximum(v_sd[:, c][sel], 1e-12)),
                "enkf": ("gauss", e_mean[:, c][sel], np.maximum(e_sd[:, c][sel], 1e-12)),
                "dincae": ("lognormal1p" if CHANNEL_TRANSFORM[c] == "log1p" else "gauss",
                           d_loc[:, c][sel], d_scale[:, c][sel]),
            }
            for k, (family, loc, scale) in parts.items():
                for a in (acc[k]["walkable_blind"], acc[k][f"channel_{name}"], day_acc[k][-1]):
                    (a.add_lognormal1p if family == "lognormal1p" else a.add)(loc, scale, x)
                point = np.expm1(loc) if family == "lognormal1p" else loc
                resid[k][c].append((x - point).astype(np.float32))
        per_day.append({"day": day, "frames": [int(lo), int(hi)],
                        "scored_cells_per_channel": int(sel.sum())})
        print(f"[uncertainty] {day}: frames [{lo}, {hi}), {int(sel.sum())} cells/channel",
              flush=True)
        del X, Y, obs, truth, v_mean, v_sd, d_loc, d_scale, e_mean, e_sd
        if device.type == "cuda":
            torch.cuda.empty_cache()

    methods = {}
    for k in keys:
        res = {t: a.result() for t, a in acc[k].items()}
        slices = {"walkable_blind": [np.concatenate(r) for r in resid[k]]}
        slices |= {f"channel_{c}": [np.concatenate(resid[k][i])] for i, c in enumerate(CHANNELS)}
        for t, arrs in slices.items():
            d = np.concatenate(arrs)
            sigma0 = float(np.sqrt(np.mean(d.astype(np.float64) ** 2)))
            null = su.Accumulator()
            for i in range(0, d.size, 1 << 24):
                chunk = d[i:i + (1 << 24)]
                null.add(np.zeros_like(chunk), np.full(chunk.shape, sigma0), chunk)
            res[f"{t}_constant_sigma_baseline"] = null.result()
            res[t]["crps_skill"] = 1.0 - res[t]["crps"] / null.result()["crps"]
        # Per day, scored against the pooled null sigma, so that summing any
        # subset of days reproduces the pooled ratios (used for bootstrap CIs).
        sigma0 = res["walkable_blind_constant_sigma_baseline"]["sigma_mean"]
        days_out = []
        for d, a in enumerate(day_acc[k]):
            r = a.result()
            dres = np.concatenate([resid[k][c][d] for c in range(len(CHANNELS))])
            null = su.Accumulator()
            null.add(np.zeros_like(dres), np.full(dres.shape, sigma0), dres)
            days_out.append({"day": per_day[d]["day"], "n": r["n"], "crps": r["crps"],
                             "crps_null": null.result()["crps"], "rmse": r["rmse"],
                             "sigma_mean": r["sigma_mean"], "coverage": r["coverage"]})
        methods[k] = {"label": FINAL_CONFIG[k]["label"], "results": res, "per_day": days_out}
        r = res["walkable_blind"]
        print(f"[uncertainty] {FINAL_CONFIG[k]['label']:<32} rmse {r['rmse']:.4f}  "
              f"crps {r['crps']:.4f}  skill {r['crps_skill']:.4f}  "
              f"sp/sk {r['spread_skill']:.3f}  cov90 {r['coverage'][90]:.3f}", flush=True)

    doc = {"space": "physical units, all methods",
           "cells": "walkable and not observed at that frame; all four channels pooled",
           "frames": "common support: DINCAE t-1/t+1, complete 4DVarNet windows, after EnKF warmup",
           "families": {"varnet_aughead": "gaussian", "enkf": "gaussian",
                        "dincae": {c: ("log1p-lognormal" if CHANNEL_TRANSFORM[i] == "log1p"
                                       else "gaussian") for i, c in enumerate(CHANNELS)}},
           "null_model": "N(point, slice RMSE^2), per method and slice",
           "clipping": "none",
           "n_days": len(files), "frames_limit": frames, "per_day": per_day,
           "methods": methods}
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "uncertainty_unified.json").write_text(json.dumps(doc, indent=2))
    print(f"[uncertainty] wrote {raw / 'uncertainty_unified.json'}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "verify", "smoke", "full", "images",
                                        "uncertainty", "redraw", "collect", "plot",
                                        "benchmark"))
    p.add_argument("--data-root", type=Path, default=Path("/scratch/work/zhangx29/data"))
    p.add_argument("--output-dir", type=Path, default=None,
                   help="default: outputs/dev/<command> for smoke/verify (test runs), "
                        "outputs/full for everything else (the reported results)")
    p.add_argument("--methods", nargs="*", choices=tuple(FINAL_CONFIG), default=None)
    p.add_argument("--n-images", type=int, default=100,
                   help="number of matched test-frame comparison boards for `images`")
    p.add_argument("--days", type=int, default=0,
                   help="`uncertainty` only: first N test days (0 = all seven)")
    p.add_argument("--frames", type=int, default=0,
                   help="`uncertainty` only: first N frames of each day (0 = whole day)")
    p.add_argument("--full-checksum", action="store_true")
    p.add_argument("--allow-incomplete", action="store_true",
                   help="development only: package/verify finished artifacts and report missing final ones")
    a = p.parse_args()
    if a.output_dir is None:
        a.output_dir = (HERE / "outputs" / "dev" / a.command if a.command in ("smoke", "verify")
                        else HERE / "outputs" / "full")
    return a


def main() -> None:
    a = parse_args()
    if a.command == "prepare":
        raise SystemExit(prepare_models(a.allow_incomplete))
    if a.command == "verify":
        r = verify(a.data_root, a.full_checksum, not a.allow_incomplete)
        write_verification(r, a.output_dir)
        raise SystemExit(0 if r["ok"] else 1)
    if a.command == "collect":
        a.output_dir.mkdir(parents=True, exist_ok=True); collect(a.output_dir); return
    if a.command == "plot":
        plot_summary(a.output_dir); return
    if a.command == "benchmark":
        from supervisor_evaluation.bench_inference import run as run_benchmark
        run_benchmark(a.output_dir, a.data_root)
        plot_summary(a.output_dir)
        return
    if a.command == "redraw":
        redraw_comparison_images(a.output_dir); return

    report = verify(a.data_root, False, not a.allow_incomplete)
    write_verification(report, a.output_dir)
    if not report["ok"]:
        raise SystemExit("verification failed; evaluation did not start")
    if a.command == "images":
        if a.n_images < len(TEST_DATES):
            raise SystemExit(f"--n-images must be at least {len(TEST_DATES)} so every test day is represented")
        generate_comparison_images(a.data_root, a.output_dir, a.n_images)
        return
    if a.command == "uncertainty":
        evaluate_uncertainty(a.data_root, a.output_dir, a.days, a.frames)
        collect(a.output_dir); plot_summary(a.output_dir)
        return
    chosen = set(a.methods or FINAL_CONFIG)
    run_available(a.command, a.data_root, a.output_dir, chosen)
    if set(UNCERTAINTY_METHODS) <= chosen:
        # smoke: one day, the same 600 frames the EnKF smoke run covers
        evaluate_uncertainty(a.data_root, a.output_dir, *((1, 600) if a.command == "smoke" else (0, 0)))
    collect(a.output_dir)
    if a.command == "full" and chosen == set(FINAL_CONFIG):
        from supervisor_evaluation.bench_inference import run as run_benchmark
        run_benchmark(a.output_dir, a.data_root)
    plot_summary(a.output_dir)


if __name__ == "__main__":
    main()
