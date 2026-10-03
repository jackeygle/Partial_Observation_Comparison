"""Validation-split comparisons of the re-selection that need no training
(supervisor_evaluation/RESELECTION_PROTOCOL.md).

    python3 supervisor_evaluation/reselect_validation.py enkf --data-root /scratch/work/zhangx29/data
    python3 supervisor_evaluation/reselect_validation.py aug  --data-root /scratch/work/zhangx29/data

enkf  runs the filter with the four process-noise candidates E1-E4 on the seven validation
      days (the observations of the evaluation protocol: routes seeded by the date) and
      scores every candidate on the blind walkable cells of every frame after the warm-up:
      RMSE and Gaussian CRPS of the ensemble mean and spread, four channels pooled.
aug   scores the two uncertainty 4DVarNet candidates (aug0, aughead) at their
      validation-selected epochs on the same validation observations: RMSE and Gaussian
      CRPS on the blind walkable cells of every frame in a complete window, unclipped,
      four channels pooled.

Nothing here reads the test split. Outputs: supervisor_evaluation/outputs/reselect/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import evaluate as ev  # noqa: E402

OUT = HERE / "outputs" / "reselect"
SCALES = {"E1": dict(channel=(1, 1, 1, 1), blind=(1.0, 1.25, 1.4, 0.9333333), kind="residual"),
          "E2": dict(channel=(1, 1, 1, 1), blind=(1, 1, 1, 1), kind="residual"),
          "E3": dict(channel=(1.0, 1.25, 1.4, 0.9333333), blind=(1, 1, 1, 1), kind="residual"),
          "E4": dict(channel=(1, 1, 1, 1), blind=(1, 1, 1, 1), kind="gaussian")}


def score_pooled(acc, mean, sd, truth, sel):
    for c in range(4):
        acc.add(mean[:, c][sel], np.maximum(sd[:, c][sel], 1e-12), truth[:, c][sel])


def run_enkf(data_root: Path, days: int) -> None:
    from compare import score_uncertainty as su
    cfg = ev.FINAL_CONFIG["enkf"]
    obs_dir, out = OUT / "enkf_observations", OUT / "enkf"
    obs_dir.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    ev._python_module("methods.enkf.checks.export_obs_for_enkf",
                      ["--split", "valid", "--frames", "0", "--outdir", str(obs_dir),
                       "--obs-every-k", "1"], ROOT, data_root)
    dates = sorted(p.stem.replace("obs_", "") for p in obs_dir.glob("obs_*.npz"))
    if days:
        dates = dates[:days]
    result = {}
    for name, s in SCALES.items():
        acc = su.Accumulator()
        for day in dates:
            exp = out / f"{name}_{day}.npz"
            if not exp.exists():
                args = [
                    "--noise-kind", s["kind"], "--density-noise-space", "log1p",
                    "--log-density-mean-correction", "ensemble",
                    "--model-checkpoint", str(ev.MODELS / cfg["file"]),
                    "--input-frames", "5", "--bank", str(ev.MODELS / cfg["bank_file"]),
                    "--scale", str(cfg["q_scale"]), "--temporal-rho", str(cfg["temporal_rho"]),
                    "--channel-scales", *map(str, s["channel"]),
                    "--blind-channel-scales", *map(str, s["blind"]),
                    "--cross-channel-matrix", *map(str, cfg["cross_channel_matrix"]),
                    "--ensemble", str(cfg["ensemble"]), "--radius", str(cfg["localization_radius"]),
                    "--frames", "1000000", "--warmup", str(cfg["warmup"]),
                    "--day", day, "--obs-dir", str(obs_dir),
                    "--calibration-export", str(exp), "--out", str(out / f"{name}_{day}.json")]
                if s["kind"] == "residual" and cfg.get("bank_native_std"):
                    args.append("--bank-native-std")
                ev._python_module("methods.enkf.enkf_opt.experiments.eval_structured_q_gpu",
                                  args, ROOT, data_root)
            with np.load(exp) as z:
                sel = z["walkable"][None] & ~z["observed"]
                score_pooled(acc, z["mean"], z["spread"], z["truth"], sel)
        r = acc.result()
        result[name] = {"rmse": r["rmse"], "crps": r["crps"], "spread_rmse": r["spread_skill"],
                        "n": r["n"], "config": s}
        print(f"[enkf] {name}: RMSE {r['rmse']:.5f}  CRPS {r['crps']:.5f}  "
              f"spread/RMSE {r['spread_skill']:.3f}", flush=True)
    base = result["E2"]["rmse"]
    ok = {k: v for k, v in result.items() if v["rmse"] <= 1.05 * base}
    win = min(ok, key=lambda k: ok[k]["crps"])
    doc = {"split": "valid", "days": dates, "cells": "blind walkable, every frame after the warm-up",
           "rule": "lowest CRPS among candidates with RMSE <= 1.05 x RMSE(E2)",
           "candidates": result, "eligible": sorted(ok), "winner": win}
    (OUT / "enkf_validation.json").write_text(json.dumps(doc, indent=2))
    print(f"[enkf] winner {win} -> {OUT / 'enkf_validation.json'}", flush=True)


def run_aug(data_root: Path, days: int) -> None:
    import torch
    from crowdcore import config
    config.CFG["data"]["root"] = str(data_root.resolve())
    from crowdcore import navigation, observation_model as om
    from compare import score_uncertainty as su
    from methods.senseiver import dataset as senseiver_ds
    from methods.varnet.checks.model_io import load_solver, reported_ckpt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = {"aug0": ROOT / "methods/varnet/runs/varnet_aug0_h96_s0",
            "aughead": ROOT / "methods/varnet/runs/varnet_aughead_obs_h96_s0"}
    solvers = {}
    for k, rd in runs.items():
        p = reported_ckpt(str(rd))
        sol, args, _ = load_solver(p, device)
        if not getattr(sol, "augmented_var", False):
            raise RuntimeError(f"{p} has no augmented variance")
        solvers[k] = (sol, int(args["dT"]), Path(p).name)
    walk = navigation.build_valid_mask_from_config().astype(bool)
    obs_cfg = senseiver_ds.obs_config()
    files = [Path(p) for p in om.split_files("valid")]
    if days:
        files = files[:days]
    acc = {k: su.Accumulator() for k in runs}
    for fp in files:
        X = np.asarray(om.load_state(str(fp))[0], dtype=np.float32)
        obs = om.generate_observations(
            X, sensing_range=obs_cfg["sensing_range"], num_agents=obs_cfg["num_agents"],
            add_noise=obs_cfg["add_noise"], seed=om.day_seed(fp),
            valid_mask=navigation.build_valid_mask_from_config(X),
            obs_every_k=obs_cfg["obs_every_k"])
        Y, Om = obs["Y"].astype(np.float32), obs["Omega"].astype(bool)
        Omc = np.repeat(Om[:, None], 4, axis=1)
        X0 = om.fill_missing_state(Y, Omc, method=obs_cfg["init_method"])
        for k, (sol, dT, _) in solvers.items():
            t_idx = list(range(1, (len(X) // dT) * dT))
            vr = ev._predict_varnet_selected(sol, dT, Y, Omc, X0, t_idx, device,
                                             with_spread=True, clip=False)
            mean = np.stack([vr[t][0] for t in t_idx]); sd = np.stack([vr[t][1] for t in t_idx])
            sel = walk[None] & ~Om[t_idx]
            score_pooled(acc[k], mean, sd, X[t_idx], sel)
            del vr, mean, sd
        print(f"[aug] {fp.name.split('_')[0]} done", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = {}
    for k, a in acc.items():
        r = a.result()
        result[k] = {"checkpoint": solvers[k][2], "rmse": r["rmse"], "crps": r["crps"],
                     "spread_rmse": r["spread_skill"], "n": r["n"]}
        print(f"[aug] {k} ({solvers[k][2]}): RMSE {r['rmse']:.5f}  CRPS {r['crps']:.5f}  "
              f"spread/RMSE {r['spread_skill']:.3f}", flush=True)
    win = min(result, key=lambda k: result[k]["crps"])
    doc = {"split": "valid", "days": [f.name for f in files],
           "cells": "blind walkable, every frame in a complete window, unclipped",
           "rule": "lowest CRPS", "candidates": result, "winner": win}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "aug_validation.json").write_text(json.dumps(doc, indent=2))
    print(f"[aug] winner {win} -> {OUT / 'aug_validation.json'}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("enkf", "aug"))
    p.add_argument("--data-root", type=Path, default=Path("/scratch/work/zhangx29/data"))
    p.add_argument("--days", type=int, default=0, help="first N validation days (smoke test)")
    a = p.parse_args()
    global OUT
    if a.days:
        OUT = HERE / "outputs" / "dev" / "reselect_smoke"
    (run_enkf if a.command == "enkf" else run_aug)(a.data_root, a.days)


if __name__ == "__main__":
    main()
