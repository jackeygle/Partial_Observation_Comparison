"""Sequential KF test selected automatically from one-step calibration results."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from crowdcore import navigation
from methods.dincae.state import channel_valid
from methods.enkf.checks.compare_joint_uncertainty import accumulate, finalize
from methods.enkf.checks.run_lowrank_unetkf import load_models, run_file
from methods.enkf.lowrank.model import STATE_SCALE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--calibration", required=True)
    p.add_argument("--new-checkpoint", required=True)
    p.add_argument("--old-checkpoint", required=True)
    p.add_argument("--obs", required=True)
    p.add_argument("--frames", type=int, default=5000)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--new-alphas", default="",
                   help="comma-separated override for new_cov sequential scales")
    p.add_argument("--new-component-scales", default="",
                   help="comma-separated factor:diagonal scale pairs; overrides --new-alphas")
    p.add_argument("--blind-gammas", default="1",
                   help="comma-separated blind-walkable diagonal std multipliers")
    p.add_argument("--blind-empty-gammas", default="1",
                   help="extra std multipliers where blind and predicted density is empty")
    p.add_argument("--skip-old", action="store_true")
    p.add_argument("--skip-mean", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--export-npz", default="",
                   help="directory to write est_<day>.npz (Est/Spread) into, in the "
                        "layout compare/plot_reconstruction_sequence.py reads; only "
                        "valid when the arguments select a single variant")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def alpha_from_label(label):
    if not label.startswith("alpha"):
        raise ValueError(f"unexpected alpha label: {label}")
    return float(label[5:])


def selections(truth, observed, walkable):
    # channel_valid returns (C,T,H,W); restore the state layout (T,C,H,W).
    defined = np.ascontiguousarray(channel_valid(truth).transpose(1, 0, 2, 3))
    defined &= walkable[None, None]
    obs = np.broadcast_to(observed[:, None], truth.shape)
    walking = np.broadcast_to(walkable[None, None], truth.shape)
    return {
        "all": np.ones_like(truth, dtype=bool),
        "walkable": walking,
        "observed": obs,
        "blind": ~obs,
        "walkable_blind": walking & ~obs,
        "defined": defined,
        "defined_blind": defined & ~obs,
    }


def score(estimate, spread, truth, observed, walkable, warmup):
    estimate = torch.from_numpy(estimate[warmup:]).double()
    variance = torch.from_numpy(spread[warmup:]).double().square()
    truth_t = torch.from_numpy(truth[warmup:]).double()
    scale = torch.tensor(STATE_SCALE, dtype=torch.float64).view(1, 4, 1, 1)
    result = {}
    for name, selected in selections(truth, observed, walkable).items():
        selected_t = torch.from_numpy(np.ascontiguousarray(selected[warmup:]))
        acc = {key: 0.0 for key in
               ("se", "spread", "variance", "crps", "nll", "cov50", "cov90", "count")}
        accumulate(acc, estimate, variance, truth_t, selected_t, scale)
        result[name] = finalize(acc)
    return result


def main():
    args = parse_args()
    if args.frames < 2 or not 0 <= args.warmup < args.frames:
        raise SystemExit("require frames >= 2 and 0 <= warmup < frames")
    device = torch.device(args.device)
    calibration = json.loads(Path(args.calibration).read_text())
    blind_gammas = tuple(float(value) for value in args.blind_gammas.split(","))
    blind_empty_gammas = tuple(float(value) for value in args.blind_empty_gammas.split(","))
    if (not blind_gammas or not blind_empty_gammas
            or any(value <= 0 for value in blind_gammas + blind_empty_gammas)):
        raise SystemExit("blind gamma values must be positive")
    checkpoints = {"new_cov": args.new_checkpoint, "old_cov": args.old_checkpoint}
    variants = []
    model_names = ("new_cov",) if args.skip_old else tuple(checkpoints)
    for name in model_names:
        if name == "new_cov" and args.new_component_scales:
            parsed = []
            for spec in args.new_component_scales.split(","):
                factor_text, diagonal_text = spec.split(":", 1)
                parsed.append((float(factor_text), float(diagonal_text)))
            if any(factor < 0 or diagonal <= 0 for factor, diagonal in parsed):
                raise SystemExit("component scales require factor >= 0 and diagonal > 0")
            scales = {f"components_f{factor:g}_d{diagonal:g}": (factor, diagonal)
                      for factor, diagonal in parsed}
        elif name == "new_cov" and args.new_alphas:
            parsed = [float(value) for value in args.new_alphas.split(",")]
            scales = {f"sweep_alpha{alpha:g}": (alpha, alpha) for alpha in parsed}
            if any(alpha <= 0 for alpha in parsed):
                raise SystemExit("--new-alphas values must be positive")
        else:
            selected = calibration["selection"][name]["posterior"]
            scales = {
                "best_crps": (alpha_from_label(selected["best_crps_walkable_blind"]),) * 2,
                "best_calibrated": (alpha_from_label(
                    selected["best_calibrated_walkable_blind"]),) * 2,
            }
        for criterion, (factor_scale, diagonal_scale) in scales.items():
            for blind_gamma in blind_gammas:
                for empty_gamma in blind_empty_gammas:
                    if any(v[0] == name and math.isclose(v[2], factor_scale)
                           and math.isclose(v[3], diagonal_scale)
                           and math.isclose(v[5], blind_gamma)
                           and math.isclose(v[6], empty_gamma) for v in variants):
                        continue
                    variants.append((name, criterion, factor_scale, diagonal_scale,
                                     "full", blind_gamma, empty_gamma))
    if not args.skip_mean:
        variants.append(("new_cov", "mean_only", 1.0, 1.0, "mean", 1.0, 1.0))

    with np.load(args.obs) as z:
        total = min(args.frames, len(z["X_true"]))
        truth = z["X_true"][:total].astype(np.float64)
        observed = z["Omega"][:total].astype(bool)
    walkable = navigation.build_valid_mask_from_config().astype(bool)
    static_mask = torch.from_numpy(walkable).to(device)
    loaded = {name: load_models(path, device) for name, path in checkpoints.items()}
    results = {}
    exported = None
    for (name, criterion, factor_scale, diagonal_scale, mode, blind_gamma,
         empty_gamma) in variants:
        mean_net, cov_net, train_args, mean_path = loaded[name]
        if mode != "full":
            label = "new_mean_only"
        elif criterion.startswith("components_"):
            label = f"{name}_{criterion}"
        else:
            label = f"{name}_{criterion}_f{factor_scale:g}_d{diagonal_scale:g}"
        if mode == "full":
            label += f"_blindg{blind_gamma:g}_emptyg{empty_gamma:g}"
        print(f"[{label}] mode={mode} history={train_args.get('input_frames', 1)}", flush=True)
        estimate, spread, seconds = run_file(
            args.obs, mean_net, cov_net, static_mask, device, total, mode,
            factor_scale, diagonal_scale, int(train_args.get("input_frames", 1)),
            blind_gamma, empty_gamma)
        results[label] = {
            "model": name, "criterion": criterion,
            "alpha": factor_scale if math.isclose(factor_scale, diagonal_scale) else None,
            "factor_covariance_scale": factor_scale,
            "diagonal_covariance_scale": diagonal_scale, "mode": mode,
            "blind_diagonal_std_scale": blind_gamma,
            "blind_empty_diagonal_std_scale": empty_gamma,
            "input_frames": int(train_args.get("input_frames", 1)),
            "mean_checkpoint": mean_path, "seconds": seconds,
            "metrics": score(estimate, spread, truth, observed, walkable, args.warmup),
        }
        key = results[label]["metrics"]["walkable_blind"]
        print(json.dumps({"variant": label, "rmse": key["rmse"],
                          "crps": key["crps"], "spread_skill": key["spread_skill"]}),
              flush=True)

        if args.export_npz:
            # compare/plot_reconstruction_sequence.py reads one est_<day>.npz per
            # directory, so exporting a second variant here would silently replace
            # the first and mislabel whatever gets plotted.
            if exported is not None:
                raise SystemExit(
                    f"--export-npz takes a single variant; {exported!r} was already "
                    f"written and {label!r} would overwrite it. Narrow --blind-gammas "
                    f"/--blind-empty-gammas, or run one configuration per export.")
            exported = label
            day = Path(args.obs).stem.replace("obs_", "")
            export_dir = Path(args.export_npz)
            export_dir.mkdir(parents=True, exist_ok=True)
            np.savez(export_dir / f"est_{day}.npz", Est=estimate, Spread=spread)
            (export_dir / f"est_{day}.source.json").write_text(json.dumps({
                "variant": label, "produced_by": "eval_pedpred5_sequential.py",
                "obs": args.obs, "frames": total,
                "new_checkpoint": args.new_checkpoint,
            }, indent=2) + "\n")
            print(f"[export] {export_dir / f'est_{day}.npz'}  variant={label}", flush=True)

    output = {
        "config": vars(args), "frames": total, "warmup": args.warmup,
        "calibration_selection": calibration["selection"], "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n")
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
