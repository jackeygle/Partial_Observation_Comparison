"""Verify the dt100/window-normalised smoke checkpoint on a CUDA device.

Checks use three real windows from the first training day. They establish
finite checkpoint values and batch grouping consistency, not test accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch

from crowdcore import observation_model
from methods.varnet.checks.model_io import load_solver
from methods.varnet.train import build_windows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; this check does not run model inference on CPU.")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Smoke checkpoint is missing: {args.checkpoint}")

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    solver, config, checkpoint = load_solver(str(args.checkpoint), device="cuda")
    expected = {"dT": 100, "kt": 3, "hidden": 96, "normg_scope": "window", "fold_axes": "ct"}
    for key, value in expected.items():
        if config.get(key) != value:
            raise SystemExit(f"Unexpected {key}: {config.get(key)!r}, expected {value!r}")
    if solver.n_iter != 20:
        raise SystemExit(f"Unexpected effective iteration count: {solver.n_iter}")

    nonfinite = [name for name, tensor in checkpoint["solver"].items()
                 if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all().item()]
    if nonfinite:
        raise SystemExit(f"Non-finite checkpoint tensors: {nonfinite}")
    metrics_path = args.checkpoint.parent / "metrics.jsonl"
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    if not metrics:
        raise SystemExit("Smoke metrics are missing.")
    latest = metrics[-1]
    scalars = {key: latest[key] for key in ("train_loss", "rec_unobs_mse", "r_score")}
    scalars.update(latest.get("per_channel", {}))
    if not all(math.isfinite(float(value)) for value in scalars.values()):
        raise SystemExit(f"Non-finite smoke metrics: {scalars}")
    print(f"[checkpoint] epoch={checkpoint['epoch']} step={checkpoint['step']} K={solver.n_iter}; finite", flush=True)

    files = observation_model.split_files("train")
    tensors = build_windows(
        files, dT=100, sensing_range=config["sensing_range"],
        num_agents=config["num_agents"], seed=config.get("data_seed", config.get("seed", 0)),
        max_days=1, add_noise=not config.get("no_noise", False),
        obs_every_k=config.get("obs_every_k", 1), trajectory_mode=config["trajectory_mode"],
    )
    n_windows = tensors[0].shape[0]
    indices = [n_windows // 4, n_windows // 2, 3 * n_windows // 4]
    selected = [tensor[indices].to("cuda") for tensor in tensors]
    del tensors
    truth, observations, mask, initial = selected
    for parameter in solver.parameters():
        parameter.requires_grad_(False)

    def predict(order: list[int]) -> torch.Tensor:
        # GradSolver requires state gradients even at inference.
        with torch.enable_grad():
            prediction = solver(initial[order].clone(), observations[order], mask[order])
        prediction = prediction.detach().cpu()
        if not torch.isfinite(prediction).all().item():
            raise SystemExit("Non-finite prediction during grouping check.")
        return prediction

    grouped = predict([0, 1, 2])
    single = torch.cat([predict([index]) for index in range(3)], dim=0)
    reversed_group = predict([2, 1, 0]).flip(0)
    checks = {}
    for name, other in (("batch3_vs_batch1", single), ("reversed_batch3", reversed_group)):
        delta = (grouped - other).abs()
        checks[name] = {"max_absolute_difference": float(delta.max()),
                        "mean_absolute_difference": float(delta.mean()),
                        "atol": 1e-5, "rtol": 1e-5,
                        "passed": bool(torch.allclose(grouped, other, atol=1e-5, rtol=1e-5))}
        print(f"[{name}] {checks[name]}", flush=True)

    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "gpu": torch.cuda.get_device_name(), "precision": "float32, TF32 disabled",
        "config": expected, "n_iter_effective": solver.n_iter,
        "smoke_epoch_metrics": latest,
        "data_day": str(files[0]), "window_indices": indices,
        "checks": checks, "passed": all(check["passed"] for check in checks.values()),
        "scope": "Functional smoke and batch grouping checks on training windows; not validation/test accuracy.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if not result["passed"]:
        raise SystemExit("Batch grouping check exceeded the stated float32 tolerance; inspect the JSON report.")


if __name__ == "__main__":
    main()
