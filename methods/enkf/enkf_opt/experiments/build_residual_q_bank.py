"""Build structured process-noise samples from PedPred3 one-step residuals.

Each row is

    q_t = x_{t+1}^{true} - f(x_t^{true}) - mean_train_residual.

The mean residual is removed because the EnKF process noise must be zero mean; it
is stored separately for diagnosis.  A draw retains all cross-channel and spatial
correlations of one real forecast error field, unlike independent Gaussian Q.
Only TRAIN days are read, so the test-day evaluation cannot leak into this bank.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import h5py
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
OPT = os.path.join(ROOT, "methods", "enkf", "enkf_opt")
sys.path.insert(0, OPT)

from crowdcore import navigation, observation_model as om  # noqa: E402
from methods.enkf.lcskf.dynamics.model import load_pedpred3, surrogate_mean  # noqa: E402

H, W, F = 36, 12, 4
STATE_DIM = H * W * F
PROC_STD = np.asarray((0.02829307, 0.31263075, 0.12325809, 0.41680932))


def _clip_physical(x: np.ndarray) -> np.ndarray:
    y = x.reshape(-1, F, H, W).copy()
    np.clip(y[:, 0], 0, 5, out=y[:, 0])
    np.clip(y[:, 1:3], -5, 5, out=y[:, 1:3])
    np.clip(y[:, 3], 0, 2, out=y[:, 3])
    return y.reshape(-1, STATE_DIM)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20250922)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--model-checkpoint", default=os.path.join(
        ROOT, "methods", "enkf", "runs", "pedpred3_5to5_clip_s0", "dyn150_best.pt"))
    ap.add_argument("--input-frames", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(OPT, "experiments", "outputs",
                                                   "pedpred3_train_residual_q_8192.npz"))
    a = ap.parse_args()
    if a.samples < 2:
        raise ValueError("--samples must be at least 2")

    files = om.split_files("train")
    if not files:
        raise FileNotFoundError("training split contains no grid-cache files")
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    if a.input_frames < 1:
        raise ValueError("--input-frames must be positive")
    model_path = os.path.abspath(a.model_checkpoint)
    model = load_pedpred3(model_path, device).eval()

    # Allocate samples evenly across days.  This prevents a long day from dominating Q.
    per_day = np.full(len(files), a.samples // len(files), dtype=int)
    per_day[:a.samples % len(files)] += 1
    rng = np.random.RandomState(a.seed)
    residuals = np.empty((a.samples, STATE_DIM), dtype=np.float32)
    density_log_residuals = np.empty((a.samples, H * W), dtype=np.float32)
    forecast_density_mean = np.empty(a.samples, dtype=np.float32)
    forecast_density_delta = np.empty(a.samples, dtype=np.float32)
    walkable = navigation.build_valid_mask_from_config().reshape(-1)
    cursor = 0
    t0 = time.time()
    provenance = []
    for day_i, (path, count) in enumerate(zip(files, per_day)):
        if count == 0:
            continue
        with h5py.File(path, "r") as h5:
            grid = h5["grid"]
            if grid.shape[1:] != (F, H, W):
                raise ValueError(f"unexpected grid shape {grid.shape} in {path}")
            valid_idx = np.arange(a.input_frames - 1, grid.shape[0] - 1)
            idx = np.sort(rng.choice(valid_idx, size=count, replace=False))
            # Each forecast sees the same causal history convention as the final EnKF.
            # Reading these short slices explicitly avoids h5py's restriction on repeated
            # fancy indices when adjacent target histories overlap.
            xhist = np.stack([grid[int(j) - a.input_frames + 1:int(j) + 1]
                              for j in idx]).astype(np.float32)
            x1 = grid[idx + 1].astype(np.float32).reshape(count, STATE_DIM)
        for lo in range(0, count, a.batch):
            hi = min(lo + a.batch, count)
            with torch.inference_mode():
                hist = torch.from_numpy(xhist[lo:hi]).to(device)
                pred = surrogate_mean(model, hist, horizon=1)[:, 0]
                pred = pred.reshape(hi - lo, STATE_DIM).cpu().numpy()
            pred = _clip_physical(pred)
            residuals[cursor + lo:cursor + hi] = x1[lo:hi] - pred
            x1_density = x1[lo:hi].reshape(-1, F, H, W)[:, 0].reshape(-1, H * W)
            pred_density = pred.reshape(-1, F, H, W)[:, 0].reshape(-1, H * W)
            density_log_residuals[cursor + lo:cursor + hi] = (
                np.log1p(x1_density) - np.log1p(pred_density))
            x0_density = xhist[lo:hi, -1, 0].reshape(-1, H * W)
            forecast_density_mean[cursor + lo:cursor + hi] = pred_density[:, walkable].mean(1)
            forecast_density_delta[cursor + lo:cursor + hi] = (
                pred_density[:, walkable].mean(1) - x0_density[:, walkable].mean(1))
        provenance.append({"file": os.path.basename(path), "samples": int(count),
                           "first_index": int(idx[0]), "last_index": int(idx[-1])})
        cursor += count
        print(f"[{day_i + 1:02d}/{len(files)}] {os.path.basename(path)}: {count} "
              f"samples, total={cursor}/{a.samples}", flush=True)

    raw_mean = residuals.mean(axis=0, dtype=np.float64).astype(np.float32)
    residuals -= raw_mean[None]
    density_log_raw_mean = density_log_residuals.mean(
        axis=0, dtype=np.float64).astype(np.float32)
    density_log_residuals -= density_log_raw_mean[None]
    centered_mean = residuals.mean(axis=0, dtype=np.float64)
    ch_std = residuals.reshape(a.samples, F, -1).std(axis=(0, 2), dtype=np.float64)
    density_log_std = float(density_log_residuals.std(dtype=np.float64))
    model_hash = hashlib.sha256()
    with open(model_path, "rb") as model_file:
        for chunk in iter(lambda: model_file.read(8 << 20), b""):
            model_hash.update(chunk)
    model_sha256 = model_hash.hexdigest()
    meta = {
        "definition": (f"x_true[t+1] - clip(PedPred3(x_true[t-{a.input_frames-1}:t])) "
                       "- train_mean_residual"),
        "split": "train", "samples": a.samples, "seed": a.seed,
        "model": model_path, "model_sha256": model_sha256,
        "input_frames": a.input_frames, "device": device_name, "files": provenance,
        "raw_bias_channel_rms": np.sqrt((raw_mean.reshape(F, -1) ** 2).mean(axis=1)).tolist(),
        "centered_channel_std": ch_std.tolist(), "reference_proc_std": PROC_STD.tolist(),
        "density_log_definition": "log1p(x_true_density[t+1]) - log1p(clipped_prediction_density)",
        "density_log_raw_bias_rms": float(np.sqrt(np.mean(density_log_raw_mean ** 2))),
        "density_log_centered_std": density_log_std,
        "forecast_density_mean_quantiles": np.quantile(
            forecast_density_mean, (0, .25, .5, .75, 1)).tolist(),
        "forecast_density_delta_quantiles": np.quantile(
            forecast_density_delta, (0, .25, .5, .75, 1)).tolist(),
        "seconds": round(time.time() - t0, 2),
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, residuals=residuals, mean_residual=raw_mean,
                        density_log_residuals=density_log_residuals,
                        density_log_mean_residual=density_log_raw_mean,
                        forecast_density_mean=forecast_density_mean,
                        forecast_density_delta=forecast_density_delta,
                        metadata=np.asarray(json.dumps(meta)))
    print(f"[check] max |centered coordinate mean|={np.max(np.abs(centered_mean)):.3e}")
    print(f"[check] residual channel std={ch_std.tolist()}")
    print(f"[check] density log-residual std={density_log_std:.8f}")
    print(f"[out] {a.out}", flush=True)


if __name__ == "__main__":
    main()
