"""select_checkpoint.py — pick Senseiver's epoch on the WHOLE validation split.

train.py keeps `best.pt` by its own validation score, but that score is computed on
three of the seven validation days at every 20th frame (`--valid-days 3
--valid-stride 20`), while DINCAE and 4DVarNet choose their epoch on all seven
validation days at every frame (their checks/select_checkpoint.py). This script puts
Senseiver on the same footing: a run trained with `--save-every-epoch` leaves one
`epoch_NNN.pt` per epoch, and every one of them is scored here on

  * all seven validation days, every frame except the first and the last of a day
    (the frames the test protocol scores),
  * the observations of the routes the run was trained under (`trajectory_mode`
    and the observation seed from the run's own args, as DINCAE's and 4DVarNet's
    selection scripts do),
  * the blind walkable cells, all four channels, predictions clipped to the
    physical bounds -- the scope of the other methods' selection.

The epoch with the lowest pooled MSE is written to <run-dir>/select_valid.json.

Usage (GPU node):
    python3 -m methods.senseiver.checks.select_checkpoint --run-dir methods/senseiver/runs/senseiver_A_full
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import torch

from crowdcore import navigation as nav
from crowdcore import observation_model as om
from methods.senseiver import dataset as ds
from methods.senseiver import sensors
from methods.senseiver.network import Senseiver

LO = np.asarray([0, -5, -5, 0], np.float32)[None, :, None, None]
HI = np.asarray([5, 5, 5, 2], np.float32)[None, :, None, None]


def predict(model, Yf, Of, frames, dev):
    """Same code path as supervisor_evaluation/evaluate.py:_predict_senseiver_selected."""
    pe = model.pos_enc.detach().cpu().numpy()
    mean = model.in_mean.detach().cpu().numpy()
    std = model.in_std.detach().cpu().numpy()
    if model.time_window > 1:
        L = model.time_window
        windows = [(Yf[max(0, t - L + 1):t + 1], Of[max(0, t - L + 1):t + 1]) for t in frames]
        tok, pad, dt, _, cell = sensors.build_batch_temporal(windows, pe, mean, std,
                                                             return_cell_idx=True)
        with torch.no_grad():
            return model.reconstruct(tok.to(dev), pad.to(dev), dt.to(dev),
                                     cell.to(dev)).cpu().numpy()
    tok, pad, _ = sensors.build_batch(Yf[frames], Of[frames], pe, mean, std)
    with torch.no_grad():
        return model.reconstruct(tok.to(dev), pad.to(dev)).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--split", default="valid", choices=["valid"])
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--days", type=int, default=0, help="first N days only (smoke test)")
    ap.add_argument("--frames", type=int, default=0, help="first N frames per day (smoke test)")
    ap.add_argument("--epochs", default="",
                    help="score only epochs A-B (inclusive) and write select_valid_part_A_B.json, "
                         "so several jobs can share the work; then run with --merge")
    ap.add_argument("--merge", action="store_true",
                    help="combine the select_valid_part_*.json files into select_valid.json")
    a = ap.parse_args()
    if a.merge:                                   # bookkeeping only, no GPU needed
        merge(a.run_dir)
        return
    if not torch.cuda.is_available():
        raise SystemExit("No GPU: run this on a GPU node")
    dev = torch.device("cuda")
    ckpts = sorted(glob.glob(os.path.join(a.run_dir, "epoch_*.pt")))
    if a.epochs:
        lo, hi = (int(v) for v in a.epochs.split("-"))
        ckpts = [p for p in ckpts if lo <= int(re.search(r"epoch_(\d+)", p).group(1)) <= hi]
    if not ckpts:
        raise SystemExit(f"no epoch_*.pt in {a.run_dir}; train with --save-every-epoch")
    args0 = torch.load(ckpts[0], map_location="cpu", weights_only=False)["args"]
    mode = args0.get("trajectory_mode", "fixed")
    base = args0["seed"] if args0.get("valid_obs_seed") is None else args0["valid_obs_seed"]
    if args0.get("obs_seed") is not None and args0.get("valid_obs_seed") is None:
        base = args0["obs_seed"]
    cfg = ds.obs_config()
    walk = nav.build_valid_mask_from_config().astype(bool)

    files = om.split_files(a.split)
    if a.days:
        files = files[:a.days]
    days = []
    for fp in files:
        X, _ = om.load_state(str(fp))
        X = np.asarray(X, np.float32)
        if a.frames:
            X = X[:a.frames]
        T = len(X)
        obs = om.generate_observations(
            X, sensing_range=cfg["sensing_range"], num_agents=cfg["num_agents"],
            add_noise=cfg["add_noise"], seed=om.day_seed(fp, base, mode),
            valid_mask=nav.build_valid_mask_from_config(X), obs_every_k=cfg["obs_every_k"])
        Y, Om = obs["Y"].astype(np.float32), obs["Omega"].astype(bool)
        days.append((os.path.basename(str(fp)), X, Y.reshape(T, 4, -1), Om.reshape(T, -1),
                     walk[None] & ~Om))
        print(f"[data] {days[-1][0]}: {T} frames, observation seed "
              f"{om.day_seed(fp, base, mode)} ({mode})", flush=True)

    rows = []
    for p in ckpts:
        ck = torch.load(p, map_location=dev, weights_only=False)
        model = Senseiver(**ck["hparams"]).to(dev)
        model.load_state_dict(ck["model"])
        model.eval()
        se = n = 0.0
        for name, X, Yf, Of, blind in days:
            frames = list(range(1, len(X) - 1))
            for i in range(0, len(frames), a.chunk):
                f = frames[i:i + a.chunk]
                pred = np.clip(predict(model, Yf, Of, f, dev), LO, HI)
                b = np.repeat(blind[f][:, None], 4, axis=1)
                se += float(((pred - X[f]) ** 2)[b].sum())
                n += int(b.sum())
        ep = int(re.search(r"epoch_(\d+)", p).group(1))
        rows.append({"epoch": ep, "ckpt": os.path.basename(p), "mse": se / n,
                     "rmse": float(np.sqrt(se / n))})
        print(f"[select] epoch {ep:3d}  blind walkable RMSE {rows[-1]['rmse']:.5f}", flush=True)
        del model, ck
        torch.cuda.empty_cache()

    best = min(rows, key=lambda r: r["mse"])
    out = {"run_dir": os.path.abspath(a.run_dir), "split": a.split,
           "days": [d[0] for d in days], "frames": "all but the first and last of each day"
           if not a.frames else f"first {a.frames}", "trajectory_mode": mode,
           "observation_base_seed": base,
           "scope": "walkable (blind cells inside the walkable region, four channels, clipped)",
           "rows": rows, "selected_epoch": best["epoch"], "selected_ckpt": best["ckpt"],
           "selected_rmse": best["rmse"]}
    name = "select_valid.json" if not (a.days or a.frames) else "select_valid_smoke.json"
    if a.epochs and not (a.days or a.frames):
        name = f"select_valid_part_{lo:03d}_{hi:03d}.json"
    with open(os.path.join(a.run_dir, name), "w") as f:
        json.dump(out, f, indent=1)
    print(f"[select] selected epoch {best['epoch']} (RMSE {best['rmse']:.5f}) -> {name}", flush=True)


def merge(run_dir: str) -> None:
    """Combine the per-range parts; every epoch_*.pt of the run must be scored exactly once."""
    parts = sorted(glob.glob(os.path.join(run_dir, "select_valid_part_*.json")))
    docs = [json.load(open(p)) for p in parts]
    rows = sorted((r for d in docs for r in d["rows"]), key=lambda r: r["epoch"])
    want = sorted(int(re.search(r"epoch_(\d+)", p).group(1))
                  for p in glob.glob(os.path.join(run_dir, "epoch_*.pt")))
    if [r["epoch"] for r in rows] != want:
        raise SystemExit(f"parts cover epochs {[r['epoch'] for r in rows]}, run has {want}")
    for key in ("split", "days", "frames", "trajectory_mode", "observation_base_seed", "scope"):
        if len({json.dumps(d[key]) for d in docs}) != 1:
            raise SystemExit(f"parts disagree on {key}")
    best = min(rows, key=lambda r: r["mse"])
    out = {**{k: docs[0][k] for k in ("run_dir", "split", "days", "frames", "trajectory_mode",
                                      "observation_base_seed", "scope")},
           "parts": [os.path.basename(p) for p in parts], "rows": rows,
           "selected_epoch": best["epoch"], "selected_ckpt": best["ckpt"],
           "selected_rmse": best["rmse"]}
    with open(os.path.join(run_dir, "select_valid.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"[merge] {len(rows)} epochs; selected epoch {best['epoch']} "
          f"(RMSE {best['rmse']:.5f}) -> select_valid.json", flush=True)


if __name__ == "__main__":
    main()
