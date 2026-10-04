"""channel_scores.py — per-channel validation RMSE of given Senseiver checkpoints, for the
window rule of supervisor_evaluation/RESELECTION_PROTOCOL.md (no channel more than 5% above
k = 1). Same days, frames, cells, observations and clipping as select_checkpoint.py.

    python3 -m methods.senseiver.checks.channel_scores --out <json> name=path [name=path ...]
"""
from __future__ import annotations

import argparse
import os
import json

import numpy as np
import torch

from crowdcore import navigation as nav
from crowdcore import observation_model as om
from methods.senseiver import dataset as ds
from methods.senseiver.checks.select_checkpoint import HI, LO, predict
from methods.senseiver.network import Senseiver

CHANNELS = ("density", "vx", "vy", "var")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+", help="name=path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=512)
    a = ap.parse_args()
    dev = torch.device("cuda")
    models = {}
    for spec in a.ckpts:
        name, path = spec.split("=", 1)
        ck = torch.load(path, map_location=dev, weights_only=False)
        m = Senseiver(**ck["hparams"]).to(dev); m.load_state_dict(ck["model"]); m.eval()
        mode = ck["args"].get("trajectory_mode", "fixed")
        base = ck["args"]["seed"] if ck["args"].get("valid_obs_seed") is None else ck["args"]["valid_obs_seed"]
        models[name] = (m, mode, base, path)
    if len({(v[1], v[2]) for v in models.values()}) != 1:
        raise SystemExit("candidates were trained on different routes; not comparable")
    _, mode, base, _ = next(iter(models.values()))
    cfg = ds.obs_config()
    walk = nav.build_valid_mask_from_config().astype(bool)
    se = {k: np.zeros(4) for k in models}; n = 0
    for fp in om.split_files("valid"):
        X = np.asarray(om.load_state(str(fp))[0], np.float32); T = len(X)
        obs = om.generate_observations(
            X, sensing_range=cfg["sensing_range"], num_agents=cfg["num_agents"],
            add_noise=cfg["add_noise"], seed=om.day_seed(fp, base, mode),
            valid_mask=nav.build_valid_mask_from_config(X), obs_every_k=cfg["obs_every_k"])
        Yf = obs["Y"].astype(np.float32).reshape(T, 4, -1)
        Om = obs["Omega"].astype(bool); Of = Om.reshape(T, -1)
        blind = walk[None] & ~Om
        frames = list(range(1, T - 1))
        n += int(blind[frames].sum())
        for i in range(0, len(frames), a.chunk):
            f = frames[i:i + a.chunk]; b = blind[f]
            for k, (m, *_rest) in models.items():
                p = np.clip(predict(m, Yf, Of, f, dev), LO, HI)
                d2 = (p - X[f]) ** 2
                se[k] += np.array([d2[:, c][b].sum() for c in range(4)])
        print(f"[channels] {os.path.basename(str(fp)).split('_')[0]} done", flush=True)
    out = {k: {"checkpoint": models[k][3],
               "pooled_rmse": float(np.sqrt(se[k].sum() / (4 * n))),
               **{f"{c}_rmse": float(np.sqrt(se[k][i] / n)) for i, c in enumerate(CHANNELS)}}
           for k in models}
    json.dump(out, open(a.out, "w"), indent=2)
    for k, v in out.items():
        print(k, {x: round(y, 5) for x, y in v.items() if x != "checkpoint"}, flush=True)


if __name__ == "__main__":
    main()
