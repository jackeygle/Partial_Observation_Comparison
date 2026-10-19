"""Check that a 4DVarNet checkpoint's test reconstruction does not depend on the window batch.

    python3 -m supervisor_evaluation.checks.batch_independence --ckpt <path> [--day atc-20130811]

Reconstructs one test day through compare5.run_varnet (the accuracy path) with batches of 1 and
16 windows and reports the largest absolute difference and the blind walkable RMSE of each.
Expected: ~float rounding for normg_scope='window'; large for the batch-normalised checkpoints.
"""
import argparse
import json

import numpy as np
import torch

from crowdcore import navigation, observation_model as om
from compare.compare5 import run_varnet
from methods.senseiver import dataset as senseiver_ds
from methods.varnet.checks.model_io import load_solver

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--day", default="atc-20130811")
a = ap.parse_args()
dev = torch.device("cuda")
fp = next(p for p in om.split_files("test") if a.day in str(p))
X, _ = om.load_state(str(fp)); X = np.asarray(X, np.float32)
cfg = senseiver_ds.obs_config()
obs = om.generate_observations(X, sensing_range=cfg["sensing_range"], num_agents=cfg["num_agents"],
                               add_noise=cfg["add_noise"], seed=om.day_seed(fp),
                               valid_mask=navigation.build_valid_mask_from_config(X),
                               obs_every_k=cfg["obs_every_k"])
Y, Om = obs["Y"].astype(np.float32), obs["Omega"].astype(bool)
Omc = np.repeat(Om[:, None], 4, axis=1)
X0 = om.fill_missing_state(Y, Omc, method=cfg["init_method"])
sol, args, _ = load_solver(a.ckpt, dev)
walk = navigation.build_valid_mask_from_config().astype(bool)
out = {"ckpt": a.ckpt, "day": a.day, "normg_scope": args.get("normg_scope", "batch")}
rec = {}
for b in (1, 16):
    rec[b], _ = run_varnet(sol, Y, Omc, X0, int(args["dT"]), dev, b)
for b, r in rec.items():
    lo = np.asarray([0, -5, -5, 0], np.float32)[None, :, None, None]
    hi = np.asarray([5, 5, 5, 2], np.float32)[None, :, None, None]
    rc = np.clip(r, lo, hi); n = len(rc)
    blind = walk[None] & ~Om[:n]
    out[f"blind_rmse_b{b}"] = float(np.sqrt(((rc - X[:n]) ** 2).transpose(1, 0, 2, 3)[:, blind].mean()))
out["max_abs_diff_b1_vs_b16"] = float(np.abs(rec[1] - rec[16]).max())
print(json.dumps(out, indent=1), flush=True)
