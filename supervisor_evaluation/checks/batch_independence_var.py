"""Batch independence of an augmented-variance 4DVarNet: mean AND variance.

    python3 -m supervisor_evaluation.checks.batch_independence_var --ckpt <path> [--day atc-20130811]

Solves the first 8 windows of one test day with return_var=True, once as one batch of 8 and
once window by window, and reports the largest absolute difference of the mean and of the
variance. Expected ~float rounding for normg_scope='window'.
"""
import argparse
import json

import numpy as np
import torch

from crowdcore import navigation, observation_model as om
from methods.senseiver import dataset as senseiver_ds
from methods.varnet.checks.model_io import load_solver

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--day", default="atc-20130811")
ap.add_argument("--n", type=int, default=8)
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
if not sol.augmented_var:
    raise SystemExit("not an augmented-variance solver")
dT = int(args["dT"])
win = lambda z: torch.from_numpy(om.to_windows(z, dT)[:a.n]).float().to(dev)
Yw, Mw, X0w = win(Y), win(Omc.astype(np.float32)), win(X0)
sol.eval()
with torch.enable_grad():
    m8, v8 = sol.solve(X0w.clone(), Yw, Mw, return_var=True)
    ones = [sol.solve(X0w[i:i + 1].clone(), Yw[i:i + 1], Mw[i:i + 1], return_var=True)
            for i in range(a.n)]
m1 = torch.cat([o[0] for o in ones]); v1 = torch.cat([o[1] for o in ones])
out = {"ckpt": a.ckpt, "normg_scope": args.get("normg_scope", "batch"), "windows": a.n,
       "max_abs_diff_mean": float((m8 - m1).abs().max()),
       "max_abs_diff_var": float((v8 - v1).abs().max()),
       "max_rel_diff_var": float(((v8 - v1).abs() / v1.abs().clamp_min(1e-6)).max()),
       "finite": bool(torch.isfinite(m8).all() and torch.isfinite(v8).all())}
print(json.dumps(out, indent=1), flush=True)
