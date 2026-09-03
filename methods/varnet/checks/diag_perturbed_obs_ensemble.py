"""
diag_perturbed_obs_ensemble.py  —  uncertainty from perturbed observations, no new parameters

The classical way to get an analysis-error estimate out of a variational scheme without a
trained variance head: assimilate y + eps_i for i = 1..N with eps_i drawn from the observation
error distribution, and take the spread of the resulting analyses. See "An ensemble of perturbed
analyses to approximate the analysis error covariance in 4D-Var", Tellus A 2020.

Why it is worth measuring here. Our five members differ only in weight initialisation, so their
disagreement samples optimisation randomness, which is not the observation error. Perturbing y
samples the distribution we actually have a model of (config obs_std), and the spread it
produces has passed through Phi, so it is flow-dependent in the sense Beauchamp et al. (AIES
2025, Sec. 2c) mean when they say ensemble schemes counteract the observation-sampling issue.

Needs no retraining: one checkpoint, N solves per window.

    sbatch sbatch/submit_perturbed_obs.sbatch
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from crowdcore import config                                                        # noqa: E402
from crowdcore import observation_model as om                                       # noqa: E402
from crowdcore import navigation as nav                                             # noqa: E402
from methods.varnet.checks.model_io import load_solver                      # noqa: E402

OUT = os.path.join(ROOT, "check_outputs", "eval", "perturbed_obs_ensemble.json")
CHAN = list(config.get("grid", "channels"))
OBS_STD = np.asarray(config.get("observation", "obs_std"), np.float32)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = os.environ.get("AUDIT_CKPT", "runs/varnet_ml5_s0/varnet_best.pt")
NP_ENS = int(os.environ.get("N_PERT", 20))
NW = int(os.environ.get("AUDIT_WINDOWS", 24))
print(f"[device] {DEV}   ckpt {CKPT}   N={NP_ENS}", flush=True)

S, A, ck = load_solver(os.path.join(ROOT, CKPT), DEV)
S.eval()

DT = A["dT"]

_f = om.split_files("test")[0]
X_day, _ = om.load_state(_f)
valid = nav.build_valid_mask_from_config(X_day)
X_day = X_day[:(X_day.shape[0] // DT) * DT]
out = om.generate_observations(X_day, A["sensing_range"], A["num_agents"], add_noise=True,
                              seed=A.get("data_seed", 0) or 0, valid_mask=valid,
                              obs_every_k=1)
X0_day = om.fill_missing_state(out["Y"], out["Omega_c"],
                             method=config.get("observation", "init_method"))
w = lambda a: torch.from_numpy(om.to_windows(a, DT)).float()
_XW, _YW, _MW, _X0W = (w(X_day), w(out["Y"]),
                       w(out["Omega_c"].astype(np.float32)), w(X0_day))
IDX = np.linspace(0, _XW.shape[0] - 1, min(NW, _XW.shape[0])).round().astype(int)
print(f"[data] {os.path.basename(_f)}  {len(IDX)} windows, {NP_ENS} perturbations each",
      flush=True)

STD = torch.from_numpy(OBS_STD).to(DEV).reshape(1, len(CHAN), 1, 1, 1)
gen = torch.Generator(device="cpu").manual_seed(0)

acc = {k: [] for k in ("x", "mu", "sig_pert", "sig_head", "mask")}
for wi in IDX:
    Xb, Yb = _XW[wi:wi + 1].to(DEV), _YW[wi:wi + 1].to(DEV)
    Mb, X0b = _MW[wi:wi + 1].to(DEV), _X0W[wi:wi + 1].to(DEV)
    xs = []
    for _ in range(NP_ENS):
        # eps only where something was observed: perturbing a blind cell would be perturbing 0
        eps = torch.randn(Yb.shape, generator=gen).to(DEV) * STD * Mb
        with torch.enable_grad():
            xs.append(S(X0b.clone(), Yb + eps, Mb).detach())
    xs = torch.stack(xs)
    mu = xs.mean(0)
    acc["mu"].append(mu.cpu())
    acc["sig_pert"].append(xs.std(0, unbiased=False).cpu())
    acc["x"].append(Xb.cpu())
    acc["mask"].append(Mb.cpu())
    if S.grad_net.out_var is not None:         # the model's own sigma on the unperturbed solve
        with torch.enable_grad():
            _, vh = S(X0b.clone(), Yb, Mb, return_var=True)
        acc["sig_head"].append(vh.detach().sqrt().cpu())
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
C = {k: torch.cat(v) for k, v in acc.items() if v}
print(f"[done] {tuple(C['mu'].shape)}", flush=True)


def score(mu, sig, x, tag):
    err = (x - mu)
    rmse = float(torch.sqrt((err ** 2).mean()))
    return dict(tag=tag, rmse=rmse, sigma_mean=float(sig.mean()),
                spread_skill=float(sig.mean() / rmse),
                cov90=float(((err.abs() <= 1.6449 * sig).float()).mean()),
                n=int(x.numel()))


R = {}
x, mu, m = C["x"], C["mu"], C["mask"]
R["perturbed_obs"] = score(mu, C["sig_pert"], x, "perturbed observations")
if "sig_head" in C:
    R["head"] = score(mu, C["sig_head"], x, "the trained head, same windows")
blind, obs = m == 0, m == 1
for nm, sel in (("blind", blind), ("observed", obs)):
    R[f"perturbed_obs_{nm}"] = score(mu[sel], C["sig_pert"][sel], x[sel], nm)
    if "sig_head" in C:
        R[f"head_{nm}"] = score(mu[sel], C["sig_head"][sel], x[sel], nm)
for c, nm in enumerate(CHAN):
    R[f"perturbed_obs_channel_{nm}"] = score(mu[:, c], C["sig_pert"][:, c], x[:, c], nm)

json.dump(dict(ckpt=CKPT, day=os.path.basename(_f), windows=len(IDX), n_pert=NP_ENS,
               obs_std=OBS_STD.tolist(), results=R), open(OUT, "w"), indent=1)
print(f"[json] {OUT}\n")
print(f"{'':<26}{'sigma':>9}{'RMSE':>9}{'sp/sk':>8}{'90%':>8}")
for k in sorted(R):
    d = R[k]
    print(f"  {k:<24}{d['sigma_mean']:>9.4f}{d['rmse']:>9.4f}"
          f"{d['spread_skill']:>8.3f}{d['cov90'] * 100:>7.1f}%")
