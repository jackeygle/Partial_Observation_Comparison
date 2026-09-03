"""
diag_collapse.py  —  has our own uncertainty collapsed, and where?

The EnKF this thesis compares against has spread/skill 0.011: its ensemble spread has fallen to
~1% of the actual error and the predictive likelihood diverges. Ours reads 2.49 in aggregate,
i.e. the opposite sign, but an aggregate cannot rule out collapse in two places:

  (a) PER CELL. sigma^2 = softplus(.) + 1e-6, and Lakshminarayanan et al.'s footnote-2 floor is
      there precisely because the NLL is unbounded below as sigma^2 -> 0. If a slice of cells is
      sitting on that floor, the NLL blows up there exactly as the EnKF's does.
  (b) ACROSS MEMBERS. Sec. 2.4 splits the total into mean_m sigma_m^2 (aleatoric) plus
      Var_m(mu_m) (epistemic). The epistemic half IS an ensemble spread and can collapse the
      same way an EnKF's does -- five members that agree with each other are five members that
      have stopped being an ensemble.

Reports, on one test day: the sigma^2 distribution against the floor, the epistemic share of the
total variance, and how far the five members actually differ from one another.

    sbatch sbatch/submit_collapse.sbatch
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "checks"))
import config                                                        # noqa: E402
import observation_model as om                                       # noqa: E402
import navigation as nav                                             # noqa: E402
from model_io import load_solver                                     # noqa: E402

OUT = os.path.join(ROOT, "check_outputs", "eval", "collapse.json")
CHAN = list(config.get("grid", "channels"))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FMT = os.environ.get("AUDIT_FMT", "runs/varnet_vsb0_s{}")
NW = int(os.environ.get("AUDIT_WINDOWS", 12))

solvers, A = [], None
for s in range(5):
    sol, A, _ = load_solver(os.path.join(ROOT, FMT.format(s), "varnet_best.pt"), DEV)
    solvers.append(sol)
DT, EPSF = A["dT"], float(A.get("var_eps", 1e-6))
K = A.get("obs_every_k") or config.get("observation", "obs_every_k")
print(f"[members] 5  dT={DT}  var_eps={EPSF:g}", flush=True)

X = np.asarray(om.load_state(om.split_files("test")[0])[0])
X = X[:(len(X) // DT) * DT]
valid = nav.build_valid_mask_from_config(X)
o = om.generate_observations(X, add_noise=True, valid_mask=valid, obs_every_k=K)
x0 = om.fill_missing_state(o["Y"], o["Omega_c"],
                           method=config.get("observation", "init_method"))
w = lambda a: torch.from_numpy(om.to_windows(a, DT)).float()
Yw, Mw, X0w, Xw = w(o["Y"]), w(o["Omega_c"].astype(np.float32)), w(x0), w(np.asarray(X))
NW = min(NW, Xw.shape[0])

S2, ALE, EPI, ERR, MU_M = [], [], [], [], []
for i in range(NW):
    yb, mb, x0b = (t[i:i+1].to(DEV) for t in (Yw, Mw, X0w))
    xb = Xw[i:i+1].to(DEV)
    mus, vs = [], []
    for sol in solvers:
        with torch.enable_grad():
            xr, vr = sol(x0b.clone(), yb, mb, return_var=True)
        mus.append(xr.detach()); vs.append(vr.detach())
    mu, var = torch.stack(mus), torch.stack(vs)
    f = lambda t: t.reshape(-1).cpu().numpy()
    S2.append(f(var))                                   # every member's sigma^2, pooled
    ALE.append(f(var.mean(0)))
    EPI.append(f(mu.var(0, unbiased=False)))
    ERR.append(f((mu.mean(0) - xb) ** 2))
    MU_M.append(mu.reshape(5, -1).cpu().numpy())
    del mu, var, mus, vs
    if DEV.type == "cuda":
        torch.cuda.empty_cache()

s2 = np.concatenate(S2); ale = np.concatenate(ALE)
epi = np.concatenate(EPI); err = np.concatenate(ERR)
mu_m = np.concatenate(MU_M, axis=1)

res = {"var_eps": EPSF, "n_cells": int(ale.size), "windows": NW}

print("\n=== (a) per-cell: is sigma^2 sitting on the floor? ===", flush=True)
res["sigma2"] = {"min": float(s2.min()), "q": {}}
for q in (0.001, 0.01, 0.1, 0.5, 0.9, 0.99):
    v = float(np.quantile(s2, q)); res["sigma2"]["q"][str(q)] = v
    print(f"  quantile {q:<6} sigma^2 = {v:.3e}   ({v/EPSF:,.0f}x the floor)", flush=True)
for mult in (1.01, 2, 10, 100):
    fr = float(np.mean(s2 < mult * EPSF))
    res["sigma2"][f"frac_below_{mult}x_floor"] = fr
    print(f"  fraction below {mult:>5}x floor: {fr*100:.4f}%", flush=True)

print("\n=== (b) across members: has the ensemble collapsed? ===", flush=True)
tot = ale + epi
res["decomposition"] = {
    "epistemic_share_of_variance": float(epi.mean() / tot.mean()),
    "aleatoric_share_of_variance": float(ale.mean() / tot.mean()),
    "epistemic_spread_skill": float(np.sqrt(epi.mean() / err.mean())),
    "aleatoric_spread_skill": float(np.sqrt(ale.mean() / err.mean())),
    "total_spread_skill": float(np.sqrt(tot.mean() / err.mean()))}
for k, v in res["decomposition"].items():
    print(f"  {k:34s} {v:.4f}", flush=True)

# how different are the five members, really?
rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
pair = [rms(mu_m[i] - mu_m[j]) for i in range(5) for j in range(i + 1, 5)]
res["members"] = {"pairwise_rms_diff_mean": float(np.mean(pair)),
                  "pairwise_rms_diff_min": float(np.min(pair)),
                  "pairwise_rms_diff_max": float(np.max(pair)),
                  "reconstruction_rms": rms(mu_m.mean(0)),
                  "err_rms": float(np.sqrt(err.mean()))}
m = res["members"]
print(f"  pairwise RMS difference between members: mean {m['pairwise_rms_diff_mean']:.5f} "
      f"(min {m['pairwise_rms_diff_min']:.5f}, max {m['pairwise_rms_diff_max']:.5f})", flush=True)
print(f"  for scale: the ensemble's own error RMS is {m['err_rms']:.5f}  -> members differ from "
      f"each other by {m['pairwise_rms_diff_mean']/m['err_rms']*100:.1f}% of the error they make",
      flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=2)
print(f"\n[written] {OUT}")
