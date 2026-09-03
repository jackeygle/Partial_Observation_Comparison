"""
eval_sigma_fair.py  —  does the learnt sigma^2 beat trivial alternatives, fitted FAIRLY?

The earlier reading of checks/eval_uncertainty.py said the learnt sigma^2 loses to a constant.
That comparison was rigged in the constant's favour: its value was
sqrt(mean((x - mu)^2)) computed on THE SAME DATA it was scored on. A baseline fitted on the test
set is not a baseline. Two things had to be settled before any claim:

  (a) give the constant only validation data, like any honest baseline;
  (b) let the learnt sigma^2 have ONE free parameter too. It came out at spread/skill 2.49,
      i.e. uniformly ~2.5x too large, and a single global rescale would fix that without
      touching its spatial structure. If the rescaled version then wins, the finding is
      "the learnt sigma^2 has useful structure but the wrong overall scale", which is a
      completely different sentence from "it loses to a constant".

Four contenders. Every fitted quantity comes from the VALID split and is applied unchanged to
the 7 TEST days, so none of them sees the test set:

  raw        the ensemble's sigma_tot as trained                        0 fitted params
  scaled     sigma_tot * s                                             1
  constant   a single sigma for every cell                             1
  lookup     sigma^2 = E[err^2 | |x_hat|], 24 quantile bins            24

s and the constant are fitted by minimising validation NLL, which is the criterion they are
then judged on -- the same protocol Gal & Ghahramani Sec. 5.3 use for tau (they use Bayesian
optimisation; one scalar needs only a grid).

sigma_tot is Sec. 2.4 moment matching: sqrt(mean_m sigma_m^2 + Var_m(mu_m)).

    sbatch sbatch/submit_sigma_fair.sbatch
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

OUT = os.path.join(ROOT, "check_outputs", "eval", "sigma_fair.json")
CHAN = list(config.get("grid", "channels"))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FMT = os.environ.get("AUDIT_FMT", "runs/varnet_vsb0_s{}")
MEMBERS = [int(z) for z in os.environ.get("AUDIT_MEMBERS", "0,1,2,3,4").split(",")]
N_TEST_DAYS = int(os.environ.get("AUDIT_TEST_DAYS", 7))
BATCH = int(os.environ.get("AUDIT_BATCH", 2))
NBIN = 24
EPS = 1e-6

solvers, A = [], None
for s in MEMBERS:
    p = os.path.join(ROOT, FMT.format(s), "varnet_best.pt")
    sol, A, _ = load_solver(p, DEV)
    if sol.grad_net.out_var is None:
        raise SystemExit(f"{p} has no variance read-out")
    solvers.append(sol)
DT = A["dT"]
K = A.get("obs_every_k") or config.get("observation", "obs_every_k")
print(f"[members] {len(solvers)} x {FMT}  dT={DT}  k={K}  dev={DEV}", flush=True)


def harvest(files):
    """Ensemble mu*, sigma_tot, |x_hat|, truth and mask over the given days."""
    acc = {t: [] for t in ("mu", "sig", "x", "mask")}
    for f in files:
        X = np.asarray(om.load_state(f)[0])
        X = X[:(len(X) // DT) * DT]
        valid = nav.build_valid_mask_from_config(X)
        o = om.generate_observations(X, add_noise=True, valid_mask=valid, obs_every_k=K)
        x0 = om.fill_missing_state(o["Y"], o["Omega_c"],
                                   method=config.get("observation", "init_method"))
        w = lambda a: torch.from_numpy(om.to_windows(a, DT)).float()
        Yw, Mw, X0w, Xw = (w(o["Y"]), w(o["Omega_c"].astype(np.float32)), w(x0),
                           w(np.asarray(X)))
        for i in range(0, Xw.shape[0], BATCH):
            yb, mb, x0b = (t[i:i+BATCH].to(DEV) for t in (Yw, Mw, X0w))
            mus, vs = [], []
            for sol in solvers:
                with torch.enable_grad():
                    xr, vr = sol(x0b.clone(), yb, mb, return_var=True)
                mus.append(xr.detach()); vs.append(vr.detach())
            mu, var = torch.stack(mus), torch.stack(vs)
            tot = var.mean(0) + mu.var(0, unbiased=False)     # Sec. 2.4
            acc["mu"].append(mu.mean(0).cpu()); acc["sig"].append(tot.sqrt().cpu())
            acc["x"].append(Xw[i:i+BATCH]); acc["mask"].append(mb.cpu())
            del mu, var, mus, vs, xr, vr
            if DEV.type == "cuda":
                torch.cuda.empty_cache()
        print(f"  {os.path.basename(f)}  {Xw.shape[0]} windows", flush=True)
    return {t: torch.cat(v).numpy() for t, v in acc.items()}


V = harvest(om.split_files("valid")[:1])
T = harvest(om.split_files("test")[:N_TEST_DAYS])
for tag, D in (("valid", V), ("test", T)):
    print(f"[{tag}] {D['x'].size:,} cells", flush=True)

nll = lambda s, e2: float(np.mean(np.log(s ** 2) / 2 + e2 / (2 * s ** 2)))
eV, eT = (V["x"] - V["mu"]) ** 2, (T["x"] - T["mu"]) ** 2

# --- fit the one-parameter contenders on VALID only -------------------------------
grid_s = np.logspace(-1.5, 0.7, 89)
scale = float(min(grid_s, key=lambda z: nll(V["sig"] * z, eV)))
grid_c = np.logspace(-3, 0.5, 89)
const = float(min(grid_c, key=lambda z: nll(np.full_like(V["sig"], z), eV)))
# --- and the lookup, also on VALID only ------------------------------------------
aV, aT = np.abs(V["mu"]), np.abs(T["mu"])
edges = np.unique(np.quantile(aV, np.linspace(0, 1, NBIN + 1)))
bV = np.clip(np.digitize(aV, edges[1:-1]), 0, len(edges) - 2)
table = np.array([eV[bV == b].mean() if (bV == b).any() else eV.mean()
                  for b in range(len(edges) - 1)])
sig_lookup = np.sqrt(np.maximum(
    table[np.clip(np.digitize(aT, edges[1:-1]), 0, len(edges) - 2)], EPS))
print(f"[fit on valid] scale s = {scale:.4f}   constant sigma = {const:.4f}   "
      f"lookup bins = {len(table)}", flush=True)
for nm, val, g in (("scale", scale, grid_s), ("constant", const, grid_c)):
    if val in (g[0], g[-1]):
        print(f"[warn] {nm} sits on the grid edge ({val:.4g}); widen it", flush=True)

CONTENDERS = {"raw": T["sig"], "scaled": T["sig"] * scale,
              "constant": np.full_like(T["sig"], const), "lookup": sig_lookup}

from scipy.stats import norm                                          # noqa: E402
blind = T["mask"] == 0
empty = np.repeat(T["x"][:, 0:1] == 0, len(CHAN), axis=1)


def report(sel, label):
    e2, out = eT[sel], {}
    for nm, sg in CONTENDERS.items():
        s = sg[sel]
        out[nm] = {"nll": nll(s, e2),
                   "spread_skill": float(s.mean() / np.sqrt(e2.mean())),
                   "cov90": float(np.mean(np.abs(np.sqrt(e2)) <= norm.ppf(0.95) * s)),
                   "sigma_mean": float(s.mean())}
    print(f"\n=== {label}  ({sel.sum():,} cells) ===", flush=True)
    print("  %-9s %9s %8s %8s" % ("", "NLL", "sp/sk", "90%cov"), flush=True)
    for nm, q in out.items():
        print("  %-9s %9.4f %8.2f %7.1f%%" % (nm, q["nll"], q["spread_skill"],
                                              q["cov90"] * 100), flush=True)
    return out


res = {"fit": {"scale": scale, "constant": const, "n_bins": int(len(table))},
       "members": MEMBERS, "run_fmt": FMT, "test_days": N_TEST_DAYS,
       "valid_day": os.path.basename(om.split_files("valid")[0]),
       "splits": {}}
res["splits"]["all"] = report(np.ones_like(blind, dtype=bool), "all cells")
res["splits"]["blind"] = report(blind, "blind cells")
res["splits"]["blind_empty"] = report(blind & empty, "blind AND truly empty")
res["splits"]["blind_occupied"] = report(blind & ~empty, "blind AND occupied")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=2)
print(f"\n[written] {OUT}")
