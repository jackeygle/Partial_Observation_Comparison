"""
eval_threeway_accuracy.py  —  state-estimation accuracy for all three methods, one code path.

Why this exists rather than collecting the numbers that already sit in check_outputs/eval: they
are not the same quantity. test_metrics_b0_k1.json reports the MEAN OF PER-DAY RMSEs, while
uncertainty_vsb0.json pools every cell across the 7 days and takes one RMSE. Those differ, and
putting them side by side in a chart would be an error, not a rounding difference. Everything
here is pooled the same way, over the same days, from the same observation realisation.

What is scored:
  enkf              the EnKF's own exported estimate (check_outputs/enkf_k1_full)
  varnet_mse        a single 4DVarNet trained on Eq. 14, the plain squared loss
  varnet_nll_each   EACH of the 5 NLL-trained members, scored on its own -> mean +- std
  varnet_nll_ens5   the 5-member ensemble, i.e. the mean of their reconstructions (Sec. 2.4)

Why both of the last two. The ensemble exists to produce an uncertainty, but its point estimate
is the members' mean, so an accuracy table that only shows the ensemble mixes the benefit of
AVERAGING with the effect of the LOSS. There is also no such thing as "the" single member --
there are five, so the honest single-model figure is their mean and spread, the same way
Lakshminarayanan et al. report mean +- error over their folds. If the ensemble lands below the
best individual member, averaging did something that picking a lucky seed would not.

Their Sec. 3.3 expects the NLL arm to pay some RMSE for its likelihood ("our method is slightly
worse in terms of RMSE ... because our method optimizes for NLL"). Whether that holds here, and
whether averaging more than pays it back, is what these rows answer. Note the current MSE arm
(runs/varnet_b0_k1) also differs in ARCHITECTURE, so it is not yet a single-variable contrast;
runs/varnet_mse5_s* are training for exactly that.

    sbatch sbatch/submit_threeway_accuracy.sbatch
"""
from __future__ import annotations
import json
import os
from crowdcore import paths
import sys

import numpy as np
import torch

# 这个脚本原来住在 4dvarnet_enkf/ 下，ROOT 一直指那个目录（runs/、check_outputs/
# 都挂在它下面）。2026-09-03 重构后它搬到了顶层，dirname(dirname(__file__)) 会变成
# 仓库根，于是每一条 os.path.join(ROOT, ...) 都会静默指错地方 —— 所以显式绑定。
ROOT = paths.method(paths.VARNET)
from crowdcore import config                                                        # noqa: E402
from crowdcore import observation_model as om                                       # noqa: E402
from crowdcore import navigation as nav                                             # noqa: E402
from methods.varnet.checks.model_io import load_solver                                     # noqa: E402

OUT = os.path.join(ROOT, "check_outputs", "eval", "threeway_accuracy.json")
ENKF_SRC = os.path.join(ROOT, "check_outputs", "enkf_k1_full")
CHAN = list(config.get("grid", "channels"))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MSE_RUN = os.environ.get("AUDIT_MSE", "runs/varnet_b0_k1")
DE_FMT = os.environ.get("AUDIT_DE", "runs/varnet_vsb0_s{}")
DE_MEMBERS = [int(z) for z in os.environ.get("AUDIT_DE_MEMBERS", "0,1,2,3,4").split(",")]
N_DAYS = int(os.environ.get("AUDIT_DAYS", 7))
BATCH = int(os.environ.get("AUDIT_BATCH", 2))

mse_solver, A_M, _ = load_solver(os.path.join(ROOT, MSE_RUN, "varnet_best.pt"), DEV)
de_solvers, A_D = [], None
for m in DE_MEMBERS:
    sol, A_D, _ = load_solver(os.path.join(ROOT, DE_FMT.format(m), "varnet_best.pt"), DEV)
    de_solvers.append(sol)
assert A_M["dT"] == A_D["dT"], (A_M["dT"], A_D["dT"])
DT = A_M["dT"]
K = A_D.get("obs_every_k") or config.get("observation", "obs_every_k")
print(f"[cfg] dT={DT} k={K} days={N_DAYS}  mse={MSE_RUN}  de={DE_FMT} x{len(de_solvers)}",
      flush=True)

# accumulated sums of squares, so the pooled RMSE is exact rather than an average of averages
_names = ["enkf", "varnet_mse", "varnet_nll_ens5"] + \
         [f"varnet_nll_s{m}" for m in DE_MEMBERS]
_blank = lambda: {"se": 0.0, "n": 0, "se_blind": 0.0, "n_blind": 0, "se_obs": 0.0, "n_obs": 0,
                  "se_ch": np.zeros(len(CHAN)), "n_ch": np.zeros(len(CHAN))}
acc = {m: _blank() for m in _names}


def add(m, est, truth, blind):
    a = acc[m]
    d2 = (est - truth) ** 2
    a["se"] += float(d2.sum()); a["n"] += d2.size
    a["se_blind"] += float(d2[blind].sum()); a["n_blind"] += int(blind.sum())
    a["se_obs"] += float(d2[~blind].sum()); a["n_obs"] += int((~blind).sum())
    for c in range(len(CHAN)):
        a["se_ch"][c] += float(d2[:, c].sum()); a["n_ch"][c] += d2[:, c].size


files = om.split_files("test")[:N_DAYS]
for f in files:
    day = os.path.basename(f).split("_")[0]
    X = np.asarray(om.load_state(f)[0])
    X = X[:(len(X) // DT) * DT]
    valid = nav.build_valid_mask_from_config(X)
    o = om.generate_observations(X, add_noise=True, valid_mask=valid, obs_every_k=K)
    x0 = om.fill_missing_state(o["Y"], o["Omega_c"],
                               method=config.get("observation", "init_method"))
    Om = o["Omega_c"].astype(bool)
    blind_all = ~Om

    # --- the EnKF's own export, restricted to the same frames -----------------
    ez = os.path.join(ENKF_SRC, f"est_{day}.npz")
    if os.path.exists(ez):
        est = np.load(ez)["Est"][:X.shape[0]]
        add("enkf", est, X, blind_all[:est.shape[0]])
    else:
        print(f"  [warn] no EnKF export for {day}", flush=True)

    # --- the two learnt methods ----------------------------------------------
    w = lambda a: torch.from_numpy(om.to_windows(a, DT)).float()
    Yw, Mw, X0w, Xw = (w(o["Y"]), w(o["Omega_c"].astype(np.float32)), w(x0), w(np.asarray(X)))
    for i in range(0, Xw.shape[0], BATCH):
        yb, mb, x0b = (t[i:i+BATCH].to(DEV) for t in (Yw, Mw, X0w))
        xb = Xw[i:i+BATCH].numpy()
        bl = (mb == 0).cpu().numpy()
        with torch.enable_grad():
            xm = mse_solver(x0b.clone(), yb, mb).detach().cpu().numpy()
        add("varnet_mse", xm, xb, bl)
        mus = []
        for mi, sol in zip(DE_MEMBERS, de_solvers):
            with torch.enable_grad():
                mu = sol(x0b.clone(), yb, mb).detach()
            add(f"varnet_nll_s{mi}", mu.cpu().numpy(), xb, bl)   # each member on its own
            mus.append(mu)
        add("varnet_nll_ens5", torch.stack(mus).mean(0).cpu().numpy(), xb, bl)
        del mus, mu
        if DEV.type == "cuda":
            torch.cuda.empty_cache()
    print(f"  {day}  {Xw.shape[0]} windows", flush=True)

res = {"days": len(files), "dT": DT, "obs_every_k": K, "mse_run": MSE_RUN,
       "de_fmt": DE_FMT, "de_members": DE_MEMBERS,
       "pooling": "sum of squared errors over every cell of every day, then one sqrt",
       "methods": {}}
for m, a in acc.items():
    if a["n"] == 0:
        continue
    r = lambda se, n: float(np.sqrt(se / n)) if n else None
    res["methods"][m] = {
        "rmse_all": r(a["se"], a["n"]),
        "rmse_blind": r(a["se_blind"], a["n_blind"]),
        "rmse_observed": r(a["se_obs"], a["n_obs"]),
        "per_channel": {CHAN[c]: r(a["se_ch"][c], a["n_ch"][c]) for c in range(len(CHAN))},
        "n_cells": a["n"]}

# the five members summarised as mean +- std, since "the single model" is not one number
_ind = [res["methods"][f"varnet_nll_s{m}"] for m in DE_MEMBERS
        if f"varnet_nll_s{m}" in res["methods"]]
if _ind:
    res["nll_single_member_summary"] = {
        f"rmse_{k}": {"mean": float(np.mean([q[f"rmse_{k}"] for q in _ind])),
                      "std": float(np.std([q[f"rmse_{k}"] for q in _ind])),
                      "min": float(np.min([q[f"rmse_{k}"] for q in _ind])),
                      "max": float(np.max([q[f"rmse_{k}"] for q in _ind]))}
        for k in ("all", "blind", "observed")}

hdr = "%-18s %9s %9s %9s   %s" % ("", "all", "blind", "observed",
                                  "  ".join(f"{c:>8s}" for c in CHAN))
print("\n" + hdr, flush=True)
for m in ("enkf", "varnet_mse"):
    if m not in res["methods"]:
        continue
    q = res["methods"][m]
    print("%-18s %9.4f %9.4f %9.4f   %s" % (
        m, q["rmse_all"], q["rmse_blind"], q["rmse_observed"],
        "  ".join(f"{q['per_channel'][c]:8.4f}" for c in CHAN)), flush=True)
if _ind:
    S = res["nll_single_member_summary"]
    print("%-18s %9s %9s %9s" % (
        f"nll, each of {len(_ind)}",
        f"{S['rmse_all']['mean']:.4f}", f"{S['rmse_blind']['mean']:.4f}",
        f"{S['rmse_observed']['mean']:.4f}"), flush=True)
    print("%-18s %9s %9s %9s" % (
        "   +- std", f"±{S['rmse_all']['std']:.4f}", f"±{S['rmse_blind']['std']:.4f}",
        f"±{S['rmse_observed']['std']:.4f}"), flush=True)
    print("%-18s %9s %9s %9s" % (
        "   best member", f"{S['rmse_all']['min']:.4f}", f"{S['rmse_blind']['min']:.4f}",
        f"{S['rmse_observed']['min']:.4f}"), flush=True)
q = res["methods"]["varnet_nll_ens5"]
print("%-18s %9.4f %9.4f %9.4f   %s" % (
    "nll, ens of 5", q["rmse_all"], q["rmse_blind"], q["rmse_observed"],
    "  ".join(f"{q['per_channel'][c]:8.4f}" for c in CHAN)), flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=2)
print(f"\n[written] {OUT}")
