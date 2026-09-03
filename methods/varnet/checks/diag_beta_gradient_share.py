"""
diag_beta_gradient_share.py  —  why does beta=0 lose 42% blind-zone MSE to beta=1?

Both runs are the same architecture, same seed, same 150 epochs, same var_eps. The ONLY
difference is the factor sg(sigma^2)^beta on each point's NLL, and that factor changes exactly
one thing -- the weight each point's squared error carries in the mean's gradient:

    beta=0   dL/dmu = -(X - mu) / sigma^2      per-point weight  1/sigma^2
    beta=1   dL/dmu = -(X - mu) / 2            per-point weight  uniform  (= MSE)

So any accuracy difference has to come from the SPREAD of 1/sigma^2. This script measures that
spread and, more importantly, tests the consequence it predicts.

Hypothesis. 62% of these cells are empty, and an empty cell's density and velocity are
identically zero -- perfectly predictable. sigma^2 collapses toward var_eps there, 1/sigma^2
blows up, and the empty cells take over the mean's gradient. The occupied cells, which is
where every bit of vx signal lives, are starved.

The prediction that would confirm it, and which MSE alone cannot distinguish: beta=0's
degradation must be CONCENTRATED IN OCCUPIED CELLS. If beta=0 matches (or beats) beta=1 on
empty cells while losing badly on occupied ones, the mechanism is the weighting. If beta=0
loses uniformly across both, the weighting is not the story and something else is wrong.

Reported per channel, restricted to blind cells (Omega == 0) since that is the metric in
question, and split by whether the TRUE density in that cell is zero.

    sbatch sbatch/submit_beta_share.sbatch
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

OUT = os.path.join(ROOT, "check_outputs", "eval", "beta_gradient_share.json")
CHAN = list(config.get("grid", "channels"))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NW = int(os.environ.get("AUDIT_WINDOWS", 24))
# AUDIT_RUNS is "label=path;label=path". SEMICOLONS, not commas: `sbatch --export` splits its
# argument on commas, so a comma-separated value silently loses everything after the first entry
# and this script then reports one model as if that were the whole comparison. Both separators
# are accepted so an interactive comma does not fail, but scripts should use ';'.
RUNS = dict(kv.split("=", 1) for kv in
            os.environ.get("AUDIT_RUNS",
                           "beta0=runs/varnet_vrb0_s0/varnet_best.pt;"
                           "beta1=runs/varnet_vrb1_s0/varnet_best.pt")
            .replace(",", ";").split(";") if kv)
print(f"[runs] {len(RUNS)}: {list(RUNS)}", flush=True)
print(f"[device] {DEV}   windows {NW}", flush=True)

S0, A0, _ = load_solver(os.path.join(ROOT, next(iter(RUNS.values()))), DEV)
DT = A0["dT"]
_f = om.split_files("test")[0]
X_day, _ = om.load_state(_f)
valid = nav.build_valid_mask_from_config(X_day)
X_day = X_day[:(X_day.shape[0] // DT) * DT]
out = om.generate_observations(X_day, A0["sensing_range"], A0["num_agents"], add_noise=True,
                              seed=A0.get("data_seed", 0) or 0, valid_mask=valid,
                              obs_every_k=1)
X0_day = om.fill_missing_state(out["Y"], out["Omega_c"],
                              method=config.get("observation", "init_method"))
w = lambda a: torch.from_numpy(om.to_windows(a, DT)).float()
XW, YW, MW, X0W = (w(X_day), w(out["Y"]),
                   w(out["Omega_c"].astype(np.float32)), w(X0_day))
NW = min(NW, XW.shape[0])
print(f"[data] {NW} windows of dT={DT}", flush=True)

res = {}
for tag, ck in RUNS.items():
    p = os.path.join(ROOT, ck)
    if not os.path.exists(p):
        print(f"[skip] {tag}: no checkpoint yet at {ck}", flush=True)
        continue
    S, A, _ = load_solver(p, DEV)
    sig2, err2, blind, empty = [], [], [], []
    for i in range(NW):
        xb, yb, mb, x0b = (t[i:i+1].to(DEV) for t in (XW, YW, MW, X0W))
        with torch.enable_grad():
            xr, vr = S(x0b.clone(), yb, mb, return_var=True)
        sig2.append(vr.detach().cpu().numpy()[0])
        err2.append(((xr.detach() - xb) ** 2).cpu().numpy()[0])
        blind.append((mb == 0).cpu().numpy()[0])
        # "empty" = the TRUE density in that cell is zero; broadcast across channels
        empty.append(np.repeat((xb[:, 0:1] == 0).cpu().numpy()[0], len(CHAN), axis=0))
    sig2 = np.concatenate([a.reshape(len(CHAN), -1) for a in sig2], axis=1)
    err2 = np.concatenate([a.reshape(len(CHAN), -1) for a in err2], axis=1)
    blind = np.concatenate([a.reshape(len(CHAN), -1) for a in blind], axis=1)
    empty = np.concatenate([a.reshape(len(CHAN), -1) for a in empty], axis=1)

    per = {}
    for c, name in enumerate(CHAN):
        B = blind[c]                                   # blind cells only
        s, e, z = sig2[c][B], err2[c][B], empty[c][B]
        wgt = 1.0 / s                                  # the beta=0 gradient weight
        per[name] = {
            "n_blind": int(B.sum()),
            "frac_empty": float(z.mean()),
            "sigma2_median_empty": float(np.median(s[z])) if z.any() else None,
            "sigma2_median_occupied": float(np.median(s[~z])) if (~z).any() else None,
            "sigma2_ratio_occ_over_empty": (float(np.median(s[~z]) / np.median(s[z]))
                                            if z.any() and (~z).any() else None),
            # share of the mean's total gradient weight that the empty cells take
            "grad_weight_share_empty": float(wgt[z].sum() / wgt.sum()) if z.any() else None,
            "mse_empty": float(e[z].mean()) if z.any() else None,
            "mse_occupied": float(e[~z].mean()) if (~z).any() else None,
        }
    # The claim this tests: dividing by sigma^2 acts as an implicit PER-CHANNEL normaliser.
    # vx has ~11x density's error but a comparable sigma^2, so 1/sigma^2 compresses the
    # cross-channel gradient ratio and vx loses the priority MSE gives it. Measured as each
    # channel's share of sum|dL/dmu| over blind cells, which is what actually drives training.
    g = np.abs(np.sqrt(err2)) / sig2                      # beta=0 weight: |err| / sigma^2
    g1 = np.abs(np.sqrt(err2))                            # beta=1 weight: |err|
    for c, name in enumerate(CHAN):
        B = blind[c]
        per[name]["grad_share_if_beta0"] = float(g[c][B].sum() / g[:, :][blind].sum()) \
            if blind.any() else None
    tot0 = sum(g[c][blind[c]].sum() for c in range(len(CHAN)))
    tot1 = sum(g1[c][blind[c]].sum() for c in range(len(CHAN)))
    for c, name in enumerate(CHAN):
        per[name]["grad_share_if_beta0"] = float(g[c][blind[c]].sum() / tot0)
        per[name]["grad_share_if_beta1"] = float(g1[c][blind[c]].sum() / tot1)

    res[tag] = per
    print(f"[{tag}] done", flush=True)

if len(res) == 2:
    _k0, _k1 = list(res.keys())
    cmp = {}
    for name in CHAN:
        a, b = res[_k0][name], res[_k1][name]
        f = lambda k: (a[k] / b[k] - 1.0) * 100 if (a[k] and b[k]) else None
        cmp[name] = {"empty_pct": f("mse_empty"), "occupied_pct": f("mse_occupied")}
    res[f"{_k0}_vs_{_k1}_mse_pct"] = cmp

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=2)
print(json.dumps(res, indent=2))
print(f"\n[written] {OUT}")
