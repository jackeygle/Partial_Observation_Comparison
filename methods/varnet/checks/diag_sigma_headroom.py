"""
diag_sigma_headroom.py  —  is the learnt sigma^2 leaving information on the table?

The measured problem (checks/diag_beta_gradient_share.py): sigma^2 is spatially flat. It
varies 0.67-0.97x between empty and occupied blind cells while the TRUE error varies 17-23x.
A flat sigma^2 makes 1/sigma^2 a per-channel constant, which degenerates Eq.1 into the
per-channel weighted MSE that eight earlier loss experiments established degrades optimisation.

The suspected cause is structural. Lakshminarayanan et al. Sec. 2.2.1 puts mu and sigma^2 in
the SAME final layer, so sigma^2 sees whatever determines mu. Ours does not: mu = x_hat is a
20-step accumulator (x = x - upd), while sigma^2 = softplus(out_var(h_last)) reads only the
LSTM hidden state. sigma^2 therefore has no access to x_hat -- and |x_hat| is exactly the
empty-vs-occupied signal it is blind to.

This decides whether that is worth a retrain, WITHOUT retraining. Fit the simplest possible
competitor -- a 1-D quantile lookup sigma_hat^2 = E[err^2 | |x_hat|] -- and compare it with the
model's own sigma^2 on the same points.

FIT AND SCORE ARE DIFFERENT DAYS. An earlier version of this lookup was fitted and scored on
one day, which flatters it; the numbers from that run must not be reused. Here the bins and
their values come from a TRAIN-split day and are applied unchanged to a TEST day.

Reading it:
  spread_skill  = mean(sigma^2) / mean(err^2), 1.0 is calibrated in aggregate
  occ_ratio     = sigma^2 occupied / sigma^2 empty, against err^2's own 17-23x
  nll           = Eq.1 without the constant, lower is better
If the lookup beats the learnt sigma^2 on occ_ratio and nll, the information is available and
the read-out is not using it -- fix the read-out. If it does not, a diagonal sigma^2 cannot
resolve this and that is a limitation to state, not a bug to chase.

    sbatch sbatch/submit_sigma_headroom.sbatch
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

OUT = os.path.join(ROOT, "check_outputs", "eval", "sigma_headroom.json")
CHAN = list(config.get("grid", "channels"))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NW = int(os.environ.get("AUDIT_WINDOWS", 12))
NBIN = 24
CK = os.environ.get("AUDIT_CK", "runs/varnet_vrb0_s0/varnet_best.pt")
EPS = 1e-6

S, A, _ = load_solver(os.path.join(ROOT, CK), DEV)
DT = A["dT"]
print(f"[device] {DEV}  ck {CK}  dT {DT}  windows/day {NW}", flush=True)


def harvest(path):
    """Run the solver over one day and return (|x_hat|, err^2, sigma^2, empty) per channel."""
    X_day, _ = om.load_state(path)
    valid = nav.build_valid_mask_from_config(X_day)
    X_day = X_day[:(X_day.shape[0] // DT) * DT]
    o = om.generate_observations(X_day, A["sensing_range"], A["num_agents"], add_noise=True,
                                seed=A.get("data_seed", 0) or 0, valid_mask=valid,
                                obs_every_k=1)
    X0 = om.fill_missing_state(o["Y"], o["Omega_c"],
                               method=config.get("observation", "init_method"))
    w = lambda a: torch.from_numpy(om.to_windows(a, DT)).float()
    XW, YW, MW, X0W = w(X_day), w(o["Y"]), w(o["Omega_c"].astype(np.float32)), w(X0)
    n = min(NW, XW.shape[0])
    ax, ae, av, az = [], [], [], []
    for i in range(n):
        xb, yb, mb, x0b = (t[i:i+1].to(DEV) for t in (XW, YW, MW, X0W))
        with torch.enable_grad():
            xr, vr = S(x0b.clone(), yb, mb, return_var=True)
        B = (mb == 0)[0].reshape(len(CHAN), -1).cpu().numpy()        # blind cells only
        f = lambda t: t.detach()[0].reshape(len(CHAN), -1).cpu().numpy()
        xh, er, vv = f(xr.abs()), f((xr - xb) ** 2), f(vr)
        zz = np.repeat((xb[:, 0:1] == 0)[0].reshape(1, -1).cpu().numpy(), len(CHAN), axis=0)
        ax.append(np.where(B, xh, np.nan)); ae.append(np.where(B, er, np.nan))
        av.append(np.where(B, vv, np.nan)); az.append(np.where(B, zz, np.nan))
    cat = lambda L: np.concatenate(L, axis=1)
    return cat(ax), cat(ae), cat(av), cat(az)


fit_day = om.split_files("train")[0]
score_day = om.split_files("test")[0]
print(f"[fit ] {os.path.basename(fit_day)}\n[score] {os.path.basename(score_day)}", flush=True)
Xf, Ef, _, _ = harvest(fit_day)
Xs, Es, Vs, Zs = harvest(score_day)

res = {"fit_day": os.path.basename(fit_day), "score_day": os.path.basename(score_day),
       "checkpoint": CK, "n_bins": NBIN, "per_channel": {}}

for c, name in enumerate(CHAN):
    kf = ~np.isnan(Xf[c]); ks = ~np.isnan(Xs[c])
    xf, ef = Xf[c][kf], Ef[c][kf]
    xs, es, vs, zs = Xs[c][ks], Es[c][ks], Vs[c][ks], Zs[c][ks].astype(bool)

    # lookup built on the FIT day only: quantile bins of |x_hat| -> mean err^2 in that bin
    edges = np.unique(np.quantile(xf, np.linspace(0, 1, NBIN + 1)))
    idx = np.clip(np.digitize(xf, edges[1:-1]), 0, len(edges) - 2)
    table = np.array([ef[idx == b].mean() if (idx == b).any() else ef.mean()
                      for b in range(len(edges) - 1)])
    # applied unchanged to the SCORE day
    vhat = np.maximum(table[np.clip(np.digitize(xs, edges[1:-1]), 0, len(edges) - 2)], EPS)

    nll = lambda v: float(np.mean(np.log(v) / 2 + es / (2 * v)))
    ratio = lambda v: float(v[~zs].mean() / v[zs].mean()) if zs.any() and (~zs).any() else None
    res["per_channel"][name] = {
        "n_scored": int(ks.sum()),
        "err2_occ_over_empty": ratio(es),
        "learnt":  {"spread_skill": float(vs.mean() / es.mean()),
                    "occ_over_empty": ratio(vs), "nll": nll(vs)},
        "lookup":  {"spread_skill": float(vhat.mean() / es.mean()),
                    "occ_over_empty": ratio(vhat), "nll": nll(vhat)},
    }
    r = res["per_channel"][name]
    print(f"[{name:8s}] err2 occ/empty {r['err2_occ_over_empty']:6.1f}x | "
          f"learnt occ/empty {r['learnt']['occ_over_empty']:5.2f} nll {r['learnt']['nll']:8.4f}"
          f" | lookup occ/empty {r['lookup']['occ_over_empty']:6.2f} "
          f"nll {r['lookup']['nll']:8.4f}", flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=2)
print(f"\n[written] {OUT}")
