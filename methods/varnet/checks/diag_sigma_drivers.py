"""
diag_sigma_drivers.py  —  which of the head's inputs does sigma actually key off?

The distance-to-observation lookup explained at most 40% of sigma (checks/
diag_sigma_is_it_flow.py), which refutes "sigma is a distance map" but says nothing about what
the remaining variance IS. The head sees 12 pointwise inputs -- x_hat, Omega and
|x_hat - Phi(x_hat)|, four channels each -- so this fits the same kind of one-dimensional
lookup against every one of them and reports R^2. The winner is what sigma is really tracking.

Two readings matter. If sigma follows |x_hat| it is signal-dependent: bigger values, bigger
error, which is close to trivial. If it follows the prior residual it is tracking where the
reconstruction breaks the learnt dynamics, which is the flow-dependence Beauchamp et al. (AIES
2025, Sec. 2c) say a useful posterior needs.

Quantile bins rather than uniform, because the inputs are heavily skewed (62% of cells empty).

    sbatch sbatch/submit_sigma_drivers.sbatch
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

OUT = os.path.join(ROOT, "check_outputs", "eval",
                   os.environ.get("AUDIT_OUT", "sigma_drivers.json"))
CHAN = list(config.get("grid", "channels"))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NW = int(os.environ.get("AUDIT_WINDOWS", 24))
NBIN = 32
MEMBERS = [f"runs/{r}/varnet_best.pt" for r in
           os.environ.get("AUDIT_RUNS",
                          "varnet_ml5_s0,varnet_ml5_s1,varnet_ml5_s2,"
                          "varnet_ml5_s3,varnet_ml5_s4").split(",")]
print(f"[device] {DEV}   windows {NW}   bins {NBIN}", flush=True)

S0, A0, _ = load_solver(os.path.join(ROOT, MEMBERS[0]), DEV)
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
_XW, _YW, _MW, _X0W = (w(X_day), w(out["Y"]),
                       w(out["Omega_c"].astype(np.float32)), w(X0_day))
IDX = np.linspace(0, _XW.shape[0] - 1, min(NW, _XW.shape[0])).round().astype(int)
print(f"[data] {os.path.basename(_f)}  {len(IDX)} windows", flush=True)

XH, RES, MSK, SIG_H, SIG_S = [], [], [], [], []
for wi in IDX:
    Yb, Mb = _YW[wi:wi + 1].to(DEV), _MW[wi:wi + 1].to(DEV)
    X0b = _X0W[wi:wi + 1].to(DEV)
    mus, vars_ = [], []
    for mp in MEMBERS:
        Sm, _, ckm = load_solver(os.path.join(ROOT, mp), DEV)
        Sm.eval()

        with torch.enable_grad():
            xh, vh = Sm(X0b.clone(), Yb, Mb, return_var=True)
        xh, vh = xh.detach(), vh.detach()
        vars_.append(vh)
        mus.append(xh)
        if mp == MEMBERS[0]:
            with torch.no_grad():
                XH.append(xh[0].cpu().numpy())
                RES.append((xh - Sm.phi(xh)).abs()[0].cpu().numpy())
    mu, var = torch.stack(mus), torch.stack(vars_)
    SIG_H.append(var.mean(0).sqrt()[0].cpu().numpy())
    SIG_S.append((var.mean(0) + mu.var(0, unbiased=False)).sqrt()[0].cpu().numpy())
    MSK.append(Mb[0].cpu().numpy())
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
XH, RES, MSK = np.stack(XH), np.stack(RES), np.stack(MSK)
SIG_H, SIG_S = np.stack(SIG_H), np.stack(SIG_S)
print(f"[shapes] xhat {XH.shape}  sigma {SIG_H.shape}", flush=True)

from scipy.ndimage import distance_transform_edt                     # noqa: E402
DIST = np.empty(MSK.shape[:1] + MSK.shape[2:], np.float32)
for a in range(MSK.shape[0]):
    for t in range(MSK.shape[2]):
        m = MSK[a, 0, t] > 0.5
        DIST[a, t] = distance_transform_edt(~m) if m.any() else 0.0


def r2_lookup(feat, targ):
    """R^2 of a quantile-binned one-dimensional lookup, fit on half the windows."""
    nw = targ.shape[0]
    fit = np.zeros(nw, bool)
    fit[::2] = True
    ff, tf = feat[fit].ravel(), targ[fit].ravel()
    fe, te = feat[~fit].ravel(), targ[~fit].ravel()
    edges = np.unique(np.quantile(ff, np.linspace(0, 1, NBIN + 1)))
    if edges.size < 3:
        return 0.0
    bf = np.clip(np.digitize(ff, edges[1:-1]), 0, edges.size - 2)
    be = np.clip(np.digitize(fe, edges[1:-1]), 0, edges.size - 2)
    means = np.full(edges.size - 1, tf.mean(), np.float64)
    for b in range(edges.size - 1):
        sel = bf == b
        if sel.any():
            means[b] = tf[sel].mean()
    yhat = means[be]
    ss_res = float(((te - yhat) ** 2).sum())
    ss_tot = float(((te - te.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot


FEATS = {"distance_to_obs": np.repeat(DIST[:, None], len(CHAN), 1)}
for c, nm in enumerate(CHAN):
    FEATS[f"|xhat_{nm}|"] = np.repeat(np.abs(XH[:, c])[:, None], len(CHAN), 1)
    FEATS[f"resid_{nm}"] = np.repeat(RES[:, c][:, None], len(CHAN), 1)

# The residual is an INPUT to the head, so sigma correlating with it is circular -- it shows
# the head uses the feature, not that the feature carries error information. Scoring the same
# lookups against the TRUE error settles that: a feature that predicts |xhat - x| is worth
# keying off, one that does not is a feature the head is using for nothing.
ERR = np.abs(XH - np.stack([_XW[wi:wi + 1][0].numpy() for wi in IDX]))

RES_OUT = {}
for tag, SIG in (("head_alone", SIG_H), ("combined", SIG_S), ("TRUE_ERROR", ERR)):
    RES_OUT[tag] = {}
    for c, out_nm in enumerate(CHAN):
        row = {fn: r2_lookup(F[:, c], SIG[:, c]) for fn, F in FEATS.items()}
        RES_OUT[tag][out_nm] = row
    print(f"\n  {tag}:  R^2 of a 1-D lookup, per output channel")
    hdr = ["distance_to_obs"] + [f"|xhat_{n}|" for n in CHAN] + [f"resid_{n}" for n in CHAN]
    print("      " + "".join(f"{h[:11]:>13}" for h in ["sigma of"] + hdr))
    for out_nm in CHAN:
        r = RES_OUT[tag][out_nm]
        best = max(r, key=r.get)
        print("      " + f"{out_nm:>13}" + "".join(f"{r[h]:>13.3f}" for h in hdr)
              + f"   <- {best} {r[best]:.3f}")

# and the direct question: does the head's sigma predict the true error better than the
# best single hand-made feature does?
# The decisive comparison: calibrate the amplitude lookup as if it were a sigma, and score it
# the way we score the head. If a one-feature lookup is better calibrated than 548 trained
# parameters, the head's inputs are the problem, not its size or its training length.
def calibrate_and_score(feat, err, truth_mu_shape=None):
    """Fit sigma_hat = sqrt(E[err^2 | feature]) on half the windows, score on the other half."""
    nw = err.shape[0]
    fit = np.zeros(nw, bool)
    fit[::2] = True
    ff, ef = feat[fit].ravel(), err[fit].ravel()
    fe, ee = feat[~fit].ravel(), err[~fit].ravel()
    edges = np.unique(np.quantile(ff, np.linspace(0, 1, NBIN + 1)))
    bf = np.clip(np.digitize(ff, edges[1:-1]), 0, edges.size - 2)
    be = np.clip(np.digitize(fe, edges[1:-1]), 0, edges.size - 2)
    rms = np.full(edges.size - 1, float(np.sqrt((ef ** 2).mean())))
    for b in range(edges.size - 1):
        sel = bf == b
        if sel.any():
            rms[b] = float(np.sqrt((ef[sel] ** 2).mean()))
    sg = rms[be]
    rmse = float(np.sqrt((ee ** 2).mean()))
    return dict(sigma_mean=float(sg.mean()), rmse=rmse,
                spread_skill=float(sg.mean() / rmse),
                cov90=float((np.abs(ee) <= 1.6449 * sg).mean()))


def score_sigma(sig, err):
    nw = err.shape[0]
    keep = np.zeros(nw, bool)
    keep[1::2] = True                          # the same held-out half
    sg, ee = sig[keep].ravel(), err[keep].ravel()
    rmse = float(np.sqrt((ee ** 2).mean()))
    return dict(sigma_mean=float(sg.mean()), rmse=rmse,
                spread_skill=float(sg.mean() / rmse),
                cov90=float((np.abs(ee) <= 1.6449 * sg).mean()))


RES_OUT["calibration_head_vs_amplitude"] = {}
print("\n  calibration: the trained head against a lookup on |xhat| alone")
print(f"      {'channel':<9}{'':<4}{'sigma':>9}{'sp/sk':>8}{'90%':>8}")
for c, nm in enumerate(CHAN):
    a = score_sigma(SIG_H[:, c], ERR[:, c])
    b = score_sigma(SIG_S[:, c], ERR[:, c])
    d = calibrate_and_score(np.abs(XH[:, c]), ERR[:, c])
    RES_OUT["calibration_head_vs_amplitude"][nm] = dict(head=a, combined=b, amplitude=d)
    for lab, r in (("head", a), ("5-member", b), ("|xhat|", d)):
        print(f"      {nm:<9}{lab:<10}{r['sigma_mean']:>9.4f}{r['spread_skill']:>8.3f}"
              f"{r['cov90'] * 100:>7.1f}%")

RES_OUT["sigma_vs_error"] = {
    nm: dict(r2_head=r2_lookup(SIG_H[:, c], ERR[:, c]),
             r2_combined=r2_lookup(SIG_S[:, c], ERR[:, c]))
    for c, nm in enumerate(CHAN)}
print("\n  does sigma predict the TRUE error?  R^2 of a lookup on sigma")
for nm in CHAN:
    r = RES_OUT["sigma_vs_error"][nm]
    print(f"      {nm:<9s} head {r['r2_head']:+.3f}   combined {r['r2_combined']:+.3f}")

json.dump(dict(day=os.path.basename(_f), windows=len(IDX), bins=NBIN, results=RES_OUT),
          open(OUT, "w"), indent=1)
print(f"\n[json] {OUT}")
