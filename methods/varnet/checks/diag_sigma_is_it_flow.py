"""
diag_sigma_is_it_flow.py  —  is our sigma flow-dependent, or just a distance-to-observation map?

Beauchamp et al. (AIES 2025, Sec. 2c) name the failure mode we may have walked into: a posterior
covariance "mainly driven by the sampling of the observation, which may not be realistic for
dynamical systems", and they note that ensemble schemes exist precisely to counteract "the
sampling issue used in OI or in most of the variational schemes".

Our head reads the mask directly and puts most of its first-layer weight mass there, so the
concern is concrete: sigma may be an expensive way to draw a map of how far each cell is from
the nearest observation, carrying no information about the flow.

Test 1 — falsify the head. Fit a mask-only predictor of sigma: for each channel, the mean sigma
as a function of the (rounded) Euclidean distance to the nearest observed cell in that frame.
Fit on half the windows, score on the other half. If that lookup explains most of the variance
of our sigma, the head is not adding flow information and we should say so.

Test 2 — is the inverse-Hessian route even available. The classical identity A = (grad^2 J)^-1
holds AT a minimum of J. Our solver is trained on a reconstruction loss, not to minimise J, so
measure ||grad J|| at x0 and at xhat: if it has not collapsed, the identity does not apply and
that route is closed for us.

    sbatch sbatch/submit_sigma_audit.sbatch
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

OUT = os.path.join(ROOT, "check_outputs", "eval", "sigma_audit.json")
CHAN = list(config.get("grid", "channels"))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NW = int(os.environ.get("AUDIT_WINDOWS", 40))
MEMBERS = [f"runs/varnet_ml5_s{s}/varnet_best.pt" for s in range(5)]
print(f"[device] {DEV}", flush=True)

# ── one test day, NW windows strided across it ──────────────────────────────────────────
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
print(f"[data] {os.path.basename(_f)}  {len(IDX)} of {_XW.shape[0]} windows", flush=True)

# ── every member's sigma^2, and the combined one ────────────────────────────────────────
SIG_HEAD, SIG_STAR, MASKS = [], [], []
for wi in IDX:
    Xb, Yb = _XW[wi:wi + 1].to(DEV), _YW[wi:wi + 1].to(DEV)
    Mb, X0b = _MW[wi:wi + 1].to(DEV), _X0W[wi:wi + 1].to(DEV)
    mus, vars_ = [], []
    for mp in MEMBERS:
        Sm, _, ckm = load_solver(os.path.join(ROOT, mp), DEV)
        Sm.eval()

        with torch.enable_grad():
            xh, vh = Sm(X0b.clone(), Yb, Mb, return_var=True)
        vars_.append(vh.detach())
        mus.append(xh.detach())
    mu, var = torch.stack(mus), torch.stack(vars_)
    SIG_HEAD.append(var.mean(0).sqrt()[0].cpu().numpy())            # the head alone
    SIG_STAR.append((var.mean(0) + mu.var(0, unbiased=False)).sqrt()[0].cpu().numpy())
    MASKS.append(Mb[0, 0].cpu().numpy() > 0.5)                      # (T,H,W), shared per cell
    if dev_free := (DEV.type == "cuda"):
        torch.cuda.empty_cache()
SIG_HEAD = np.stack(SIG_HEAD)          # (NW, C, T, H, W)
SIG_STAR = np.stack(SIG_STAR)
MASKS = np.stack(MASKS)                # (NW, T, H, W)
print(f"[sigma] head {SIG_HEAD.shape}  combined {SIG_STAR.shape}", flush=True)

# ── distance to the nearest observed cell, per frame ────────────────────────────────────
from scipy.ndimage import distance_transform_edt                     # noqa: E402
DIST = np.empty(MASKS.shape, np.float32)
for a in range(MASKS.shape[0]):
    for t in range(MASKS.shape[1]):
        m = MASKS[a, t]
        # a frame with no observation at all has an undefined distance; mark it and skip below
        DIST[a, t] = distance_transform_edt(~m) if m.any() else -1.0
DBIN = np.rint(DIST).astype(np.int32)
print(f"[dist] frames with no observation: {(DBIN[:, :, 0, 0] < 0).sum()} of "
      f"{DBIN.shape[0] * DBIN.shape[1]}", flush=True)


def r2_maskonly(sig):
    """R^2 of a (channel, rounded distance) lookup fitted on half the windows."""
    res = {}
    nw = sig.shape[0]
    fit, ev = np.arange(0, nw, 2), np.arange(1, nw, 2)
    for c, nm in enumerate(CHAN):
        yf, df = sig[fit, c].ravel(), np.repeat(DBIN[fit][:, None], 1, 1)[:, 0].ravel()
        ye, de = sig[ev, c].ravel(), np.repeat(DBIN[ev][:, None], 1, 1)[:, 0].ravel()
        ok_f, ok_e = df >= 0, de >= 0
        yf, df, ye, de = yf[ok_f], df[ok_f], ye[ok_e], de[ok_e]
        # the lookup, plus the channel mean as the fallback for unseen distances
        table = {int(d): float(yf[df == d].mean()) for d in np.unique(df)}
        gmean = float(yf.mean())
        pred = np.array([table.get(int(d), gmean) for d in np.unique(de)])
        lut = dict(zip(np.unique(de).tolist(), pred.tolist()))
        yhat = np.array([lut[int(d)] for d in de], np.float32)
        ss_res = float(((ye - yhat) ** 2).sum())
        ss_tot = float(((ye - ye.mean()) ** 2).sum())
        res[nm] = dict(r2_distance=1.0 - ss_res / ss_tot,
                       r2_channel_mean=0.0,          # by construction
                       sigma_mean=float(ye.mean()), sigma_std=float(ye.std()),
                       n=int(ye.size))
    return res


AUD = dict(day=os.path.basename(_f), windows=len(IDX), n_members=len(MEMBERS),
           head_alone=r2_maskonly(SIG_HEAD), combined=r2_maskonly(SIG_STAR))
for tag in ("head_alone", "combined"):
    print(f"\n  {tag}:  R^2 of a (channel, distance-to-observation) lookup")
    for nm in CHAN:
        d = AUD[tag][nm]
        print(f"     {nm:<9s} R2 {d['r2_distance']:+.4f}   sigma {d['sigma_mean']:.4f} "
              f"+- {d['sigma_std']:.4f}")

# ── test 2: is x_hat anywhere near a stationary point of J? ─────────────────────────────
def grad_norm(x, y, m, sol):
    xv = x.clone().requires_grad_(True)
    J = sol.var_cost(xv - sol.phi(xv), (xv - y) * m)
    g = torch.autograd.grad(J, xv)[0]
    return float(J), float(g.norm()), float(torch.sqrt((g ** 2).mean()))


S0.eval()
G = []
for wi in IDX[:8]:
    Xb, Yb = _XW[wi:wi + 1].to(DEV), _YW[wi:wi + 1].to(DEV)
    Mb, X0b = _MW[wi:wi + 1].to(DEV), _X0W[wi:wi + 1].to(DEV)
    with torch.enable_grad():
        xh = S0(X0b.clone(), Yb, Mb).detach()
    j0, n0, r0 = grad_norm(X0b, Yb, Mb, S0)
    j1, n1, r1 = grad_norm(xh, Yb, Mb, S0)
    G.append(dict(J0=j0, J1=j1, gnorm0=n0, gnorm1=n1, grms0=r0, grms1=r1, ratio=n1 / n0))
AUD["stationarity"] = dict(
    windows=len(G),
    J_start=float(np.mean([g["J0"] for g in G])),
    J_end=float(np.mean([g["J1"] for g in G])),
    gnorm_start=float(np.mean([g["gnorm0"] for g in G])),
    gnorm_end=float(np.mean([g["gnorm1"] for g in G])),
    ratio=float(np.mean([g["ratio"] for g in G])))
_s = AUD["stationarity"]
print(f"\n  stationarity over {_s['windows']} windows:")
print(f"     J        {_s['J_start']:.4f} -> {_s['J_end']:.4f}")
print(f"     ||grad J|| {_s['gnorm_start']:.4e} -> {_s['gnorm_end']:.4e}   "
      f"ratio {_s['ratio']:.3f}")
print(f"     the inverse-Hessian identity needs this ratio near 0")

json.dump(AUD, open(OUT, "w"), indent=1)
print(f"\n[json] {OUT}")
