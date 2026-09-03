"""
diag_h_predicts_xhat.py  —  how much of x_hat is recoverable from h_last?

The claim being tested is architectural, and until now it has only been argued. The sigma^2
read-out is a 1x1 conv, so for each spatial cell (i,j) it sees the 64 numbers of
h_last[:, :, i, j] and must produce all C*T = 800 values of sigma^2 for that column. If those
64 numbers already determine x_hat for the column, then "sigma^2 cannot see x_hat" is empty --
it could reconstruct it internally. If they do not, the read-out is genuinely blind to it.

Measured as a LINEAR probe, because that is the read-out's own function class: least squares
from h_last (64 features, plus a bias) to x_hat (800 targets), fitted and scored on disjoint
windows, reported as R^2 per channel. A linear probe is the fairest test here -- anything a 1x1
conv can do, the probe can do.

x0 is included as a second probe. It is the observation-filled initial field, it enters x_hat
directly (x_hat = x0 - sum of updates) and the read-out never sees it, so it is the concrete
thing h_last would have to have retained.

    sbatch sbatch/submit_h_probe.sbatch
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
from methods.varnet.checks.model_io import load_solver                                     # noqa: E402

OUT = os.path.join(ROOT, "check_outputs", "eval", "h_predicts_xhat.json")
CHAN = list(config.get("grid", "channels"))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CK = os.environ.get("AUDIT_CK", "runs/varnet_vsb0_s0/varnet_best.pt")
NW = int(os.environ.get("AUDIT_WINDOWS", 16))

S, A, _ = load_solver(os.path.join(ROOT, CK), DEV)
DT, HID = A["dT"], A["lstm_hidden"]
K = A.get("obs_every_k") or config.get("observation", "obs_every_k")
print(f"[cfg] {CK}  dT={DT}  hidden={HID}  -> read-out maps {HID} to {len(CHAN)*DT}", flush=True)

X = np.asarray(om.load_state(om.split_files("test")[0])[0])
X = X[:(len(X) // DT) * DT]
valid = nav.build_valid_mask_from_config(X)
o = om.generate_observations(X, add_noise=True, valid_mask=valid, obs_every_k=K)
x0f = om.fill_missing_state(o["Y"], o["Omega_c"],
                            method=config.get("observation", "init_method"))
w = lambda a: torch.from_numpy(om.to_windows(a, DT)).float()
Yw, Mw, X0w = w(o["Y"]), w(o["Omega_c"].astype(np.float32)), w(x0f)
NW = min(NW, Yw.shape[0])

# hook the hidden state the read-out actually consumes
grab = {}
S.grad_net.lstm.register_forward_hook(lambda m, i, out: grab.__setitem__("h", out[0].detach()))

H, XH, X0 = [], [], []
for i in range(NW):
    yb, mb, x0b = (t[i:i+1].to(DEV) for t in (Yw, Mw, X0w))
    with torch.enable_grad():
        xr = S(x0b.clone(), yb, mb).detach()
    h = grab["h"]                                             # (1, hidden, H, W), last step
    B, C, T, Hh, Ww = xr.shape
    # per spatial cell: hidden features -> the C*T column the read-out has to produce
    H.append(h[0].reshape(HID, -1).T.cpu().numpy())            # (cells, hidden)
    XH.append(xr[0].reshape(C * T, -1).T.cpu().numpy())        # (cells, C*T)
    X0.append(x0b[0].reshape(C * T, -1).T.cpu().numpy())
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
H, XH, X0 = np.concatenate(H), np.concatenate(XH), np.concatenate(X0)
print(f"[data] {H.shape[0]:,} spatial cells x {H.shape[1]} features -> {XH.shape[1]} targets",
      flush=True)

cut = H.shape[0] // 2                                          # disjoint fit / score halves
A1 = np.concatenate([H[:cut], np.ones((cut, 1), np.float32)], 1)
A2 = np.concatenate([H[cut:], np.ones((H.shape[0] - cut, 1), np.float32)], 1)


def probe(Y, name):
    W, *_ = np.linalg.lstsq(A1, Y[:cut], rcond=None)
    pred = A2 @ W
    truth = Y[cut:]
    per = {}
    CT = truth.shape[1] // len(CHAN)
    for c, nm in enumerate(CHAN):
        t = truth[:, c * CT:(c + 1) * CT].ravel()
        p = pred[:, c * CT:(c + 1) * CT].ravel()
        ss_res = float(((t - p) ** 2).sum())
        ss_tot = float(((t - t.mean()) ** 2).sum())
        per[nm] = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    ss_res = float(((truth - pred) ** 2).sum())
    ss_tot = float(((truth - truth.mean()) ** 2).sum())
    per["_overall"] = 1.0 - ss_res / ss_tot
    print(f"[{name}] R^2 from h_last:  " +
          "   ".join(f"{k} {v:.3f}" for k, v in per.items()), flush=True)
    return per


res = {"checkpoint": CK, "hidden": HID, "targets": len(CHAN) * DT,
       "n_cells": int(H.shape[0]), "windows": NW,
       "x_hat": probe(XH, "x_hat"), "x0": probe(X0, "x0   ")}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=2)
print(f"\n[written] {OUT}")
