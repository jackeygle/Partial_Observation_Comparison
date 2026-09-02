"""
diag_sparsification.py  —  is the learnt sigma^2 USEFUL, as opposed to calibrated?

Everything measured so far has been calibration: spread/skill, coverage, the Sec. A.2 curve.
Calibration and usefulness are independent. A sigma^2 whose overall scale is wrong but whose
ORDERING is right is useful -- one global rescale fixes it. A sigma^2 whose average magnitude is
perfect but whose ordering is random is worthless. Nothing here has tested ordering.

The sparsification (error-retention) curve tests exactly that. Sort cells by sigma descending,
drop the most-uncertain fraction, and record the RMSE of what remains. Three curves:

  sigma    the model's own ordering
  oracle   sorted by the TRUE |error| -- the best any ordering could do
  random   no information at all

Read the gap. Close to oracle means the ranking carries the error structure; close to random
means it does not, whatever its calibration says. The single number is AUSE (Area Under the
Sparsification Error), the normalised area between the sigma curve and the oracle: 0 is perfect,
1 is no better than random.

Reported for blind cells, and separately for the empty and occupied halves, because 87% of blind
cells are empty and an aggregate is dominated by them.

    sbatch sbatch/submit_sparsification.sbatch
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

OUT = os.path.join(ROOT, "check_outputs", "eval", "sparsification.json")
CHAN = list(config.get("grid", "channels"))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FMT = os.environ.get("AUDIT_FMT", "runs/varnet_vsb0_s{}")
NW = int(os.environ.get("AUDIT_WINDOWS", 12))
FRACS = np.linspace(0.0, 0.9, 19)          # fraction of most-uncertain cells removed

solvers, A = [], None
for s in range(5):
    sol, A, _ = load_solver(os.path.join(ROOT, FMT.format(s), "varnet_best.pt"), DEV)
    solvers.append(sol)
DT = A["dT"]
K = A.get("obs_every_k") or config.get("observation", "obs_every_k")
print(f"[members] 5  dT={DT}  windows={NW}", flush=True)

X = np.asarray(om.load_state(om.split_files("test")[0])[0])
X = X[:(len(X) // DT) * DT]
valid = nav.build_valid_mask_from_config(X)
o = om.generate_observations(X, add_noise=True, valid_mask=valid, obs_every_k=K)
x0 = om.fill_missing_state(o["Y"], o["Omega_c"],
                           method=config.get("observation", "init_method"))
w = lambda a: torch.from_numpy(om.to_windows(a, DT)).float()
Yw, Mw, X0w, Xw = w(o["Y"]), w(o["Omega_c"].astype(np.float32)), w(x0), w(np.asarray(X))
NW = min(NW, Xw.shape[0])

SIG, ERR, BL, EM, CH = [], [], [], [], []
for i in range(NW):
    yb, mb, x0b = (t[i:i+1].to(DEV) for t in (Yw, Mw, X0w))
    xb = Xw[i:i+1].to(DEV)
    mus, vs = [], []
    for sol in solvers:
        with torch.enable_grad():
            xr, vr = sol(x0b.clone(), yb, mb, return_var=True)
        mus.append(xr.detach()); vs.append(vr.detach())
    mu, var = torch.stack(mus), torch.stack(vs)
    tot = var.mean(0) + mu.var(0, unbiased=False)
    mu_s = mu.mean(0)
    g = lambda t: t.reshape(len(CHAN), -1).cpu().numpy()
    SIG.append(g(tot.sqrt())); ERR.append(g((mu_s - xb).abs()))
    BL.append(g(mb) == 0)
    EM.append(np.repeat((xb[:, 0:1] == 0)[0].reshape(1, -1).cpu().numpy(), len(CHAN), axis=0))
    del mu, var, mus, vs
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
cat = lambda L: np.concatenate(L, axis=1)
SIG, ERR, BL, EM = cat(SIG), cat(ERR), cat(BL), cat(EM).astype(bool)


def curves(sig, err, rng):
    """RMSE of the retained cells after dropping the top f by each ordering."""
    n = err.size
    o_sig = np.argsort(-sig)                      # most uncertain first
    o_orc = np.argsort(-err)                      # truly worst first
    o_rnd = rng.permutation(n)
    out = {}
    for nm, order in (("sigma", o_sig), ("oracle", o_orc), ("random", o_rnd)):
        e = err[order] ** 2
        # RMSE of the tail that survives after dropping the first k
        csum = np.concatenate([[0.0], np.cumsum(e)])
        tot = csum[-1]
        out[nm] = [float(np.sqrt((tot - csum[int(f * n)]) / max(n - int(f * n), 1)))
                   for f in FRACS]
    # AUSE: area between sigma and oracle, normalised by area between random and oracle
    a = np.trapz(np.array(out["sigma"]) - np.array(out["oracle"]), FRACS)
    b = np.trapz(np.array(out["random"]) - np.array(out["oracle"]), FRACS)
    out["ause"] = float(a / b) if b > 0 else None
    out["spearman_sigma_vs_err"] = float(
        np.corrcoef(np.argsort(np.argsort(sig)), np.argsort(np.argsort(err)))[0, 1])
    return out


rng = np.random.default_rng(0)
res = {"fracs": [float(f) for f in FRACS], "windows": NW, "run_fmt": FMT, "splits": {}}
SEL = {"blind": BL, "blind_empty": BL & EM, "blind_occupied": BL & ~EM}
for nm, sel in SEL.items():
    s, e = SIG[sel], ERR[sel]
    if s.size < 1000:
        continue
    res["splits"][nm] = curves(s, e, rng)
    q = res["splits"][nm]
    print(f"\n=== {nm}  ({s.size:,} cells) ===", flush=True)
    print(f"  AUSE = {q['ause']:.3f}   (0 = matches oracle, 1 = no better than random)", flush=True)
    print(f"  Spearman(sigma, |err|) = {q['spearman_sigma_vs_err']:.3f}", flush=True)
    print("  %-8s %9s %9s %9s" % ("dropped", "sigma", "oracle", "random"), flush=True)
    for j in (0, 4, 9, 14, 18):
        print("  %6.0f%% %9.4f %9.4f %9.4f" % (FRACS[j]*100, q["sigma"][j], q["oracle"][j],
                                               q["random"][j]), flush=True)

# per channel on blind cells, since vx behaves differently from density
res["per_channel_blind"] = {}
for c, name in enumerate(CHAN):
    sel = BL[c]
    q = curves(SIG[c][sel], ERR[c][sel], rng)
    res["per_channel_blind"][name] = {"ause": q["ause"],
                                      "spearman": q["spearman_sigma_vs_err"]}
    print(f"[{name:8s}] AUSE {q['ause']:.3f}   Spearman {q['spearman_sigma_vs_err']:.3f}",
          flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=2)
print(f"\n[written] {OUT}")
