"""
diag_sparsification_enkf.py  —  the same usefulness test on the EnKF, for a fair contrast.

checks/diag_sparsification.py found that our learnt sigma^2 carries almost no WITHIN-CHANNEL
ranking information: Spearman(sigma, |err|) of 0.036 / 0.035 / 0.002 / 0.040 and AUSE 0.93-1.00,
i.e. no better than a random ordering. The obvious next question is whether the EnKF's ensemble
spread does any better.

Why this contrast is fair where the calibration one was not. The EnKF's variance is collapsed by
~90x in SCALE (spread/skill 0.011), which destroys coverage and makes its NLL diverge, and we
are not allowed to retune its inflation. But AUSE and Spearman are invariant to any monotone
rescaling, so the collapse cannot penalise it here -- inflation changes the scale, not the order.
Each method is also judged against ITS OWN error, which is the right question for each: "can your
uncertainty rank your mistakes".

Uses the mask the EnKF actually saw (obs_*.npz Omega, 46.9% observed), not a regenerated one.
Spread may be a standard deviation or a variance; it makes no difference to a rank statistic.

Reported on the first 2,400 frames -- the same 12 windows of dT=200 our own measurement used --
and on the whole day, because the early frames of an ATC day are nearly empty and a subset can
flatter or punish either method.

    sbatch sbatch/submit_sparsification_enkf.sbatch
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from crowdcore import config                                                        # noqa: E402
from crowdcore import paths

SRC = paths.enkf_export("enkf_k1_full")
DAY = os.environ.get("AUDIT_DAY", "atc-20130811")
OUT = os.path.join(paths.eval_out(paths.VARNET), "sparsification_enkf.json")
CHAN = list(config.get("grid", "channels"))
FRACS = np.linspace(0.0, 0.9, 19)

o = np.load(os.path.join(SRC, f"obs_{DAY}.npz"))
e = np.load(os.path.join(SRC, f"est_{DAY}.npz"))
Xt, Om, Est, Spr = o["X_true"], o["Omega"], e["Est"], e["Spread"]
print(f"[{DAY}] frames {Xt.shape[0]}  observed {Om.mean()*100:.1f}%", flush=True)
assert Est.shape == Xt.shape and Spr.shape == Xt.shape, (Est.shape, Spr.shape, Xt.shape)


def curves(sig, err, rng):
    n = err.size
    out = {}
    for nm, order in (("sigma", np.argsort(-sig)), ("oracle", np.argsort(-err)),
                      ("random", rng.permutation(n))):
        c = np.concatenate([[0.0], np.cumsum(err[order] ** 2)])
        tot = c[-1]
        out[nm] = [float(np.sqrt((tot - c[int(f * n)]) / max(n - int(f * n), 1))) for f in FRACS]
    a = np.trapz(np.array(out["sigma"]) - np.array(out["oracle"]), FRACS)
    b = np.trapz(np.array(out["random"]) - np.array(out["oracle"]), FRACS)
    out["ause"] = float(a / b) if b > 0 else None
    out["spearman_sigma_vs_err"] = float(
        np.corrcoef(np.argsort(np.argsort(sig)), np.argsort(np.argsort(err)))[0, 1])
    return out


rng = np.random.default_rng(0)
res = {"day": DAY, "source": os.path.relpath(SRC, ROOT), "fracs": [float(f) for f in FRACS],
       "ranges": {}}

for label, sl in (("first_2400_frames", slice(0, 2400)), ("whole_day", slice(None))):
    xt, est, spr, om = Xt[sl], Est[sl], Spr[sl], Om[sl]
    err = np.abs(est - xt)
    blind = ~np.repeat(om[:, None], len(CHAN), axis=1)          # Omega is (T,H,W)
    empty = np.repeat(xt[:, 0:1] == 0, len(CHAN), axis=1)
    R = {"n_frames": int(xt.shape[0]), "splits": {}, "per_channel_blind": {}}
    print(f"\n########## {label}  ({xt.shape[0]} frames) ##########", flush=True)
    for nm, sel in (("blind", blind), ("blind_empty", blind & empty),
                    ("blind_occupied", blind & ~empty)):
        q = curves(spr[sel], err[sel], rng)
        R["splits"][nm] = q
        print(f"  {nm:16s} ({sel.sum():>12,} cells)  AUSE {q['ause']:.3f}   "
              f"Spearman {q['spearman_sigma_vs_err']:.3f}", flush=True)
    for c, name in enumerate(CHAN):
        sel = blind[:, c]
        q = curves(spr[:, c][sel], err[:, c][sel], rng)
        R["per_channel_blind"][name] = {"ause": q["ause"],
                                        "spearman": q["spearman_sigma_vs_err"]}
        print(f"    [{name:8s}] AUSE {q['ause']:.3f}   "
              f"Spearman {q['spearman_sigma_vs_err']:.3f}", flush=True)
    res["ranges"][label] = R

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=2)
print(f"\n[written] {OUT}")
