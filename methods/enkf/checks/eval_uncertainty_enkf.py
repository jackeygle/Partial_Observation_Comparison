"""
eval_uncertainty_enkf.py  —  score the EnKF's ensemble spread as a predictive uncertainty

The counterpart of checks/eval_uncertainty.py, which does the same for our deep ensemble. Both
write the same fields so the two jsons can be put side by side without reformatting.

The EnKF's uncertainty estimate IS its ensemble spread: est_*.npz stores `Est` (the ensemble
mean, its point estimate) and `Spread` (the per-cell ensemble standard deviation), so nothing
has to be re-run -- this only reads what the full-day runs already produced.

Metrics, and why these:
    CRPS          the headline. Proper scoring rule, in the units of the state, and -- unlike
                  NLL -- bounded when a method is grossly overconfident. That matters here:
                  the EnKF's sigma is ~1% of its own error, which sends the NLL's
                  (x-mu)^2/(2 sigma^2) term to ~1e18, a number that cannot go in a table.
    NLL           reported anyway, precisely so that blow-up is on the record.
    spread-skill  mean sigma / RMSE. 1.0 is calibrated, below 1 is overconfident.
    reliability   Lakshminarayanan et al. App. A.2: the fraction of truths inside the nominal
                  z% Gaussian interval, for z = 10..90.

Splits mirror eval_uncertainty.py: blind vs observed cells and per channel, because a coverage
number pooled over a grid that is ~62% empty can be inflated by cells that are trivially zero.

    python3 checks/eval_uncertainty_enkf.py --k 1 4
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from crowdcore import config  # noqa: E402

SQRT_PI = np.sqrt(np.pi)
ZS = list(range(10, 100, 10))


def crps_gaussian(mu, sigma, x):
    """Closed-form CRPS of N(mu, sigma^2) against x (Gneiting & Raftery 2007), same as the
    4DVarNet side so the two numbers are directly comparable."""
    z = (x - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / SQRT_PI)


class Acc:
    """Streaming accumulator: the full test set is ~4.8e8 points per k, too much to hold."""

    def __init__(self):
        self.crps = self.nll = self.se = self.sig = 0.0
        self.n = 0
        self.cov = {z: 0.0 for z in ZS}

    def add(self, mu, s, x):
        if x.size == 0:
            return
        self.crps += crps_gaussian(mu, s, x).sum()
        self.nll += (0.5 * np.log(2 * np.pi * s ** 2) + (x - mu) ** 2 / (2 * s ** 2)).sum()
        self.se += ((x - mu) ** 2).sum()
        self.sig += s.sum()
        self.n += x.size
        r = np.abs(x - mu) / s
        for z in ZS:
            self.cov[z] += (r <= norm.ppf(0.5 + z / 200)).sum()

    def done(self):
        if not self.n:
            return None
        rmse = float(np.sqrt(self.se / self.n))
        return {"rmse": rmse, "crps": float(self.crps / self.n), "nll": float(self.nll / self.n),
                "sigma_mean": float(self.sig / self.n),
                "spread_skill": float((self.sig / self.n) / rmse),
                "coverage": {z: float(self.cov[z] / self.n) for z in ZS},
                "n": int(self.n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--dir-fmt", default="check_outputs/enkf_k{}_full")
    ap.add_argument("--out-fmt", default="check_outputs/eval/uncertainty_enkf_k{}.json")
    args = ap.parse_args()
    channels = config.get("grid", "channels")

    for k in args.k:
        d = os.path.join(ROOT, args.dir_fmt.format(k))
        ests = sorted(glob.glob(os.path.join(d, "est_*.npz")))
        if not ests:
            print(f"[k={k}] no est_*.npz in {d} — skipped"); continue
        acc = {t: Acc() for t in ["all", "blind", "observed"]}
        acc |= {f"channel_{c}": Acc() for c in channels}
        days = 0
        for e in ests:
            o = e.replace("est_", "obs_")
            if not os.path.exists(o):
                print(f"  ! no obs file for {os.path.basename(e)} — skipped"); continue
            ze, zo = np.load(e), np.load(o)
            Est = ze["Est"].astype(np.float64)
            S = np.maximum(ze["Spread"].astype(np.float64), 1e-12)   # guard log/divide
            X = zo["X_true"][:Est.shape[0]].astype(np.float64)
            # Omega is (T,H,W): a robot observes all 4 channels of a cell together
            Om = np.repeat(zo["Omega"][:Est.shape[0]][:, None], Est.shape[1], axis=1).astype(bool)

            acc["all"].add(Est, S, X)
            acc["blind"].add(Est[~Om], S[~Om], X[~Om])
            acc["observed"].add(Est[Om], S[Om], X[Om])
            for c, nm in enumerate(channels):
                acc[f"channel_{nm}"].add(Est[:, c], S[:, c], X[:, c])
            days += 1
            print(f"  [k={k}] {os.path.basename(e)}", flush=True)

        res = {"k": k, "days": days, "source": args.dir_fmt.format(k),
               "results": {t: a.done() for t, a in acc.items() if a.done()}}
        p = os.path.join(ROOT, args.out_fmt.format(k))
        json.dump(res, open(p, "w"), indent=2)
        print(f"\n[k={k}] {days} days")
        print(f"{'split':<22}{'RMSE':>8}{'CRPS':>9}{'NLL':>12}{'sigma':>9}{'sp/sk':>8}{'90%cov':>9}")
        for t, r in res["results"].items():
            print(f"{t:<22}{r['rmse']:>8.4f}{r['crps']:>9.4f}{r['nll']:>12.3e}"
                  f"{r['sigma_mean']:>9.4f}{r['spread_skill']:>8.3f}{r['coverage'][90] * 100:>8.2f}%")
        print(f"[out] {p}\n")


if __name__ == "__main__":
    main()
