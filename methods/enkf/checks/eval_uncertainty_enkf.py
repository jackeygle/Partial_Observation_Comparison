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

# The scoring implementation lives in compare/score_uncertainty.py -- this file
# used to have a second verbatim copy, kept in sync only by a comment saying
# "same as the 4DVarNet side so the two numbers are directly comparable". That is
# a promise only a human can keep; DINCAE joining table B would have made a third
# copy, so it was pulled out. Numerical equivalence checked: this file's CRPS vs
# the new implementation, max|d| = 0.000e+00.
from compare import score_uncertainty as su                          # noqa: E402

ZS = list(su.ZS)
crps_gaussian = su.crps_gaussian
Acc = su.Accumulator


def _defined(X):
    """(T,C,H,W) bool -- that channel is defined here ∩ walkable. The cell set
    used by table A's main convention.

    `channel_valid` is imported from DINCAE's side, the rule is never copied
    (same source as compare5.py). It returns (C,T,H,W); transposed back to
    (T,C,H,W) here to line up with Est/X's layout.
    """
    from crowdcore import navigation as nav
    from methods.dincae.state import channel_valid
    cv = np.ascontiguousarray(channel_valid(X).transpose(1, 0, 2, 3))
    return cv & nav.build_valid_mask_from_config()[None, None].astype(bool)


def days_of(ests):
    """Yield (name, Est, Spread, X_true, Omega) per day. Shared by both passes so the two
    reads cannot drift apart in how they build the mask or cast dtypes."""
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
        yield os.path.basename(e), Est, S, X, Om


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--dir-fmt", default="check_outputs/enkf_k{}_full")
    ap.add_argument("--out-fmt", default="check_outputs/eval/uncertainty_enkf_k{}.json")
    ap.add_argument("--no-const-baseline", dest="const_baseline", action="store_false",
                    help="skip the constant-sigma null model (saves one pass over the exports)")
    args = ap.parse_args()
    channels = config.get("grid", "channels")

    for k in args.k:
        d = os.path.join(ROOT, args.dir_fmt.format(k))
        ests = sorted(glob.glob(os.path.join(d, "est_*.npz")))
        if not ests:
            print(f"[k={k}] no est_*.npz in {d} — skipped"); continue
        acc = {t: Acc() for t in ["all", "blind", "observed",
                                  "defined", "defined_blind", "defined_observed"]}
        acc |= {f"channel_{c}": Acc() for c in channels}
        days = 0
        for nm_, Est, S, X, Om in days_of(ests):
            acc["all"].add(Est, S, X)
            acc["blind"].add(Est[~Om], S[~Om], X[~Om])
            acc["observed"].add(Est[Om], S[Om], X[Om])
            # Main convention defined = that channel is defined ∩ walkable, the same
            # cells as table A's main result and DINCAE's sigma-hat scoring
            # convention. The rule is never copied, just imported from its single definition.
            D = _defined(X)
            acc["defined"].add(Est[D], S[D], X[D])
            acc["defined_blind"].add(Est[D & ~Om], S[D & ~Om], X[D & ~Om])
            acc["defined_observed"].add(Est[D & Om], S[D & Om], X[D & Om])
            for c, nm in enumerate(channels):
                acc[f"channel_{nm}"].add(Est[:, c], S[:, c], X[:, c])
            days += 1
            print(f"  [k={k}] {nm_}", flush=True)

        # ---- second pass: the constant-sigma null model -----------------------------
        # sigma is that split's own RMSE, so it cannot be scored in the pass that measures
        # it; and Acc streams (~4.8e8 points per k) so the residuals cannot be kept around.
        # Hence a second read. Only the two baselines are accumulated here, so this pass
        # skips the per-channel work.
        if args.const_baseline:
            cs = {t: acc[t].result()["rmse"]
                  for t in ("all", "blind", "defined", "defined_blind") if acc[t].result()}
            for t in cs:
                acc[f"{t}_constant_sigma_baseline"] = Acc()
            print(f"  [k={k}] second pass, constant sigma = "
                  + ", ".join(f"{t} {v:.4f}" for t, v in cs.items()), flush=True)
            for nm_, Est, S, X, Om in days_of(ests):
                if "all" in cs:
                    acc["all_constant_sigma_baseline"].add(
                        Est, np.full_like(Est, cs["all"]), X)
                if "blind" in cs:
                    Eb = Est[~Om]
                    acc["blind_constant_sigma_baseline"].add(
                        Eb, np.full_like(Eb, cs["blind"]), X[~Om])
                D = _defined(X)
                for t, sel in (("defined", D), ("defined_blind", D & ~Om)):
                    if t in cs:
                        E = Est[sel]
                        acc[f"{t}_constant_sigma_baseline"].add(
                            E, np.full_like(E, cs[t]), X[sel])

        res = {"k": k, "days": days, "source": args.dir_fmt.format(k),
               "results": {t: a.result() for t, a in acc.items() if a.result()}}
        p = os.path.join(ROOT, args.out_fmt.format(k))
        # The 2026-09-03 refactor moved this script from methods/varnet/checks to
        # methods/enkf/checks, so ROOT (and with it the default --out-fmt) now points at a
        # check_outputs/eval that nothing had created yet. Two full passes over 17 GB of
        # exports were lost to that before this line existed.
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(res, open(p, "w"), indent=2)
        print(f"\n[k={k}] {days} days")
        print(f"{'split':<22}{'RMSE':>8}{'CRPS':>9}{'NLL':>12}{'sigma':>9}{'sp/sk':>8}{'90%cov':>9}")
        for t, r in res["results"].items():
            print(f"{t:<22}{r['rmse']:>8.4f}{r['crps']:>9.4f}{r['nll']:>12.3e}"
                  f"{r['sigma_mean']:>9.4f}{r['spread_skill']:>8.3f}{r['coverage'][90] * 100:>8.2f}%")
        print(f"[out] {p}\n")


if __name__ == "__main__":
    main()
