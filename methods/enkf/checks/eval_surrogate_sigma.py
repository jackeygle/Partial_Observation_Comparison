"""
eval_surrogate_sigma.py — is the sigma network's uncertainty calibrated, one step ahead?

Scores the surrogate's one-step forecast distribution N(mu, sigma^2) on the VALID split, where
mu = model.surrogate_mean (the mean the EnKF propagates) and sigma^2 = exp(raw sigma output),
clamped as in training. Each pair s = (surrogate_mean_s/best.pt, surrogate_sigma_s/best.pt) on
its own, and the 5-pair ensemble as one Gaussian with the mixture's mean and variance
(Lakshminarayanan et al. 2017, Sec. 2.4):
    mu* = mean_s mu_s,   sigma*^2 = mean_s (sigma_s^2 + mu_s^2) - mu*^2
Per channel, on walkable cells, occupied walkable cells (true density > 0) and empty walkable
cells. Same closed-form CRPS and coverage as compare/score_uncertainty.py, computed on the GPU,
plus two calibration readings that spread/skill (mean sigma / RMSE) does not give:
  rms sp/sk  sqrt(mean sigma^2) / RMSE -- mean sigma / RMSE is biased low by Jensen whenever
             sigma varies from cell to cell, even for a perfectly calibrated sigma
  E[z^2]     z = (x - mu) / sigma; 1.0 when calibrated
and CRPS against a per-channel constant-sigma null (sigma = that channel's RMSE on that cell set).

    source sbatch/_env.sh && cd methods/enkf
    python3 -u -m methods.enkf.checks.eval_surrogate_sigma
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch

from crowdcore import navigation as nav
from methods.enkf.surrogate.data import PairFrames
from methods.enkf.surrogate.model import CH, load_pedpred3, raw_forecast, surrogate_mean
from methods.enkf.surrogate.train import LOGVAR_MAX, LOGVAR_MIN

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # methods/enkf
Z50, Z90 = 0.6744897501960817, 1.6448536269514722                     # norm.ppf(0.75), norm.ppf(0.95)
STATS = ("se", "crps", "sig", "sig2", "z2", "c50", "c90", "crps_null")


def crps_gaussian(mu, sigma, x):
    """Gneiting & Raftery 2007, as compare/score_uncertainty.crps_gaussian."""
    z = (x - mu) / sigma
    cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    pdf = torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    return sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


def forecasts(pairs, x):
    """{name: (mu, sigma^2)} for every pair and, with more than one, their ensemble."""
    out, mus, vs = {}, [], []
    for s, (mean_net, sigma_net) in pairs.items():
        mu = surrogate_mean(mean_net, x)
        var = torch.exp(raw_forecast(sigma_net, x).clamp(LOGVAR_MIN, LOGVAR_MAX))
        out[f"pair s{s}"] = (mu, var)
        mus.append(mu)
        vs.append(var)
    if len(pairs) > 1:
        M, V = torch.stack(mus), torch.stack(vs)
        m = M.mean(0)
        out[f"ensemble of {len(pairs)}"] = (m, ((V + M * M).mean(0) - m * m).clamp_min(math.exp(LOGVAR_MIN)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(HERE, "check_outputs", "eval", "surrogate_sigma_valid.json"))
    ap.add_argument("--allow-cpu", action="store_true")
    a = ap.parse_args()
    if not torch.cuda.is_available() and not a.allow_cpu:
        raise SystemExit("no GPU -- run on a GPU node; add --allow-cpu to force CPU")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runs = os.path.join(HERE, "runs")
    pairs = {s: (load_pedpred3(f"{runs}/surrogate_mean_s{s}/best.pt", dev).eval(),
                 load_pedpred3(f"{runs}/surrogate_sigma_s{s}/best.pt", dev).eval()) for s in a.seeds}
    names = [f"pair s{s}" for s in a.seeds] + ([f"ensemble of {len(a.seeds)}"] if len(a.seeds) > 1 else [])
    valid = PairFrames("valid", dev, a.days, a.max_frames)
    walk = torch.from_numpy(nav.build_valid_mask_from_config().astype(bool)).to(dev).view(1, 1, 1, *valid.frames.shape[-2:])
    scopes = ("walkable", "occupied", "empty")
    acc = {(n, sc): {k: torch.zeros(4, device=dev, dtype=torch.float64) for k in STATS} for n in names for sc in scopes}
    count = {sc: 0 for sc in scopes}

    def masks(y):
        d = y[:, :, 0:1]
        return {"walkable": walk.expand_as(d), "occupied": walk & (d > 0), "empty": walk & (d == 0)}

    def batches():
        for i in range(0, valid.n_pairs, a.batch):
            yield valid.batch(torch.arange(i, min(i + a.batch, valid.n_pairs), device=dev))

    with torch.no_grad():
        for x, y in batches():                                            # pass 1: the forecast itself
            M = {sc: m.double() for sc, m in masks(y).items()}
            for sc, m in M.items():
                count[sc] += int(m.sum())
            for n, (mu, var) in forecasts(pairs, x).items():
                sig = var.sqrt()
                e = y - mu
                terms = {"se": e * e, "crps": crps_gaussian(mu, sig, y), "sig": sig, "sig2": var,
                         "z2": e * e / var, "c50": (e.abs() <= Z50 * sig).float(), "c90": (e.abs() <= Z90 * sig).float()}
                for sc, m in M.items():
                    for k, t in terms.items():
                        acc[(n, sc)][k] += (t.double() * m).sum(dim=(0, 1, 3, 4))
        rmse = {key: torch.sqrt(d["se"] / count[key[1]]) for key, d in acc.items()}
        for x, y in batches():                                            # pass 2: constant-sigma null
            M = {sc: m.double() for sc, m in masks(y).items()}
            for n, (mu, _) in forecasts(pairs, x).items():
                for sc, m in M.items():
                    s0 = rmse[(n, sc)].float().clamp_min(1e-12).view(1, 1, 4, 1, 1).expand_as(mu)
                    acc[(n, sc)]["crps_null"] += (crps_gaussian(mu, s0, y).double() * m).sum(dim=(0, 1, 3, 4))

    res = {}
    for (n, sc), d in acc.items():
        c, r = count[sc], rmse[(n, sc)]
        res.setdefault(n, {})[sc] = {ch: {"rmse": float(r[j]), "spread_skill": float(d["sig"][j] / c / r[j]),
                                          "rms_spread_skill": float(torch.sqrt(d["sig2"][j] / c) / r[j]),
                                          "mean_z2": float(d["z2"][j] / c), "cov50": float(d["c50"][j] / c),
                                          "cov90": float(d["c90"][j] / c), "crps": float(d["crps"][j] / c),
                                          "crps_over_null": float(d["crps"][j] / d["crps_null"][j])}
                                     for j, ch in enumerate(CH)}
    doc = {"split": "valid", "n_days": valid.n_days, "n_pairs": valid.n_pairs, "cells": count, "results": res}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(doc, open(a.out, "w"), indent=2)

    for sc in scopes:
        print(f"\n== {sc} cells ({count[sc]:,} cell-frames), valid {valid.n_days} days")
        print(f"{'model':<16}{'ch':<8}{'RMSE':>8}{'sp/sk':>7}{'rms':>7}{'E[z2]':>8}{'cov50':>7}{'cov90':>7}{'CRPS/null':>10}")
        for n in names:
            for ch in CH:
                v = res[n][sc][ch]
                print(f"{n:<16}{ch:<8}{v['rmse']:>8.4f}{v['spread_skill']:>7.2f}{v['rms_spread_skill']:>7.2f}"
                      f"{v['mean_z2']:>8.2f}{v['cov50']:>7.2f}{v['cov90']:>7.2f}{v['crps_over_null']:>10.3f}")
    print(f"\n[out] {a.out}")


if __name__ == "__main__":
    main()
