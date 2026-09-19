"""Does sigma-hat track WHERE the error is, or only how crowded the cell is?

The picture shows large sigma exactly where the reconstruction is wrong, which is
what a useful sigma-hat looks like -- but crowded cells have both a larger error
and a larger predicted variance, so the two can agree for a reason that carries
no information. The constant-sigma null in eval_uncertainty_enkf rules out "sigma
knows nothing"; it does not rule out "sigma is a stand-in for density".

This stratifies by the true density and asks whether the correlation survives
inside a stratum, and scores a second null whose sigma is that stratum's own RMSE
-- a sigma that knows the crowding and nothing else.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from crowdcore import navigation


def crps_gaussian(mu, sigma, truth):
    """Closed-form CRPS for a Gaussian predictive distribution."""
    sigma = np.maximum(sigma, 1e-9)
    z = (truth - mu) / sigma
    pdf = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    cdf = 0.5 * (1.0 + np_erf(z / np.sqrt(2.0)))
    return sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / np.sqrt(np.pi))


def np_erf(x):
    # Abramowitz & Stegun 7.1.26; float64 here, max error ~1.5e-7, far below the
    # differences this diagnostic reports.
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741)
                * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return sign * y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--est-dir", required=True)
    p.add_argument("--obs-dir", default="check_outputs/enkf_k1_full")
    p.add_argument("--days", default="", help="comma-separated stems; default all in --est-dir")
    p.add_argument("--stride", type=int, default=20, help="frame subsampling")
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--channel", type=int, default=0, help="0=density")
    p.add_argument("--strata", type=int, default=5)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    walkable = navigation.build_valid_mask_from_config().astype(bool)
    if args.days:
        stems = args.days.split(",")
    else:
        # est_<stem>.source.json sits beside each export, so match the extension too.
        stems = sorted(f[4:-4] for f in os.listdir(args.est_dir)
                       if f.startswith("est_") and f.endswith(".npz"))

    err_all, sig_all, dens_all, mu_all, truth_all = [], [], [], [], []
    for stem in stems:
        z = np.load(os.path.join(args.est_dir, f"est_{stem}.npz"))
        o = np.load(os.path.join(args.obs_dir, f"obs_{stem}.npz"))
        sl = slice(args.warmup, None, args.stride)
        est = z["Est"][sl, args.channel]
        spread = z["Spread"][sl, args.channel]
        truth = o["X_true"][sl, args.channel]
        omega = o["Omega"][sl].astype(bool)
        blind = walkable[None] & ~omega
        err_all.append(np.abs(est - truth)[blind])
        sig_all.append(spread[blind])
        dens_all.append(truth[blind])
        mu_all.append(est[blind])
        truth_all.append(truth[blind])
        print(f"[{stem}] {blind.sum():,} blind walkable cell-frames", flush=True)

    err = np.concatenate(err_all).astype(np.float64)
    sig = np.concatenate(sig_all).astype(np.float64)
    dens = np.concatenate(dens_all).astype(np.float64)
    mu = np.concatenate(mu_all).astype(np.float64)
    truth = np.concatenate(truth_all).astype(np.float64)

    out = {"n": int(err.size), "days": stems, "stride": args.stride,
           "channel": args.channel,
           "overall_corr_sigma_abserr": float(np.corrcoef(sig, err)[0, 1])}

    # Null A: one constant sigma everywhere (the published null).
    const = float(np.sqrt(np.mean((truth - mu) ** 2)))
    crps_model = float(np.mean(crps_gaussian(mu, sig, truth)))
    crps_const = float(np.mean(crps_gaussian(mu, np.full_like(sig, const), truth)))

    # Null B: sigma = that density stratum's own RMSE. Knows the crowding, nothing else.
    edges = np.quantile(dens, np.linspace(0, 1, args.strata + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    idx = np.clip(np.searchsorted(edges, dens, side="right") - 1, 0, args.strata - 1)
    sigma_density = np.empty_like(sig)
    strata = []
    for s in range(args.strata):
        m = idx == s
        if m.sum() < 2:
            continue
        rmse_s = float(np.sqrt(np.mean((truth[m] - mu[m]) ** 2)))
        sigma_density[m] = rmse_s
        strata.append({
            "lo": float(edges[s]), "hi": float(edges[s + 1]), "n": int(m.sum()),
            "corr_within": float(np.corrcoef(sig[m], err[m])[0, 1]),
            "mean_abs_err": float(err[m].mean()), "mean_sigma": float(sig[m].mean()),
            "stratum_rmse": rmse_s,
        })
    crps_density_null = float(np.mean(crps_gaussian(mu, sigma_density, truth)))

    out.update({
        "crps_model": crps_model,
        "crps_constant_null": crps_const,
        "crps_density_null": crps_density_null,
        "vs_constant_null_pct": (crps_model / crps_const - 1.0) * 100.0,
        "vs_density_null_pct": (crps_model / crps_density_null - 1.0) * 100.0,
        "strata": strata,
    })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print()
    print(f"n = {out['n']:,} blind walkable cell-frames")
    print(f"overall corr(sigma, |err|) = {out['overall_corr_sigma_abserr']:.4f}")
    print()
    print(f"{'density stratum':>22s} {'n':>10s} {'corr':>7s} {'mean|err|':>10s} {'mean sigma':>11s}")
    for s in strata:
        print(f"[{s['lo']:9.5f},{s['hi']:9.5f}] {s['n']:10,d} {s['corr_within']:7.4f} "
              f"{s['mean_abs_err']:10.5f} {s['mean_sigma']:11.5f}")
    print()
    print(f"CRPS  model                = {crps_model:.6f}")
    print(f"CRPS  constant-sigma null  = {crps_const:.6f}   model {out['vs_constant_null_pct']:+.1f}%")
    print(f"CRPS  density-stratum null = {crps_density_null:.6f}   model {out['vs_density_null_pct']:+.1f}%")
    print()
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
