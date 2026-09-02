"""
eval_uncertainty.py  —  is the predicted uncertainty any good, and is it better than the EnKF's?

Scores a deep ensemble (Lakshminarayanan et al. 2017, arXiv:1612.01474) against the EnKF's
ensemble spread on the held-out test days, on four metrics:

    CRPS              the headline. A proper scoring rule that penalises both a wrong mean and
                      a wrong spread, and unlike NLL it stays finite and interpretable when a
                      method is wildly overconfident -- which the EnKF is (its spread is ~0.011
                      of its own RMSE), so NLL alone would explode into an unreadable number.
    NLL               their metric, for direct comparability with the paper.
    spread-skill      mean predicted sigma / RMSE. 1.0 is perfect; below 1 is overconfident.
    reliability       their Appendix A.2 protocol: build the z% Gaussian interval from
                      (mean, sigma) and count how often the truth falls inside. Ideal is z%.

Ensemble combination follows their Sec. 2.4 exactly:

    mu*     = mean_m mu_m
    sigma*^2 = mean_m (sigma_m^2 + mu_m^2) - mu*^2
             = mean_m sigma_m^2   +   var_m mu_m
               ^ aleatoric            ^ epistemic

Both halves are reported separately, because they answer different questions and only one of
them is the analogue of the EnKF's spread. The EnKF perturbs each member's observations
(innov = (y_obs + noise) - Y_f), so its spread already carries observation noise: the
structurally matched comparison is against the TOTAL sigma*. The epistemic half is what
measures disagreement between independently trained models, and it is the honest number to
quote when claiming the model knows where it is unsure.

Three splits are mandatory, not optional:
  * blind vs observed cells -- ~62% of cells are empty and a coverage number computed over all
    of them can be inflated by cells that are trivially zero everywhere;
  * per channel -- density is mostly zeros, `var` lives three orders of magnitude away;
  * against a CONSTANT sigma baseline fitted to the global RMSE. If the learned sigma cannot
    beat "everyone gets the average error", it carries no information, and no amount of
    favourable comparison against the EnKF would make it useful.

Run (CPU is fine, but the solves are the slow part -- prefer a GPU node):
    python3 checks/eval_uncertainty.py --members 0 1 2 3 4 --days 7
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "checks"))
import config  # noqa: E402
import navigation as nav  # noqa: E402
import observation_model as om  # noqa: E402
from model_io import load_solver  # noqa: E402

SQRT2, SQRT_PI = np.sqrt(2.0), np.sqrt(np.pi)


def crps_gaussian(mu, sigma, x):
    """Closed-form CRPS of N(mu, sigma^2) against the observed x (Gneiting & Raftery 2007).

        CRPS = sigma * [ z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ],  z = (x - mu)/sigma

    Lower is better, and it is in the units of x, so it can be read next to the RMSE.
    """
    z = (x - mu) / sigma
    Phi = 0.5 * (1.0 + torch.erf(z / SQRT2))
    phi = torch.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    return sigma * (z * (2 * Phi - 1) + 2 * phi - 1.0 / SQRT_PI)


def nll_gaussian(mu, sigma, x):
    return 0.5 * torch.log(2 * np.pi * sigma ** 2) + (x - mu) ** 2 / (2 * sigma ** 2)


def coverage(mu, sigma, x, zs=range(10, 100, 10)):
    """Fraction of truths inside the nominal z% Gaussian interval (their App. A.2)."""
    from scipy.stats import norm
    err = (x - mu).abs()
    return {z: float((err <= norm.ppf(0.5 + z / 200) * sigma).float().mean()) for z in zs}


def score(mu, sigma, x, tag, out):
    out[tag] = {
        "rmse": float(torch.sqrt(((x - mu) ** 2).mean())),
        "crps": float(crps_gaussian(mu, sigma, x).mean()),
        "nll": float(nll_gaussian(mu, sigma, x).mean()),
        "sigma_mean": float(sigma.mean()),
        "spread_skill": float(sigma.mean() / torch.sqrt(((x - mu) ** 2).mean())),
        "coverage": coverage(mu, sigma, x),
        "n": int(x.numel()),
    }
    return out[tag]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--run-fmt", default="runs/varnet_ml5_s{}")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--frames", type=int, default=0, help="0 = whole day")
    # Windows are solved in batches like checks/eval_test_days.py. A whole day at dT=200 is
    # ~7 windows, and 20 unrolled solver iterations with autograd over all of them at once
    # needs >16 GB -- it OOMs on a 16 GB debug GPU. This is purely a memory knob; results do
    # not depend on it.
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="check_outputs/eval/uncertainty_ml5.json")
    args = ap.parse_args()
    dev = torch.device(args.device)

    # ---- members -------------------------------------------------------------
    solvers, A = [], None
    for s in args.members:
        p = os.path.join(ROOT, args.run_fmt.format(s), "varnet_best.pt")
        sol, A, ck = load_solver(p, dev)
        if sol.grad_net.out_var is None:
            raise SystemExit(f"{p} has no variance read-out — was it trained with --loss nll?")
        solvers.append(sol)
    dT, k = A["dT"], A.get("obs_every_k") or config.get("observation", "obs_every_k")
    print(f"[members] {len(solvers)} x {args.run_fmt}  dT={dT}  obs_every_k={k}  dev={dev}",
          flush=True)

    files = om.split_files("test")[:args.days]
    acc = {t: [] for t in ("mu", "sig_tot", "sig_ale", "sig_epi", "x", "mask")}
    for f in files:
        X = np.asarray(om.load_state(f)[0])
        if args.frames:
            X = X[:args.frames]
        X = X[:(len(X) // dT) * dT]
        valid = nav.build_valid_mask_from_config(X)
        # identical call to checks/eval_test_days.py so the two are directly comparable
        o = om.generate_observations(X, add_noise=True, valid_mask=valid, obs_every_k=k)
        x0 = om.fill_missing_state(o["Y"], o["Omega_c"],
                                   method=config.get("observation", "init_method"))
        w = lambda a: om.to_windows(a, dT)                 # keep on CPU; batches move to dev
        Yw, Mw, X0w, Xw = (torch.from_numpy(w(o["Y"])).float(),
                           torch.from_numpy(w(o["Omega_c"].astype(np.float32))).float(),
                           torch.from_numpy(w(x0)).float(),
                           torch.from_numpy(w(np.asarray(X))).float())
        for i in range(0, Xw.shape[0], args.batch):
            yb = Yw[i:i + args.batch].to(dev); mb = Mw[i:i + args.batch].to(dev)
            x0b = X0w[i:i + args.batch].to(dev)
            mus, vars_ = [], []
            for sol in solvers:
                with torch.enable_grad():
                    xr, vr = sol(x0b.clone(), yb, mb, return_var=True)
                mus.append(xr.detach())
                vars_.append(vr.detach())
            mu = torch.stack(mus); var = torch.stack(vars_)
            # Sec. 2.4 moment matching, split into its two halves
            mu_s = mu.mean(0)
            ale = var.mean(0)                     # mean of the members' own sigma^2
            epi = mu.var(0, unbiased=False)        # disagreement between members
            acc["mu"].append(mu_s.cpu()); acc["sig_tot"].append((ale + epi).sqrt().cpu())
            acc["sig_ale"].append(ale.sqrt().cpu()); acc["sig_epi"].append(epi.sqrt().cpu())
            acc["x"].append(Xw[i:i + args.batch]); acc["mask"].append(mb.cpu())
            del mu, var, mus, vars_, xr
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        print(f"  {os.path.basename(f)}  {Xw.shape[0]} windows", flush=True)

    C = {t: torch.cat(v) for t, v in acc.items()}
    x, mu, m = C["x"], C["mu"], C["mask"]
    res = {"members": args.members, "days": len(files), "obs_every_k": k, "results": {}}
    R = res["results"]

    # ---- headline + the three mandatory splits -------------------------------
    score(mu, C["sig_tot"], x, "all", R)
    score(mu, C["sig_ale"], x, "all_aleatoric_only", R)
    score(mu, C["sig_epi"].clamp_min(1e-6), x, "all_epistemic_only", R)
    # constant-sigma null model: everyone gets the global RMSE. If the learned sigma cannot
    # beat this, it is not carrying information about WHERE the model is wrong.
    const = torch.full_like(C["sig_tot"], float(torch.sqrt(((x - mu) ** 2).mean())))
    score(mu, const, x, "all_constant_sigma_baseline", R)

    blind, obs = m == 0, m == 1
    for tag, sel in (("blind", blind), ("observed", obs)):
        score(mu[sel], C["sig_tot"][sel], x[sel], tag, R)
    for c, nm in enumerate(config.get("grid", "channels")):
        score(mu[:, c], C["sig_tot"][:, c], x[:, c], f"channel_{nm}", R)

    p = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, "w"), indent=2)

    print(f"\n{'split':<30}{'RMSE':>8}{'CRPS':>9}{'NLL':>9}{'sigma':>8}{'sp/sk':>7}{'90%cov':>8}")
    for t, r in R.items():
        print(f"{t:<30}{r['rmse']:>8.4f}{r['crps']:>9.4f}{r['nll']:>9.3f}"
              f"{r['sigma_mean']:>8.4f}{r['spread_skill']:>7.2f}{r['coverage'][90]*100:>7.1f}%")
    print(f"\n[out] {p}")
    print(f"\nEnKF k=1 for reference (measured, check_outputs/enkf_k1_full):"
          f"  spread/skill 0.011   90% coverage 2.24%")


if __name__ == "__main__":
    main()
