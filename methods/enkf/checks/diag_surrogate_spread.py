"""
diag_surrogate_spread.py — does the stochastic surrogate hold an ensemble spread, open loop?

The shipped EnKF collapses because its forecast model is deterministic and damps member
differences, with only 0.01 x Q noise added back (checks/diag_enkf_spread_growth.py). This runs
the same kind of open-loop test for the deep-ensemble surrogate: start 100 members at the TRUE
state (zero spread), propagate with no analysis step, and every step each member draws a pair s
and moves to  surrogate_mean_s(x) + sigma_s(x) * eps  (then the EnKF's clipping). A healthy
ensemble grows its spread together with its actual error, sqrt(mean var) / RMSE near 1.

The reference arm is the shipped forecast: vendored PedPred3 + N(0, 0.01 x PROC_STD), clipped.
Several start frames on one validation day; walkable cells, per channel.

    source sbatch/_env.sh && cd methods/enkf
    python3 -u -m methods.enkf.checks.diag_surrogate_spread
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from crowdcore import navigation as nav
from crowdcore import observation_model as om
from crowdcore import paths
from methods.enkf.surrogate.model import CH, CLIP_HI, CLIP_LO, load_pedpred3, mean_forecast, raw_forecast, surrogate_mean
from methods.enkf.surrogate.train import LOGVAR_MAX, LOGVAR_MIN

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # methods/enkf
PROC_STD = (0.02829307, 0.31263075, 0.12325809, 0.41680932)            # run_enkf_baseline.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--day", type=int, default=0, help="index into the valid split")
    ap.add_argument("--starts", type=int, nargs="+", default=[5000, 12000, 19000, 26000, 33000])
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--ensemble", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(HERE, "check_outputs", "eval", "surrogate_spread_valid.json"))
    ap.add_argument("--allow-cpu", action="store_true")
    a = ap.parse_args()
    if not torch.cuda.is_available() and not a.allow_cpu:
        raise SystemExit("no GPU -- run on a GPU node; add --allow-cpu to force CPU")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    day = om.split_files("valid")[a.day]
    X = torch.from_numpy(om.load_state(day)[0]).to(dev)
    walk = torch.from_numpy(nav.build_valid_mask_from_config().astype(bool)).to(dev)
    lo = torch.tensor(CLIP_LO, device=dev).view(1, 1, 4, 1, 1)
    hi = torch.tensor(CLIP_HI, device=dev).view(1, 1, 4, 1, 1)
    clip = lambda t: torch.maximum(torch.minimum(t, hi), lo)
    runs = os.path.join(HERE, "runs")
    pairs = [(load_pedpred3(f"{runs}/surrogate_mean_s{s}/best.pt", dev).eval(),
              load_pedpred3(f"{runs}/surrogate_sigma_s{s}/best.pt", dev).eval()) for s in a.seeds]
    orig = load_pedpred3(os.path.join(paths.enkf_vendor("enkf_lab"), "apt-ibex_train_model_28D.pth"), dev).eval()
    inj = 0.01 * torch.tensor(PROC_STD, device=dev).view(1, 1, 4, 1, 1)

    def step_surrogate(E):
        pick = torch.randint(len(pairs), (len(E),), device=dev)
        out = torch.empty_like(E)
        for j, (mean_net, sigma_net) in enumerate(pairs):
            idx = (pick == j).nonzero().squeeze(1)
            if idx.numel():
                xe = E[idx]
                sd = torch.exp(0.5 * raw_forecast(sigma_net, xe).clamp(LOGVAR_MIN, LOGVAR_MAX))
                out[idx] = surrogate_mean(mean_net, xe) + sd * torch.randn_like(xe)
        return clip(out)

    def step_shipped(E):
        return clip(mean_forecast(orig, E) + inj * torch.randn_like(E))

    arms = {"surrogate ensemble": step_surrogate, "shipped (PedPred3 + 0.01 x Q)": step_shipped}
    starts = [t for t in a.starts if t + a.steps < len(X)]
    # squared spread / squared error summed over starts, per arm, step, channel
    sp2 = {n: torch.zeros(a.steps + 1, 4, device=dev, dtype=torch.float64) for n in arms}
    er2 = {n: torch.zeros(a.steps + 1, 4, device=dev, dtype=torch.float64) for n in arms}
    with torch.no_grad():
        for name, step in arms.items():
            for t0 in starts:
                E = X[t0].view(1, 1, 4, *X.shape[-2:]).repeat(a.ensemble, 1, 1, 1, 1)
                for k in range(1, a.steps + 1):
                    E = step(E)
                    var = E.var(dim=0)[0][:, walk]                              # (4, n_walk)
                    err = (E.mean(dim=0)[0] - X[t0 + k])[:, walk]
                    sp2[name][k] += var.mean(dim=1).double()
                    er2[name][k] += (err * err).mean(dim=1).double()

    res = {n: {k: {ch: {"spread": float(torch.sqrt(sp2[n][k, j] / len(starts))),
                        "rmse": float(torch.sqrt(er2[n][k, j] / len(starts)))} for j, ch in enumerate(CH)}
               for k in range(1, a.steps + 1)} for n in arms}
    doc = {"day": os.path.basename(day), "starts": starts, "ensemble": a.ensemble, "seeds": a.seeds, "results": res}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(doc, open(a.out, "w"), indent=2)

    print(f"open loop from the true state, {os.path.basename(day)}, starts {starts}, walkable cells")
    print("spread = sqrt(mean ensemble variance), rmse = ensemble-mean error; ratio = spread / rmse")
    for n in arms:
        print(f"\n{n}")
        print(f"{'step':>5} | " + " | ".join(f"{ch:^22}" for ch in CH))
        for k in [1, 2, 3, 5, 10, 20, 30, 40]:
            if k > a.steps:
                continue
            cells = []
            for ch in CH:
                v = res[n][k][ch]
                cells.append(f"{v['spread']:.4f}/{v['rmse']:.4f} {v['spread'] / max(v['rmse'], 1e-12):>5.2f}")
            print(f"{k:>5} | " + " | ".join(cells))
    print(f"\n[out] {a.out}")


if __name__ == "__main__":
    main()
