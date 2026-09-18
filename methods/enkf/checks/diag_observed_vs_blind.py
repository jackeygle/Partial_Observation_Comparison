"""
diag_observed_vs_blind.py — is "observed cells score worse than blind cells" an assimilation
failure, or a composition effect?

`methods/enkf/README.md` reports the EnKF as "the only method whose **observed** cells score
worse than its **blind** cells -- the gain is so small the filter barely assimilates what it
sees", and uncertainty_enkf_k1.json backs the raw fact (RMSE blind 0.2150 vs observed 0.2632).

That inference has a hole. The two cell sets are not exchangeable:

  * a cell is "observed" when a robot is within sensing range of it, and the robots patrol the
    main corridor, which is also where the crowd is;
  * the error of every channel grows with how busy a cell is (an empty cell's density is 0 and
    its velocity is the placeholder 0 -- trivially predictable);
  * so "observed" may simply be a sample of intrinsically harder cells.

If that is what is happening, the gap should collapse once the two sets are compared at matched
true density. If instead the gap survives within every density stratum, the README's reading
stands and the analysis really is hurting where it is applied.

Localisation makes the confound worse, not better: with radius 7 on a 36x12 grid, a cell that
is "blind" this frame is usually still inside some observed cell's localisation kernel and gets
corrected anyway -- so this split was never a clean assimilated/not-assimilated contrast.

Pure post-processing of the exported npz -- no model, no filter re-run.

    python3 -m methods.enkf.checks.diag_observed_vs_blind
"""
from __future__ import annotations
import argparse
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from crowdcore import paths                                              # noqa: E402

CH = ("density", "vx", "vy", "var")
# Strata on the cell's TRUE density -- "how busy is this cell", the covariate that both drives
# the error and decides whether a robot is likely to be nearby.
EDGES = [0.0, 1e-9, 0.1, 0.25, 0.5, np.inf]
NAMES = ["empty (=0)", "(0, 0.1]", "(0.1, 0.25]", "(0.25, 0.5]", "> 0.5"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=paths.enkf_export("enkf_k1_full"))
    ap.add_argument("--days", type=int, default=0, help="0 = all")
    ap.add_argument("--out", default=os.path.join(paths.eval_out(paths.ENKF),
                                                  "observed_vs_blind.json"))
    a = ap.parse_args()

    obs_files = sorted(glob.glob(os.path.join(a.dir, "obs_*.npz")))
    if a.days:
        obs_files = obs_files[:a.days]

    nb = len(NAMES)
    # [split][stratum][channel] -> running sum of squared error, and count
    se = {s: np.zeros((nb, 4)) for s in ("observed", "blind")}
    n = {s: np.zeros((nb, 4), dtype=np.int64) for s in ("observed", "blind")}

    for f in obs_files:
        day = os.path.basename(f)[4:-4]
        est_p = os.path.join(a.dir, f"est_{day}.npz")
        if not os.path.exists(est_p):
            print(f"  {day}: no est_*.npz, skipped", flush=True)
            continue
        with np.load(f) as zo:
            X = zo["X_true"]
            Om = zo["Omega"]
        with np.load(est_p) as ze:
            Est = ze["Est"]
        T = min(len(X), len(Est))
        X, Om, Est = X[:T], Om[:T], Est[:T]
        d2 = (Est.astype(np.float64) - X.astype(np.float64)) ** 2      # (T,4,H,W)
        # stratum index per (frame, cell), from the cell's true density
        strat = np.digitize(X[:, 0], EDGES[1:-1], right=True)          # (T,H,W) in 0..nb-1
        omb = Om.astype(bool)
        for si in range(nb):
            m = strat == si
            for split, sel in (("observed", m & omb), ("blind", m & ~omb)):
                if not sel.any():
                    continue
                for c in range(4):
                    v = d2[:, c][sel]
                    se[split][si, c] += v.sum()
                    n[split][si, c] += v.size
        print(f"  {day}: {T} frames", flush=True)

    def rmse(split, si, c):
        return float(np.sqrt(se[split][si, c] / n[split][si, c])) if n[split][si, c] else None

    res = {"dir": a.dir, "days": len(obs_files), "strata": NAMES, "channels": list(CH),
           "by_stratum": {}, "pooled": {}}

    print("\n=== pooled (the number the README quotes) ===")
    print(f"{'':14s}" + "".join(f"{c:>11s}" for c in CH) + f"{'share':>9s}")
    for split in ("blind", "observed"):
        tot_se = se[split].sum(axis=0)
        tot_n = n[split].sum(axis=0)
        r = np.sqrt(tot_se / np.maximum(tot_n, 1))
        share = tot_n[0] / max(n["blind"][:, 0].sum() + n["observed"][:, 0].sum(), 1)
        res["pooled"][split] = {CH[c]: float(r[c]) for c in range(4)}
        res["pooled"][split]["n"] = int(tot_n[0])
        print(f"  {split:12s}" + "".join(f"{r[c]:11.4f}" for c in range(4)) + f"{share:9.1%}")

    print("\n=== 每个密度分层里,observed 和 blind 各自的 RMSE ===")
    print("(若差距是构成效应,分层后应当消失或反转)")
    for si, nm in enumerate(NAMES):
        tot = n["observed"][si, 0] + n["blind"][si, 0]
        frac_obs = n["observed"][si, 0] / max(tot, 1)
        print(f"\n  {nm}   该层占全部格子 {tot / max(sum(n['observed'][:, 0] + n['blind'][:, 0]), 1):.1%}"
              f" | 其中被观测到 {frac_obs:.1%}")
        print(f"{'':16s}" + "".join(f"{c:>11s}" for c in CH))
        row = {}
        for split in ("blind", "observed"):
            vals = [rmse(split, si, c) for c in range(4)]
            row[split] = {CH[c]: vals[c] for c in range(4)}
            print(f"    {split:12s}" + "".join(
                f"{v:11.4f}" if v is not None else f"{'-':>11s}" for v in vals))
        ratio = [(row["observed"][CH[c]] / row["blind"][CH[c]])
                 if row["blind"][CH[c]] else None for c in range(4)]
        row["obs_over_blind"] = {CH[c]: ratio[c] for c in range(4)}
        print(f"    {'比值 o/b':12s}" + "".join(
            f"{v:11.3f}" if v is not None else f"{'-':>11s}" for v in ratio))
        res["by_stratum"][nm] = row

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2, ensure_ascii=False)
    print(f"\n[out] {a.out}")


if __name__ == "__main__":
    main()
