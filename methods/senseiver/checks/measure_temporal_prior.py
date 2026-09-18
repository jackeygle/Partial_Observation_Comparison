"""
measure_temporal_prior.py — stage 0 of the temporal extension: is there anything in the past to use?
====================================================================================================

Before training any temporal model, measure on the data whether a window of past
frames can carry information into the cells the robots cannot see right now:

  1. Observation age: for each blind walkable cell-frame, how many frames since that
     cell was last observed
  2. recall@k: the share of blind walkable cell-frames observed somewhere in the
     previous k-1 frames -- i.e. inside a k-frame window -- for k = 2, 4, 8, 16
  3. Temporal autocorrelation of the true vx (and density) at lags 1-16 frames;
     vx both over all walkable cells and over cells occupied at both times (empty
     cells have vx = 0 and inflate the correlation)

Gate, fixed in the plan before any temporal model was trained:
  go   if recall@4 > 30% and lag-4 vx autocorrelation (occupied at both times) > 0.5
  stop if recall@8 < 15%

Uses the first 3 TRAINING days with the training seed (123), never the test split.
Imports crowdcore directly (not methods.senseiver.dataset) and mirrors its
observation config. Pure numpy.

    source sbatch/_env.sh; python3 -u -m methods.senseiver.checks.measure_temporal_prior
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from crowdcore import config as cfg
from crowdcore import navigation as nav
from crowdcore import observation_model as om

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "check_outputs", "temporal", "prior.json")
KS = (2, 4, 8, 16)
LAGS = (1, 2, 4, 8, 16)
SEED, NDAYS, TSTEP = 123, 3, 5


class Corr:
    """Pooled Pearson correlation accumulated over chunks."""

    def __init__(self):
        self.n = 0; self.sa = self.sb = self.saa = self.sbb = self.sab = 0.0

    def add(self, a, b):
        a = a.astype(np.float64); b = b.astype(np.float64)
        self.n += a.size; self.sa += a.sum(); self.sb += b.sum()
        self.saa += (a * a).sum(); self.sbb += (b * b).sum(); self.sab += (a * b).sum()

    def value(self):
        n = self.n
        den = np.sqrt((n * self.saa - self.sa ** 2) * (n * self.sbb - self.sb ** 2))
        return float((n * self.sab - self.sa * self.sb) / den) if den > 0 else float("nan")


def main():
    chans = list(cfg.get("grid", "channels"))
    iv, idn = chans.index("vx"), chans.index("density")
    days = om.split_files("train")[:NDAYS]
    ages_all = []
    corr = {L: {"vx_all": Corr(), "vx_occupied": Corr(), "density_all": Corr()} for L in LAGS}
    for f in days:
        t0 = time.perf_counter()
        X, _ = om.load_state(f)
        X = np.asarray(X, dtype=np.float32)
        T, C, H, W = X.shape
        valid = nav.build_valid_mask_from_config(X)
        out = om.generate_observations(
            X, sensing_range=cfg.get("observation", "sensing_range"),
            num_agents=cfg.get("observation", "num_agents"),
            add_noise=cfg.get("observation", "add_noise"), seed=SEED, valid_mask=valid,
            obs_every_k=cfg.get("observation", "obs_every_k"))
        Om = out["Omega"].reshape(T, H * W)
        walk = valid.reshape(-1)

        last = np.full(H * W, -(10 ** 9), dtype=np.int64)
        ages = []
        for t in range(T):
            blind = walk & ~Om[t]
            ages.append(t - last[blind])
            last[Om[t]] = t
        ages = np.concatenate(ages)
        ages_all.append(ages)

        vx = X[:, iv].reshape(T, -1)[:, walk]
        de = X[:, idn].reshape(T, -1)[:, walk]
        for L in LAGS:
            ts = np.arange(0, T - L, TSTEP)
            a, b = vx[ts], vx[ts + L]
            corr[L]["vx_all"].add(a, b)
            occ = (de[ts] > 0) & (de[ts + L] > 0)
            corr[L]["vx_occupied"].add(a[occ], b[occ])
            corr[L]["density_all"].add(de[ts], de[ts + L])
        print(f"  {os.path.basename(f).split('_')[0]}: {T} frames, walkable {int(walk.sum())}, "
              f"observed/frame {Om[:, walk].sum(1).mean():.1f}, blind walkable cell-frames {ages.size:,} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)

    ages = np.concatenate(ages_all)
    never = ages > 10 ** 8
    fin = ages[~never]
    res = {"days": [os.path.basename(f).split("_")[0] for f in days], "seed": SEED,
           "blind_walkable_cell_frames": int(ages.size),
           "never_observed_before_share": float(never.mean()),
           "age_percentiles_frames": {p: float(np.percentile(fin, p)) for p in (10, 25, 50, 75, 90)},
           "recall_at_k": {k: float((ages <= k - 1).mean()) for k in KS},
           "autocorr": {L: {kname: c.value() for kname, c in corr[L].items()} for L in LAGS}}

    print(f"\nBlind walkable cell-frames: {ages.size:,}   never observed earlier that day: "
          f"{res['never_observed_before_share'] * 100:.2f}%")
    print("Age since last observation (frames): "
          + "  ".join(f"P{p} {v:.0f}" for p, v in res["age_percentiles_frames"].items()))
    print("recall@k (blind cell seen inside a k-frame window): "
          + "  ".join(f"k={k} {v * 100:.1f}%" for k, v in res["recall_at_k"].items()))
    print(f"\n{'lag':>5}{'vx all':>10}{'vx occupied':>14}{'density':>10}")
    for L in LAGS:
        a = res["autocorr"][L]
        print(f"{L:>5}{a['vx_all']:>10.3f}{a['vx_occupied']:>14.3f}{a['density_all']:>10.3f}")

    r4, r8, ac4 = res["recall_at_k"][4], res["recall_at_k"][8], res["autocorr"][4]["vx_occupied"]
    if r8 < 0.15:
        verdict = "STOP"
    elif r4 > 0.30 and ac4 > 0.5:
        verdict = "GO"
    else:
        verdict = "BORDERLINE (neither go nor stop condition met)"
    res["gate"] = {"recall_at_4": r4, "recall_at_8": r8, "vx_occupied_lag4": ac4, "verdict": verdict}
    print(f"\nGate: recall@4 {r4 * 100:.1f}% (go > 30%), vx lag-4 occupied {ac4:.3f} (go > 0.5), "
          f"recall@8 {r8 * 100:.1f}% (stop < 15%)  ->  {verdict}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[out] {OUT}")


if __name__ == "__main__":
    main()
