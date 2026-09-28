"""Cache one test day's observation statistics for the observation-model figure.

    python3 -m thesis_figures.prepare_day          # -> thesis_figures/data/example_day.npz

Regenerates the robot observations of the example day with the evaluation's seed and
stores: the fraction of walkable cells observed at every frame, how often each cell is
observed over the day, and, for every unobserved walkable cell-frame, how many seconds
ago that cell was last observed (cells not yet observed that day are left out).
"""
from __future__ import annotations

import os

import h5py
import numpy as np

DAY = "atc-20130811"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "example_day.npz")


def main() -> None:
    from crowdcore import config
    config.CFG["data"]["root"] = "/scratch/work/zhangx29/data"
    from crowdcore import navigation, observation_model as om
    from methods.senseiver import dataset as senseiver_ds

    fp = next(p for p in om.split_files("test") if DAY in p)
    X = np.asarray(om.load_state(fp)[0], dtype=np.float32)
    with h5py.File(fp, "r") as f:
        time = np.asarray(f["time"])
    walk = navigation.build_valid_mask_from_config(X).astype(bool)
    cfg = senseiver_ds.obs_config()
    obs = om.generate_observations(
        X, sensing_range=cfg["sensing_range"], num_agents=cfg["num_agents"],
        add_noise=cfg["add_noise"], seed=om.day_seed(fp), valid_mask=walk,
        obs_every_k=cfg["obs_every_k"])
    om_ = obs["Omega"].astype(bool)                                 # (T, H, W)
    T = len(om_)

    coverage = om_[:, walk].mean(axis=1)                            # per frame
    frequency = om_.mean(axis=0)                                    # per cell
    last = np.full(walk.shape, -1, dtype=np.int64)
    ages = np.zeros(4096, dtype=np.int64)                           # histogram, seconds
    for t in range(T):
        blind = walk & ~om_[t] & (last >= 0)
        a = t - last[blind]
        ages += np.bincount(np.minimum(a, len(ages) - 1), minlength=len(ages))
        last[om_[t]] = t
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, day=DAY, time=time, coverage=coverage.astype(np.float32),
                        frequency=frequency.astype(np.float32), walkable=walk, age_hist=ages)
    cdf = np.cumsum(ages) / ages.sum()
    print(f"[day] {OUT}: {T} frames, mean coverage {coverage.mean():.3f}; "
          f"blind cells seen within 1 s {cdf[1]:.2f}, 15 s {cdf[15]:.2f}, 200 s {cdf[200]:.2f}; "
          f"median age {int(np.searchsorted(cdf, 0.5))} s")


if __name__ == "__main__":
    main()
