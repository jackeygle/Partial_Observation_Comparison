"""
rederive_obs_std.py  —  reproduce config.yaml's observation.obs_std from the data
=================================================================================

The supervisor asked: where do the 4 numbers in observation_std come from? This
script re-derives them from scratch on our grid_cache, replicating the original
project's calibration (Partial_observation/inspect_data.py):

  For each channel, over ACTIVE cells (density > 1e-6) of the test split:
      robust_std = 1.4826 * MAD             (MAD = median(|x - median(x)|))
      obs_std    = obs_factor * robust_std  (obs_factor = 0.25, an ASSUMPTION:
                                             "sensor is usually more accurate than
                                             the natural variability")

If our numbers match config.yaml's [0.0569, 0.3147, 0.0862, 0.0064], we have
reproduced them from first principles and understand exactly what they are.

Run:
    module load scicomp-pytorch-env/2026.1
    python3 checks/rederive_obs_std.py
"""

from __future__ import annotations

import os
import sys


import numpy as np

from crowdcore import config
from crowdcore import observation_model as om

DENS_THR = 1e-6          # "active cell" threshold (inspect_data.py)
OBS_FACTOR = 0.25        # obs_std = 0.25 * robust_std (the assumption)
CHANNELS = ("density", "vx", "vy", "var")


def robust_std(values):
    """1.4826 * MAD — a Gaussian-consistent, outlier-robust std estimate."""
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return 1.4826 * mad


def main():
    files = om.split_files("test")                        # same role as inspect_data's test split
    print(f"[data] {len(files)} test-split days")

    # gather all active-cell values per channel across the whole split
    pools = {c: [] for c in CHANNELS}
    for fp in files:
        X, _ = om.load_state(fp)                           # (T,4,H,W)
        active = X[:, 0] > DENS_THR                        # (T,H,W) cells with people
        for ci, name in enumerate(CHANNELS):
            pools[name].append(X[:, ci][active])           # values on active cells
    pools = {c: np.concatenate(v) for c, v in pools.items()}

    print(f"\n{'channel':8s} {'n_active':>12s} {'median':>10s} {'robust_std':>11s} "
          f"{'obs_std(=0.25x)':>15s} {'config':>10s} {'match':>7s}")
    cfg = np.asarray(config.get("observation", "obs_std"), dtype=float)
    derived = []
    for ci, name in enumerate(CHANNELS):
        vals = pools[name]
        rstd = robust_std(vals)
        ostd = OBS_FACTOR * rstd
        derived.append(ostd)
        rel = abs(ostd - cfg[ci]) / max(cfg[ci], 1e-12)
        print(f"{name:8s} {len(vals):>12d} {np.median(vals):>10.4f} {rstd:>11.4f} "
              f"{ostd:>15.4f} {cfg[ci]:>10.4f} {rel:>6.1%}")

    derived = np.asarray(derived)
    # save the derivation so figures/slides READ these numbers instead of hard-coding them
    import json
    rows = []
    for ci, name in enumerate(CHANNELS):
        vals = pools[name]; med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        rows.append({"channel": name, "median": med, "mad": mad,
                     "robust_std": 1.4826 * mad, "obs_std_derived": OBS_FACTOR * 1.4826 * mad,
                     "config": float(cfg[ci])})
    outp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "check_outputs", "eval", "obs_std_derivation.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump({"obs_factor": OBS_FACTOR, "mad_to_std": 1.4826, "channels": rows}, open(outp, "w"), indent=2)
    print(f"\n[out] {outp}")
    print(f"re-derived obs_std : {np.array2string(derived, precision=4)}")
    print(f"config obs_std     : {np.array2string(cfg, precision=4)}")
    close = np.allclose(derived, cfg, rtol=0.15)
    print(f"\n=> {'REPRODUCED (within 15%): the 4 values ARE 0.25 x 1.4826 x MAD on active cells' if close else 'DIFFERS — likely a different test split/day set than the original calibration'}")
    print("   The 0.25 factor is an ASSUMPTION (sensor noise ~ 1/4 of the field's natural"
          " variability), NOT a measured sensor spec.")


if __name__ == "__main__":
    main()
