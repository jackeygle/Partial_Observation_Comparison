"""
check_encoding.py — self-check for the information-form encoding (paper version)
=========================================================

This is a module self-test for `encoding.py`, not an "experiment". Every check
here is an **invariant** -- any of them being false means the encoding was
written wrong, and this class of bug **raises no error, it just makes results
worse**, so by the time training looks bad it is impossible to tell whether it's
the model or an encoding bug.

Checks:
  1. Shapes, no non-finite values
  2. Both slices are 0 at missing cells (the paper's core semantics: missing = zero precision)
  3. The second slice = 1 where observed (sigma^2_obs is taken as the constant 1, see encoding.py)
  4. The first slice = observed value minus the per-cell mean field (residual) where observed
  5. Target: precision is 0 at each channel's "undefined" cells (density is
     always defined; vx/vy need density>0; var needs vel_var>0), and 0 at
     non-walkable cells
  6. Reference quantities: observation coverage, and the fraction "observed at
     least once within a 3-frame window" -- the latter directly shows how much
     observational information the vanilla paper's setup can get on ATC

Usage (pure numpy/scipy, the login node is fine):
    python3 check_encoding.py
"""
from __future__ import annotations

import os
import sys

import h5py
import numpy as np

# Scripts in checks/ import source modules from the project root; the root must
# be inserted at the front (so this directory's losses.py takes priority),
# 4dvarnet_enkf can only be appended (it also has a losses.py, and inserting it
# at the front would shadow this directory's).
from crowdcore import observation_model as om                                    # noqa: E402

from methods.dincae.state import (CHANNEL_TRANSFORM, CHANNELS, StateStats,  # noqa: E402
                         NCH, channel_valid, fwd_channel)
from methods.dincae.dataset import encode_day, obs_config                         # noqa: E402
from methods.dincae.encoding import (FRESH_OFFSETS, N_IN, N_STATIC, N_TGT,        # noqa: E402
                      encode_target, observed_pair)


def main():
    stats = StateStats()
    print(f"per-cell mean field: {stats.n_train_days} days, walkable {stats.valid.sum()}/{stats.valid.size}")
    print(f"channels {CHANNELS};  N_IN={N_IN}  N_TGT={N_TGT}  ntime_win={len(FRESH_OFFSETS)}")
    print(f"per-channel transform {CHANNEL_TRANSFORM}")

    fp = om.split_files("valid")[0]                    # a held-out day
    with h5py.File(fp, "r") as f:
        X, t_unix = f["grid"][:6000], f["time"][:6000]
    print(f"dev day {os.path.basename(fp)}  T={X.shape[0]}")

    oc = obs_config()
    obs = om.generate_observations(
        X, sensing_range=oc["sensing_range"], num_agents=oc["num_agents"],
        add_noise=oc["add_noise"], seed=oc["seed"], valid_mask=stats.valid,
        obs_std=oc["obs_std"], obs_every_k=oc["obs_every_k"])
    Y, M = obs["Y"][:, :NCH], obs["Omega_c"][:, :NCH]
    scaled, invvar = observed_pair(Y, M, stats.mean, stats.std)
    tgt = encode_target(X, stats.mean, stats.std, stats.valid)

    ok = True

    def chk(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'OK  ' if cond else 'FAIL'} {name}")

    print("\n=== Invariants ===")
    for nm, arr in (("scaled", scaled), ("invvar", invvar), ("target", tgt)):
        chk(f"{nm} {arr.shape} all finite", np.isfinite(arr).all())

    chk("scaled==0 where missing", (scaled[~M] == 0).all())
    chk("invvar==0 where missing", (invvar[~M] == 0).all())
    chk("invvar==1 where observed", (invvar[M] == 1).all())
    resid = np.stack([(fwd_channel(Y[:, c].astype(np.float64), c) - stats.mean[c][None])
                     / stats.std[c] for c in range(NCH)], axis=1)
    chk("scaled == normalised(fwd(observation) - per-cell mean field) where observed",
        np.allclose(scaled[M], resid[M], atol=1e-5))

    cv = channel_valid(X)
    chk("target: precision==0 where each channel is undefined",
        all((tgt[:, 2 * c + 1][~cv[c]] == 0).all() for c in range(NCH)))
    nw = ~np.broadcast_to(stats.valid[None], cv[0].shape)
    chk("target: precision==0 on non-walkable cells",
        all((tgt[:, 2 * c + 1][nw] == 0).all() for c in range(NCH)))
    chk("target: first slice == normalised(fwd(truth) - per-cell mean field) where defined",
        all(np.allclose(tgt[:, 2 * c][cv[c] & ~nw],
                        ((fwd_channel(X[:, c].astype(np.float64), c) - stats.mean[c][None])
                         / stats.std[c])[cv[c] & ~nw], atol=1e-5)
            for c in range(NCH)))

    # End to end: encode_day's channel assembly
    xin, ytg = encode_day(fp, stats, stride=97, max_frames=6000)
    chk(f"encode_day shape {xin.shape} / {ytg.shape}",
        xin.shape[1] == N_IN and ytg.shape[1] == N_TGT)
    chk("encode_day has no non-finite values", np.isfinite(xin).all() and np.isfinite(ytg).all())
    o = N_STATIC + NCH                                  # the invvar block for the first dt
    chk("the assembled invvar block only contains 0/1",
        np.isin(xin[:, o:o + NCH], (0.0, 1.0)).all())

    print("\n=== Reference quantities (how much the vanilla paper's setup can observe on ATC) ===")
    Om = obs["Omega"]
    v = stats.valid
    print(f"  single-frame coverage of walkable cells: {100 * Om[:, v].mean():.1f}%")
    win = np.zeros_like(Om[0], dtype=bool)
    seen = []
    for t in range(1, Om.shape[0] - 1):
        win = Om[t - 1] | Om[t] | Om[t + 1]
        seen.append(win[v].mean())
    print(f"  observed at least once within a 3-frame window: {100 * np.mean(seen):.1f}%"
          f"   -> the remaining {100 - 100 * np.mean(seen):.1f}% of cells have both slices at 0, "
          "relying only on coordinates + clock + the per-cell mean field")
    for c in range(NCH):
        print(f"  target-defined fraction {CHANNELS[c]:8s}: "
              f"{100 * (cv[c] & ~nw).mean():.1f}% of (frame x cell)")

    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
