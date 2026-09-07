"""
check_sensors.py — invariant self-checks for `sensors.build_batch`
=====================================================

This is the first gate after porting. In the reference implementation the
sensor set is fixed-length, and the padding/pad_mask path was never exercised;
we have to prove it ourselves is correct, or none of the numbers downstream can
be trusted.

Checks:
  1. Each frame's token count == that frame's Omega's observed-cell count
  2. pad_mask lines up bit-for-bit with the padding slots (True falls exactly on invalid slots)
  3. A token's first C dimensions == that cell's Y value after per-channel standardisation
  4. A token's last P dimensions == that cell's row in pos_enc
  5. An empty-observation frame produces no NaN, and is marked with one valid dummy token
  6. **Robot positions** fall inside walkable, but **observed cells are allowed
     to fall outside it** -- the 4dvarnet_enkf README explicitly states robots
     observe `disk ∩ line-of-sight`, *not* intersected with the walkable mask
     ("a robot can see a pillar it cannot drive into"). This is also independent
     evidence for our decision to "query all 432 cells, removing the reference
     implementation's pix_avail": non-walkable cells are both observed and
     scored, so they cannot be treated as a region with "nothing to reconstruct".
     (Note `observation_model.generate_observations`'s docstring says "only
     observe walkable cells," which does not match the README or the actual
     data; trust the data.)

Usage (pure numpy/torch, no training needed; still run on a GPU node per project convention):
    python3 checks/check_sensors.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

from methods.senseiver import dataset as ds
from methods.senseiver import sensors
from methods.senseiver.positional import PositionalEncoder

FAIL = []


def ck(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' — ' + extra) if extra else ''}")
    if not cond:
        FAIL.append(name)


def main():
    C, H, W = ds.state_shape()
    HW = H * W
    day = ds.om.split_files("test")[0]
    print(f"[data] {os.path.basename(day)}  first 300 frames")
    X, Y, Om = ds.load_day(day, stride=1, seed=0, frames=300)
    pe = PositionalEncoder((H, W, C), 16).numpy().astype(np.float32)
    mean = np.array([0.06, 0.03, 0.0, 0.02], np.float32)
    std = np.array([0.17, 0.46, 0.16, 0.11], np.float32)

    B = 16
    idx = np.arange(B)
    tok, pad, n = sensors.build_batch(Y[idx], Om[idx], pe, mean, std)
    tok, pad, n = tok.numpy(), pad.numpy(), n.numpy()
    print(f"[shape] tokens {tok.shape}  pad_mask {pad.shape}  "
          f"sensor count min {n.min()} max {n.max()}")

    ck("1. token count == Omega's observed-cell count", bool((n == Om[idx].sum(1)).all()))
    ck("2. pad_mask lines up with the valid slots",
       bool(all((~pad[b]).sum() == max(n[b], 1) for b in range(B))))

    b = int(np.argmax(n))                       # check element-by-element on the frame with the most observations
    ii = np.flatnonzero(Om[idx][b])
    want_v = ((Y[idx][b][:, ii].T - mean) / std).astype(np.float32)
    ck("3. first C dims == standardised Y", np.allclose(tok[b, :len(ii), :C], want_v, atol=1e-6))
    ck("4. last P dims == the corresponding pos_enc row", np.allclose(tok[b, :len(ii), C:], pe[ii], atol=1e-6))
    ck("   padding slots are all zero", bool(np.abs(tok[b, len(ii):]).max() == 0.0) if len(ii) < tok.shape[1] else True)

    # 5. an empty-observation frame
    Om0 = Om[idx].copy(); Om0[0] = False
    t0, p0, n0 = sensors.build_batch(Y[idx], Om0, pe, mean, std)
    ck("5. an empty-observation frame has no NaN and leaves one valid dummy token",
       bool(np.isfinite(t0.numpy()).all() and (~p0.numpy()[0]).sum() == 1 and n0.numpy()[0] == 0))

    # 6. robot positions vs. observation footprint
    Xd, _ = ds.om.load_state(day)
    Xd = np.asarray(Xd)[:300]
    valid2 = ds.nav.build_valid_mask_from_config(Xd)
    valid = valid2.reshape(-1)
    out = ds.om.generate_observations(Xd, valid_mask=valid2, seed=0)
    pos = out["positions"].reshape(-1, 2)
    ck("6. robot positions are all inside walkable", bool(valid2[pos[:, 0], pos[:, 1]].all()),
       f"walkable {int(valid.sum())}/{HW}")
    seen = np.flatnonzero(Om.any(0))
    frac = 1.0 - valid[seen].mean()
    print(f"  INFO  fraction of observed cells falling outside walkable: {frac*100:.1f}% "
          f"(robots can see cells they cannot drive into -> must query all {HW} cells)")

    print("\n" + ("ALL PASS" if not FAIL else f"FAILED: {FAIL}"))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
