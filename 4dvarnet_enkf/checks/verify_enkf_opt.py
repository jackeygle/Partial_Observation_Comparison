"""
verify_enkf_opt.py  —  enforce enkf_opt/README.md's rule, and measure what the fast path costs
==============================================================================================

enkf_opt/ is only allowed to differ from enkf_lab/ (the pristine copy of the original
Partial_observation filter) in ways that do not change the numbers. This script is the check:
it drives BOTH copies over the same real frames, same seed, same observations, and requires

    np.array_equal(lab_output, opt_output)          # bit-identical, not "close"

for the default gain mode. It then runs enkf_opt again with gain_mode="ensemble" — the
Woodbury rewrite, which is the same gain computed in a different order and therefore NOT
bit-identical — and reports the actual deviation and speedup so the trade is a measured number
rather than an assumption.

Checked per frame: the analysis mean returned by step(), the ensemble spread, the full ensemble
X, and the internal bias_estimate state (it feeds the next forecast, so a difference there
would compound silently).

Also unit-checks the pieces that were rewritten, against the reference implementations kept in
enkf_opt for exactly this purpose:
    _localization_matrix       vs _localization_matrix_ref
    _obs_indices              vs C.nonzero()[1]
    _backproject_obs_bias     vs the original Python loop  (incl. repeated indices)
    GeneratePartialObs        vs the original double loops

Run (from Thesis_Project/4dvarnet_enkf):
    python3 checks/verify_enkf_opt.py --frames 12 --ensemble 100
"""
from __future__ import annotations
import argparse
import glob
import importlib
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H, W, F = 36, 12, 4
TOTAL, STATE_DIM = H * W, F * H * W
PROC_STD = (0.02829307, 0.31263075, 0.12325809, 0.41680932)
INIT_STD = (0.2290, 1.2660, 0.3429, 0.0259)
SRC = {"lab": os.path.join(ROOT, "enkf_lab"), "opt": os.path.join(ROOT, "enkf_opt")}


def load_copy(which):
    """Import pedpred.ENKF from one of the copies, evicting any previously imported one."""
    for mod in [k for k in list(sys.modules) if k.startswith(("pedpred", "tools"))]:
        del sys.modules[mod]
    root = SRC[which]
    sys.path.insert(0, root)
    try:
        enkf = importlib.import_module("pedpred.ENKF")
        utils = importlib.import_module("pedpred.utils")
    finally:
        sys.path.remove(root)
    return enkf, utils, root


def build_C(cells):
    """Same C the drivers build: one row per (observed cell, feature), a single 1.0 each."""
    C = np.zeros((F * len(cells), STATE_DIM))
    idx = np.empty(F * len(cells), dtype=np.int64)
    for i, (r, c) in enumerate(cells):
        for f in range(F):
            col = f * TOTAL + r * W + c
            C[i * F + f, col] = 1.0
            idx[i * F + f] = col
    return C, idx


def frames(npz, n):
    z = np.load(npz)
    Y, Om, obs_std = z["Y"], z["Omega"], z["obs_std"]
    out = []
    for t in range(min(n, Y.shape[0])):
        cells = list(zip(*np.where(Om[t])))
        if not cells:
            out.append((None, None, None))
            continue
        C, idx = build_C(cells)
        out.append((C, C @ Y[t].reshape(-1), idx))
    return out, obs_std


def run(enkf_mod, model, pre, obs_std, ensemble, radius, gain_mode=None, pass_obs_idx=False):
    """One full pass; returns per-frame estimates + spreads + ensembles + bias state."""
    kw = {} if gain_mode is None else {"gain_mode": gain_mode}
    f = enkf_mod.LocalizedEnsembleKalmanFilter(
        grid_size=(H, W), state_shape=(F, H, W), ensemble_size=ensemble,
        proc_noise_std=PROC_STD, obs_noise_std=tuple(float(s) for s in obs_std),
        init_perturb_std=INIT_STD, inflation=1.02, localization_radius=radius, **kw)
    rng = np.random.RandomState(0)
    X0 = np.zeros((ensemble, STATE_DIM))
    for i, s in enumerate(PROC_STD):
        X0[:, i * TOTAL:(i + 1) * TOTAL] = rng.normal(0, s, size=(ensemble, TOTAL))
    f.X = X0

    est, spread, ens, bias, n_bad = [], [], [], [], 0
    t0 = time.perf_counter()
    for C, y, idx in pre:
        ok = False
        if C is not None:
            try:
                step_kw = {"obs_idx": idx} if pass_obs_idx else {}
                e = f.step(C, y, model=model, **step_kw)
                ok = True
            except np.linalg.LinAlgError:
                n_bad += 1
        if not ok:
            f.X = f._clip_bounds(f.forecast(model))
            e = f.X.mean(axis=0).reshape(F, H, W)
        est.append(e)
        spread.append(f.get_std())
        ens.append(f.X.copy())
        bias.append(f.bias_estimate.copy())
    wall = time.perf_counter() - t0
    return dict(est=np.array(est), spread=np.array(spread), ens=np.array(ens),
                bias=np.array(bias), n_bad=n_bad, wall=wall)


def compare(a, b, label, strict):
    """Report equality of two passes. strict=True demands bit-identity."""
    ok = True
    for key in ("est", "spread", "ens", "bias"):
        x, y = a[key], b[key]
        same = x.shape == y.shape and np.array_equal(x, y)
        if strict:
            print(f"  {label:<34} {key:<7} bit-identical: {same}")
            ok &= same
        else:
            d = np.abs(x.astype(np.float64) - y.astype(np.float64))
            scale = np.abs(x).max() or 1.0
            print(f"  {label:<34} {key:<7} max|d|={d.max():.3e}  "
                  f"rel={d.max()/scale:.3e}  identical={same}")
    if a["n_bad"] != b["n_bad"]:
        print(f"  {label:<34} !! analysis-skip count differs: {a['n_bad']} vs {b['n_bad']}")
        ok = False
    return ok


def unit_checks(enkf_mod):
    """The rewritten helpers against the reference implementations."""
    ok = True
    f = enkf_mod.LocalizedEnsembleKalmanFilter(
        grid_size=(H, W), state_shape=(F, H, W), ensemble_size=8,
        obs_noise_std=(0.1, 0.2, 0.3, 0.4), localization_radius=7)

    rng = np.random.RandomState(1)
    # Real cells, plus the off-grid "rows" the observed_cells fallback in update() produces,
    # plus the empty case.
    for cells in (np.stack([rng.randint(0, H, 40), rng.randint(0, W, 40)], 1),
                  np.stack([rng.randint(0, 4 * H, 60), rng.randint(0, W, 60)], 1),
                  np.zeros((0, 2), dtype=int)):
        got = f._localization_matrix([tuple(c) for c in cells])
        ref = f._localization_matrix_ref([tuple(c) for c in cells])
        same = got.shape == ref.shape and np.array_equal(got, ref)
        print(f"  _localization_matrix  n={len(cells):<3} bit-identical: {same}")
        ok &= same

    cells = list(zip(rng.randint(0, H, 30).tolist(), rng.randint(0, W, 30).tolist()))
    C, idx = build_C(cells)
    got, is_sel = enkf_mod._obs_indices(C)
    same = np.array_equal(got, C.nonzero()[1]) and np.array_equal(got, idx) and is_sel
    print(f"  _obs_indices                     identical: {same}")
    ok &= same
    # a non-selection C must fall back to nonzero()
    C2 = C.copy(); C2[0, 5] = 0.5
    got2, is_sel2 = enkf_mod._obs_indices(C2)
    same = np.array_equal(got2, C2.nonzero()[1]) and not is_sel2
    print(f"  _obs_indices (non-selection C)   identical: {same}")
    ok &= same

    for oi in (idx, np.concatenate([idx, idx[:7]])):        # incl. repeated indices
        bias = rng.normal(size=len(oi))
        ref = np.zeros(STATE_DIM)
        for k, s in enumerate(oi):
            ref[s] = bias[k]
        same = np.array_equal(f._backproject_obs_bias(oi, bias), ref)
        print(f"  _backproject_obs_bias  m={len(oi):<4}     identical: {same}")
        ok &= same

    for pos in ((0, 0), (18, 6), (35, 11)):
        for rad in (2, 5, 7):
            g = enkf_mod.GeneratePartialObs((H, W), (F, H, W), pos, rad)
            cells = g.cells_within_range()
            ref_cells = [(r, c) for r in range(H) for c in range(W)
                         if np.sqrt((r - pos[0]) ** 2 + (c - pos[1]) ** 2) <= rad]
            Cg, _ = g.get_observation_matrix()
            ref_C = np.zeros((F * len(ref_cells), STATE_DIM))
            for i, (r, c) in enumerate(ref_cells):
                for ff in range(F):
                    ref_C[i * F + ff, ff * TOTAL + r * W + c] = 1.0
            same = cells == ref_cells and np.array_equal(Cg, ref_C)
            obs = rng.normal(size=ref_C.shape[0])
            rec = g.reconstruct_observation(obs, ref_C)
            ref_rec = np.full((F, H, W), np.nan)
            for k, s in enumerate(ref_C.nonzero()[1]):
                ref_rec[s // TOTAL, (s % TOTAL) // W, s % W] = obs[k]
            same &= np.array_equal(rec, ref_rec, equal_nan=True)
            ok &= same
            if not same:
                print(f"  GeneratePartialObs pos={pos} r={rad}   MISMATCH")
    print(f"  GeneratePartialObs (9 configs)   identical: {ok}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--ensemble", type=int, default=100)
    ap.add_argument("--radius", type=int, default=7)
    ap.add_argument("--day", default="atc-20130811")
    args = ap.parse_args()

    npz = os.path.join(ROOT, "check_outputs", "enkf", f"obs_{args.day}.npz")
    if not os.path.exists(npz):
        npz = sorted(glob.glob(os.path.join(ROOT, "check_outputs", "enkf", "obs_*.npz")))[0]
    pre, obs_std = frames(npz, args.frames)
    n_obs = sum(c is not None for c, _, _ in pre)
    m = [c.shape[0] for c, _, _ in pre if c is not None]
    print(f"[data] {os.path.basename(npz)}  {len(pre)} frames, {n_obs} with observations, "
          f"m in [{min(m, default=0)}, {max(m, default=0)}]  ensemble={args.ensemble}")

    results = {}
    for which in ("lab", "opt"):
        mod, utils, root = load_copy(which)
        model = utils.load_model(os.path.join(root, "apt-ibex_train_model_28D.pth"),
                                 torch.device("cpu"))
        if which == "opt":
            print("\n[unit] rewritten helpers vs reference implementations")
            unit_ok = unit_checks(mod)
            results["opt_fast"] = run(mod, model, pre, obs_std, args.ensemble, args.radius,
                                     gain_mode="ensemble")
            results["opt_idx"] = run(mod, model, pre, obs_std, args.ensemble, args.radius,
                                     gain_mode="pinv", pass_obs_idx=True)
        results[which] = run(mod, model, pre, obs_std, args.ensemble, args.radius)

    print("\n[strict] enkf_opt (gain_mode=pinv) vs enkf_lab — must be bit-identical")
    ok = compare(results["lab"], results["opt"], "opt vs lab", strict=True)
    ok &= compare(results["lab"], results["opt_idx"], "opt(obs_idx given) vs lab", strict=True)

    print("\n[fast] enkf_opt gain_mode=ensemble vs enkf_lab — same gain, different rounding")
    compare(results["lab"], results["opt_fast"], "opt(ensemble) vs lab", strict=False)

    print("\n[speed] same frames, same node, one process")
    base = results["lab"]["wall"]
    for k in ("lab", "opt", "opt_idx", "opt_fast"):
        w = results[k]["wall"]
        print(f"  {k:<9} {w:7.2f}s  {w/len(pre)*1000:7.1f} ms/frame   x{base/w:5.2f} vs lab")

    print(f"\n[verdict] {'PASS' if (ok and unit_ok) else 'FAIL'} — "
          f"strict path bit-identical: {ok}, unit checks: {unit_ok}")
    return 0 if (ok and unit_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
