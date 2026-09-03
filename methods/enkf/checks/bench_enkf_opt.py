"""
bench_enkf_opt.py  —  what the enkf_opt optimisation rounds are actually worth
=============================================================================

Times the EnKF variants on the SAME frames, in ONE process, on ONE node, interleaved
(ABC ABC ABC, not AAA BBB CCC) so that drift in the node's background load spreads over every
variant instead of landing on one of them and reading as a difference between variants. Same
reasoning as checks/bench_speed.py, which is the method-vs-method (EnKF vs 4DVarNet) benchmark;
this one is implementation-vs-implementation within the EnKF.

Variants:
    lab             enkf_lab, the untouched original
    prev            round 1 only: _localization_matrix vectorised. Not a directory in the repo
                    (enkf_opt has moved on), so it is reconstructed on demand with --make-prev
                    from enkf_lab by applying exactly that one patch.
    opt_pinv        current enkf_opt, gain_mode="pinv"      — verified bit-identical to lab
    opt_ensemble    current enkf_opt, gain_mode="ensemble"  — Woodbury gain, same gain, faster

Correctness is NOT this script's job: checks/verify_enkf_opt.py proves bit-identity of the
default path and checks/verify_enkf_gain_mode.py measures what the fast path costs numerically.

Run (from Thesis_Project/4dvarnet_enkf), on a node, not the login node:
    srun -p batch-csl -c 16 --mem=24G -t 1:00:00 python3 -u checks/bench_enkf_opt.py \
        --frames 200 --repeats 2 --make-prev /tmp/enkf_prev
"""
from __future__ import annotations
import argparse
import importlib
import json
import os
import resource
import shutil
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H, W, F = 36, 12, 4
TOTAL, STATE_DIM = H * W, F * H * W
PROC_STD = (0.02829307, 0.31263075, 0.12325809, 0.41680932)
INIT_STD = (0.2290, 1.2660, 0.3429, 0.0259)

# The round-1 patch, verbatim: the four-deep Python loop replaced by its vectorised equivalent.
PREV_OLD = '''    def _localization_matrix(self, obs_cells):
        """
        Create a diagonal localization matrix for observations.
        Only cells within `self.localization_radius` will influence updates.
        """
        H, W = self.grid_size
        num_obs = len(obs_cells)
        loc_matrix = np.zeros((num_obs, self.state_dim))

        for i, (r_obs, c_obs) in enumerate(obs_cells):
            for f in range(self.num_features):
                for r in range(H):
                    for c in range(W):
                        # Distance-based weight
                        dist = np.sqrt((r - r_obs)**2 + (c - c_obs)**2)
                        if dist <= self.localization_radius:
                            col_idx = f * H * W + r * W + c
                            loc_matrix[i, col_idx] = np.exp(-(dist**2)/(2*(self.localization_radius**2)))
        return loc_matrix
'''
PREV_NEW = '''    def _localization_matrix(self, obs_cells):
        """Round-1 vectorisation, reconstructed by checks/bench_enkf_opt.py --make-prev."""
        H, W = self.grid_size
        num_obs = len(obs_cells)
        if num_obs == 0:
            return np.zeros((0, self.state_dim))
        obs = np.asarray(obs_cells)
        dr = np.arange(H)[None, :, None] - obs[:, 0, None, None]
        dc = np.arange(W)[None, None, :] - obs[:, 1, None, None]
        dist = np.sqrt(dr ** 2 + dc ** 2)
        w = np.where(dist <= self.localization_radius,
                     np.exp(-(dist ** 2) / (2 * (self.localization_radius ** 2))), 0.0)
        return np.tile(w.reshape(num_obs, H * W), (1, self.num_features))
'''


def make_prev(dest):
    """Build the round-1 copy from enkf_lab, so the round-1 -> round-2 claim is reproducible."""
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.copytree(os.path.join(ROOT, "enkf_lab"), dest, symlinks=True)
    for dirpath, dirnames, filenames in os.walk(dest):
        for name in dirnames + filenames:
            p = os.path.join(dirpath, name)
            if not os.path.islink(p):
                os.chmod(p, os.stat(p).st_mode | 0o200)
    path = os.path.join(dest, "pedpred", "ENKF.py")
    src = open(path).read()
    if src.count(PREV_OLD) != 1:
        raise SystemExit(f"enkf_lab/pedpred/ENKF.py no longer matches the round-1 patch anchor")
    open(path, "w").write(src.replace(PREV_OLD, PREV_NEW))
    return dest


def build_C(cells):
    C = np.zeros((F * len(cells), STATE_DIM))
    idx = np.empty(F * len(cells), dtype=np.int64)
    for i, (r, c) in enumerate(cells):
        for f in range(F):
            col = f * TOTAL + r * W + c
            C[i * F + f, col] = 1.0
            idx[i * F + f] = col
    return C, idx


def load(root):
    """Import pedpred.ENKF from one copy, evicting whichever one is currently imported."""
    for mod in [k for k in list(sys.modules) if k.startswith(("pedpred", "tools"))]:
        del sys.modules[mod]
    sys.path.insert(0, root)
    try:
        E = importlib.import_module("pedpred.ENKF")
        U = importlib.import_module("pedpred.utils")
    finally:
        sys.path.remove(root)
    return E, U.load_model(os.path.join(root, "apt-ibex_train_model_28D.pth"),
                           torch.device("cpu"))


def hw():
    cpu = "CPU"
    try:
        for l in open("/proc/cpuinfo"):
            if l.startswith("model name"):
                cpu = l.split(":", 1)[1].strip(); break
    except Exception:
        pass
    return {"cpu": cpu, "cores": len(os.sched_getaffinity(0)),
            "node": os.environ.get("SLURMD_NODENAME", ""),
            "torch_threads": torch.get_num_threads()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--ensemble", type=int, default=100)
    ap.add_argument("--radius", type=int, default=7)
    ap.add_argument("--day", default="atc-20130811")
    ap.add_argument("--make-prev", default="", metavar="DIR",
                    help="reconstruct the round-1 copy here and include it (slow: the untouched "
                         "and round-1 variants dominate the runtime)")
    ap.add_argument("--skip-lab", action="store_true",
                    help="leave out the untouched original (it is ~2.6 s/frame)")
    ap.add_argument("--out", default="check_outputs/eval/bench_enkf_opt.json")
    args = ap.parse_args()

    z = np.load(os.path.join(ROOT, "check_outputs", "enkf", f"obs_{args.day}.npz"))
    Y, Om, obs_std = z["Y"], z["Omega"], z["obs_std"]
    T = min(args.frames, Y.shape[0])
    pre = []
    for t in range(T):
        cells = list(zip(*np.where(Om[t])))
        if not cells:
            pre.append((None, None, None)); continue
        C, idx = build_C(cells)
        pre.append((C, C @ Y[t].reshape(-1), idx))
    ms = [c.shape[0] for c, _, _ in pre if c is not None]
    info = hw()
    print(f"[data] {args.day}: {T} frames, {len(ms)} with observations, m in "
          f"[{min(ms)}, {max(ms)}] mean {np.mean(ms):.0f}; ensemble={args.ensemble} "
          f"radius={args.radius}", flush=True)
    print(f"[hw]   {info['cpu']} · {info['cores']} cores · node {info['node']} · "
          f"{info['torch_threads']} torch threads", flush=True)

    def runner(E, model, mode):
        kw = {} if mode is None else {"gain_mode": mode}

        def once():
            f = E.LocalizedEnsembleKalmanFilter(
                grid_size=(H, W), state_shape=(F, H, W), ensemble_size=args.ensemble,
                proc_noise_std=PROC_STD, obs_noise_std=tuple(float(s) for s in obs_std),
                init_perturb_std=INIT_STD, inflation=1.02,
                localization_radius=args.radius, **kw)
            rng = np.random.RandomState(0)
            X0 = np.zeros((args.ensemble, STATE_DIM))
            for i, s in enumerate(PROC_STD):
                X0[:, i * TOTAL:(i + 1) * TOTAL] = rng.normal(0, s, size=(args.ensemble, TOTAL))
            f.X = X0
            for C, y, _ in pre:
                if C is not None:
                    try:
                        f.step(C, y, model=model); continue
                    except np.linalg.LinAlgError:
                        pass
                f.X = f._clip_bounds(f.forecast(model))
        return once

    variants = []
    if not args.skip_lab:
        E, m = load(os.path.join(ROOT, "enkf_lab"))
        variants.append(("lab (untouched original)", runner(E, m, None)))
    if args.make_prev:
        E, m = load(make_prev(args.make_prev))
        variants.append(("prev (round 1: localization)", runner(E, m, None)))
    E, m = load(os.path.join(ROOT, "enkf_opt"))
    variants.append(("opt  gain_mode=pinv", runner(E, m, "pinv")))
    variants.append(("opt  gain_mode=ensemble", runner(E, m, "ensemble")))

    for _, fn in variants:                       # warm up every variant before any timed repeat
        fn()
    walls = {n: [] for n, _ in variants}
    cpus = {n: [] for n, _ in variants}
    for rep in range(args.repeats):              # interleaved
        for name, fn in variants:
            r0 = resource.getrusage(resource.RUSAGE_SELF)
            t0 = time.perf_counter(); fn(); w = time.perf_counter() - t0
            r1 = resource.getrusage(resource.RUSAGE_SELF)
            walls[name].append(w)
            cpus[name].append((r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime))
            print(f"  rep {rep+1}: {name:<30} {w:8.2f}s", flush=True)

    slowest = max(np.mean(walls[n]) for n, _ in variants)
    print(f"\n{'variant':<30} {'wall':>9} {'+-':>7} {'ms/frame':>9} {'cores':>6} {'speedup':>8}")
    rows = {}
    for name, _ in variants:
        w, c = np.array(walls[name]), np.array(cpus[name])
        rows[name] = {"wall_mean": float(w.mean()), "wall_std": float(w.std()),
                      "wall_all": w.tolist(), "per_frame_ms": float(w.mean() / T * 1000),
                      # CPU-seconds / wall-seconds: 1.0 = one core busy throughout
                      "cores_used": float((c / w).mean()),
                      "speedup_vs_slowest": float(slowest / w.mean())}
        r = rows[name]
        print(f"{name:<30} {r['wall_mean']:8.2f}s {r['wall_std']:6.2f}s "
              f"{r['per_frame_ms']:8.1f} {r['cores_used']:6.2f} "
              f"{r['speedup_vs_slowest']:7.2f}x")

    path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"hw": info, "day": args.day, "frames": int(T), "frames_with_obs": len(ms),
               "m_mean": float(np.mean(ms)), "ensemble": args.ensemble, "radius": args.radius,
               "repeats": args.repeats, "runs": rows}, open(path, "w"), indent=2)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
