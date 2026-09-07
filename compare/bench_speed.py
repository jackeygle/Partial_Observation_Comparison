"""
bench_speed.py  —  controlled CPU inference-speed benchmark
===========================================================

Times EnKF and 4DVarNet on the SAME frames, on the SAME node, in ONE job.

Why it is built this way
------------------------
Earlier timings were taken from ordinary SLURM jobs and are not usable for a method-vs-method
comparison: every batch-csl node was shared with 3-9 other users' jobs (e.g. csl40 at 40/40
cores allocated), so memory bandwidth and L3 were contended, and two CPU measurements even
landed on different nodes. Numbers were also collected with different core counts.

So this script:
  * runs the measurements SEQUENTIALLY and INTERLEAVED (ABC ABC ABC, not AAA BBB CCC), so a
    drift in the node's background load is spread over every method instead of landing on
    one of them and reading as a difference between methods;
  * warms up once before the timed repeats (lazy init, allocator, first-touch page faults);
  * repeats each measurement and reports mean +- std, so the spread is visible rather than
    assumed away;
  * records the actual hardware, node, thread count and CPU utilisation next to every number,
    because a timing quoted against the wrong machine is worse than no timing.

--exclusive was the original plan and was dropped: no node in batch-csl (0 of 48 idle) or
gpu-h200 was free, and SLURM scheduled the exclusive job a day out. Interleaving plus
repeats is the standard substitute — it does not remove contention, it converts it from a
hidden bias into a reported spread.

How large that spread actually is, measured across four sessions on identical EnKF code:

    csl18   170.0 / 170.3 / 171.4          mean 170.6   spread  0.4%
    csl24   159.0 / 166.0 / 181.3          mean 168.7   spread  5.5%
    csl31   162.4 / 159.0 / 129.8          mean 150.4   spread  9.7%
    csl19   288.3 / 189.5 / 185.1 / 200.8  mean ~215    spread ~20%

So the EnKF's ABSOLUTE per-frame cost is not reproducible: 130 to 290 ms for the same code.
Its k=1 pass runs for minutes and therefore absorbs whatever else lands on the node.
4DVarNet's is stable (3.3-3.8 ms) because each pass is seconds long. What survives is the
RATIO between the methods, measured inside one job on one node with the rounds interleaved:
40-80x in every session. Quote the ratio; treat any single millisecond figure as "what this
node gave that afternoon", and read the per-run spread that summarise() records next to it.

What is measured
----------------
This run is the SAME-CPU comparison: every row below is timed on one node, on the same cores,
over the same frames. The EnKF's original (unvectorised) implementation is not included — it
was measured at 4,651 ms/frame and has been retired; the optimised version is bit-identical
to it on all 7 test days at both k (14 full-day runs, 277,543 frames, Est and Spread
np.array_equal), so it is the one every comparison uses.

Batch size is not a variable here: a sweep on CPU (batch 1..16) moved the per-frame cost by
under 10% and not monotonically, so the filter's inability to batch costs it nothing in this
comparison. On a GPU it would matter — there the same sweep gave 2.9x from batch 1 to 32 —
which is why a GPU row, if requested, is reported apart from the CPU head-to-head.

  enkf_k1     EnKF, every frame observed        }  optimised implementation
  enkf_k4     EnKF, every 4th frame observed     }
  varnet_k1   4DVarNet, the k=1 model (runs/varnet_<--run>_k1, default a2)
  varnet_k4   4DVarNet, the k=4 model (runs/varnet_<--run>_k4)
  (--gpu adds a GPU row for 4DVarNet, reported apart from the four rows above)

k=1 vs k=4 is expected to make no difference to 4DVarNet: the solver runs a fixed number of
unrolled iterations over the whole dense window, so its cost does not depend on how sparse
Omega is. The EnKF's cost DOES — it pays a Kalman update per observed frame. That asymmetry
is why both k are included.

Per-frame numbers are total/n_frames for both methods, which is the honest comparison: EnKF
assimilates frame by frame, 4DVarNet reconstructs the whole window at once, and both end up
producing an estimate for the same frames.

Run:
    python3 checks/bench_speed.py --frames 1000 --repeats 3
"""
from __future__ import annotations
import argparse
import importlib
import json
import os
from crowdcore import paths
import resource
import sys
import time

import h5py
import numpy as np
import torch

# This script used to live under 4dvarnet_enkf/, and ROOT always pointed there
# (runs/, check_outputs/ hang off it). After the 2026-09-03 refactor it moved to
# the top level, where dirname(dirname(__file__)) would become the repo root, and
# every os.path.join(ROOT, ...) would silently point somewhere wrong -- hence the
# explicit binding.
ROOT = paths.method(paths.VARNET)

# after sys.path — this lives in the project root, not in checks/
from crowdcore import config  # noqa: E402
from crowdcore import observation_model as om
H, W, F = 36, 12, 4
TOTAL, STATE_DIM = H * W, F * H * W
PROC_STD = (0.02829307, 0.31263075, 0.12325809, 0.41680932)
INIT_STD = (0.2290, 1.2660, 0.3429, 0.0259)
SRC = {"orig": "/scratch/work/zhangx29/Partial_observation",
       "opt": paths.enkf_vendor("enkf_opt")}


def hw_info():
    cpu = "CPU"
    try:
        for l in open("/proc/cpuinfo"):
            if l.startswith("model name"):
                cpu = l.split(":", 1)[1].strip(); break
    except Exception:
        pass
    return {"cpu": cpu, "cores_avail": len(os.sched_getaffinity(0)),
            "node": os.environ.get("SLURMD_NODENAME", ""),
            "omp_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
            "torch_threads": torch.get_num_threads(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def measure(fn):
    """One timed call: wall seconds and CPU seconds consumed."""
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter()
    fn()
    w = time.perf_counter() - t0
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    return w, (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)


def summarise(walls, cpus, n_frames, window_frames):
    """Per-frame mean and std, plus what "per frame" actually means for this method.

    THROUGHPUT vs LATENCY. The EnKF assimilates one frame at a time, so wall/n_frames is both
    its amortised cost and the delay between an observation arriving and that frame's estimate
    being available -- for it the two numbers coincide. 4DVarNet reconstructs a whole dT-frame
    window in one solve, so wall/n_frames is throughput only: on a device you cannot produce
    frame t's estimate from frame t's observation, you wait for the window. Reporting a single
    "ms/frame" for both would compare the EnKF's latency against our throughput.

    latency_s is therefore the delay to a given frame's estimate: the cost of the smallest
    unit the method can produce independently -- one frame for a sequential filter, one whole
    dT-frame window for a windowed solver:

        latency = wall / n_frames * window_frames

    `window_frames` is 1 for per-frame methods, so the formula covers both. It used to be
    `walls.mean()` for windowed methods, which is the WHOLE BATCH: at 1000 frames and dT=200
    that is five windows, and 4DVarNet's latency was reported as 2.74 s when one window solves
    in 0.55 s. The docstring already said "one whole window's solve"; the code did not.

    dT is baked into the read-out's weight shape (C*dT channels), so shortening the window for
    online use is a retrain, not a setting.

    The std is taken on the PER-FRAME figure rather than on the batch total, which is what the
    number is quoted as.
    """
    walls, cpus = np.array(walls), np.array(cpus)
    pf = walls / n_frames
    return {"wall_mean": float(walls.mean()), "wall_std": float(walls.std()),
            "wall_all": walls.tolist(), "n_frames": int(n_frames),
            "per_frame_s": float(pf.mean()),
            "per_frame_std_s": float(pf.std()),
            "window_frames": int(window_frames),
            "per_frame_is_latency": bool(window_frames == 1),
            "latency_s": float(pf.mean() * window_frames),
            "latency_std_s": float(pf.std() * window_frames),
            # CPU-seconds / wall-seconds: 1.0 = one core busy the whole time. Shows whether a
            # method actually uses the cores it was given (the EnKF was measured at 1.36 of 4).
            "cores_used": float((cpus / walls).mean())}


# ---------------------------------------------------------------- EnKF
def build_C(cells):
    C = np.zeros((F * len(cells), STATE_DIM))
    for i, (r, c) in enumerate(cells):
        for f in range(F):
            C[i * F + f, f * TOTAL + r * W + c] = 1.0
    return C


def bench_enkf(src, obs_npz, n_frames, warmup_frames=10):
    for m in [k for k in list(sys.modules) if k.startswith(("pedpred", "tools"))]:
        del sys.modules[m]
    root = SRC[src]
    sys.path.insert(0, root)
    E = importlib.import_module("pedpred.ENKF")
    U = importlib.import_module("pedpred.utils")
    model = U.load_model(os.path.join(root, "apt-ibex_train_model_28D.pth"), torch.device("cpu"))
    sys.path.remove(root)

    z = np.load(obs_npz)
    Y, Om, obs_std = z["Y"], z["Omega"], z["obs_std"]
    # Precompute C per frame OUTSIDE the timed region: building C is our driver's bookkeeping,
    # not part of the filter, and it would otherwise be charged to the EnKF.
    pre = []
    for t in range(n_frames):
        cells = list(zip(*np.where(Om[t])))
        C = build_C(cells) if cells else None
        pre.append((C, C @ Y[t].reshape(-1) if C is not None else None))

    def once(limit=None):
        f = E.LocalizedEnsembleKalmanFilter(
            grid_size=(H, W), state_shape=(F, H, W), ensemble_size=100,
            proc_noise_std=PROC_STD, obs_noise_std=tuple(float(s) for s in obs_std),
            init_perturb_std=INIT_STD, inflation=1.02, localization_radius=7)
        rng = np.random.RandomState(0)
        X0 = np.zeros((100, STATE_DIM))
        for i, s in enumerate(PROC_STD):
            X0[:, i * TOTAL:(i + 1) * TOTAL] = rng.normal(0, s, size=(100, TOTAL))
        f.X = X0
        for C, y in (pre if limit is None else pre[:limit]):
            if C is not None:
                try:
                    f.step(C, y, model=model)
                    continue
                except np.linalg.LinAlgError:
                    pass
            f.X = f._clip_bounds(f.forecast(model))
    return once, (lambda: once(warmup_frames)), n_frames, 1   # per-frame assimilation


# ---------------------------------------------------------------- 4DVarNet
def bench_varnet(ckpt, obs_npz, n_frames, device="cpu"):
    from methods.varnet.checks.model_io import load_solver
    dev = torch.device(device)
    solver, a, _ = load_solver(ckpt, dev)
    dT = a["dT"]
    n = (n_frames // dT) * dT or dT                   # whole windows only
    z = np.load(obs_npz)
    # X0 comes straight from the export, so the solver starts from exactly the same initial
    # guess the EnKF pipeline used. Omega is (T,H,W): the robots observe all 4 channels of a
    # cell together, so the per-channel mask is that mask repeated across the channel axis.
    Y, X0 = z["Y"][:n], z["X0"][:n]
    Om = np.repeat(z["Omega"][:n][:, None], F, axis=1).astype(np.float32)

    win = lambda x: om.to_windows(x, dT, device=dev)
    yb, mb, x0b = win(Y), win(Om), win(X0)

    def once():
        with torch.enable_grad():                     # solver differentiates its cost
            solver(x0b, yb, mb)
        if dev.type == "cuda":
            torch.cuda.synchronize()                  # CUDA is async: without this we would
                                                      # time the launch, not the computation
    return once, once, n, dT              # windowed solver: the smallest unit is one dT window


def bench_dincae(run_dir, ckpt_glob, day, n_frames, device="cpu", batch=256):
    """DINCAE: one forward pass produces (mu, sigma^2). A per-frame method, so
    per-frame IS its latency.

    Only the **network forward pass** is timed; input tensors are prepared
    outside the timed region -- consistent with bench_varnet (where the windows
    are also pre-cut). Observations arrive from the robots in the first place, so
    counting the encoding as part of the inference time would charge the data
    pipeline's cost to the method.

    Note FRESH_OFFSETS=(-1,0,1): every frame needs the t-1/t/t+1 observations.
    The **wait-for-data** latency this implies is not measured here; this
    benchmark only compares compute cost.
    """
    from methods.dincae.checks.evaluate import build_inputs, load_models
    from methods.dincae.encoding import FRESH_OFFSETS, observed_pair
    from methods.dincae.state import NCH, StateStats
    dev = torch.device(device)
    stats = StateStats()
    models, _, _ = load_models(run_dir, ckpt_glob, dev)

    oc_sr = config.get("observation", "sensing_range")
    with h5py.File(day, "r") as f:
        T = min(f["grid"].shape[0], n_frames + 2 * max(FRESH_OFFSETS) + 2)
        X = f["grid"][:T]
        t_unix = f["time"][:T]
    obs = om.generate_observations(
        X, sensing_range=oc_sr, num_agents=config.get("observation", "num_agents"),
        add_noise=True, seed=0, valid_mask=stats.valid,
        obs_std=np.asarray(config.get("observation", "obs_std")),
        obs_every_k=config.get("observation", "obs_every_k"))
    scaled, invvar = observed_pair(obs["Y"][:, :NCH], obs["Omega_c"][:, :NCH],
                                   stats.mean, stats.std)
    lo, hi = -min(FRESH_OFFSETS), T - max(FRESH_OFFSETS)
    idx = np.arange(lo, hi)[:n_frames]
    H, W = X.shape[2], X.shape[3]
    chunks = [torch.from_numpy(build_inputs(scaled, invvar, t_unix, idx[b:b + batch], H, W)).to(dev)
              for b in range(0, len(idx), batch)]

    def once():
        with torch.no_grad():
            for xb in chunks:
                for m in models:
                    m(xb)[-1]
        if dev.type == "cuda":
            torch.cuda.synchronize()
    return once, once, len(idx), 1        # per-frame


def bench_senseiver(ckpt, day, n_frames, device="cpu", batch=256):
    """Senseiver: sensors -> field, per-frame and self-contained within a frame,
    so per-frame is its latency.

    Only `model.reconstruct(tok, pad)` is timed; tokenisation happens outside
    the timed region -- the same convention as DINCAE/4DVarNet.
    """
    from methods.senseiver import dataset as sds
    from methods.senseiver import sensors as ssens
    from methods.senseiver.network import Senseiver     # model.py has the layers, the class is in network.py
    dev = torch.device(device)
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    model = Senseiver(**ck["hparams"]).to(dev)
    model.load_state_dict(ck["model"] if "model" in ck else ck["state_dict"])
    model.eval()
    _X, Y, Om = sds.load_day(day, stride=1, seed=0, frames=n_frames,
                             obs_every_k=config.get("observation", "obs_every_k"))
    pe = model.pos_enc.detach().cpu().numpy()
    mean = model.in_mean.detach().cpu().numpy()
    std = model.in_std.detach().cpu().numpy()
    chunks = []
    for i in range(0, Y.shape[0], batch):
        tok, pad, _ = ssens.build_batch(Y[i:i + batch], Om[i:i + batch], pe, mean, std)
        chunks.append((tok.to(dev), pad.to(dev)))

    def once():
        with torch.no_grad():
            for tok, pad in chunks:
                model.reconstruct(tok, pad)
        if dev.type == "cuda":
            torch.cuda.synchronize()
    return once, once, int(Y.shape[0]), 1  # per-frame


def bench_varnet_ensemble(ckpts, obs_npz, n_frames, device="cpu"):
    """Time the WHOLE M-member ensemble: M forward passes plus the Sec. 2.4 combination.

    Measured rather than inferred as M x the single-model time. The members are independent
    networks with no shared computation, so M x is the right first guess, but it is a guess:
    M models resident at once change the allocator's and the cache's behaviour, and the
    moment-matching itself costs something. A benchmark that multiplies is not a benchmark.
    """
    from methods.varnet.checks.model_io import load_solver
    dev = torch.device(device)
    solvers, a = [], None
    for c in ckpts:
        sol, a, _ = load_solver(c, dev)
        solvers.append(sol)
    dT = a["dT"]
    n = (n_frames // dT) * dT or dT
    z = np.load(obs_npz)
    Y, X0 = z["Y"][:n], z["X0"][:n]
    Om = np.repeat(z["Omega"][:n][:, None], F, axis=1).astype(np.float32)
    win = lambda x: om.to_windows(x, dT, device=dev)
    yb, mb, x0b = win(Y), win(Om), win(X0)

    def once():
        mus, vs = [], []
        for sol in solvers:
            with torch.enable_grad():
                xr, vr = sol(x0b, yb, mb, return_var=True)
            mus.append(xr.detach()); vs.append(vr.detach())
        mu, var = torch.stack(mus), torch.stack(vs)
        _ = var.mean(0) + mu.var(0, unbiased=False)        # Sec. 2.4 moment matching
        if dev.type == "cuda":
            torch.cuda.synchronize()
    return once, once, n, dT              # windowed solver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--day", default="atc-20130811")
    ap.add_argument("--gpu", action="store_true",
                    help="also time 4DVarNet on a GPU (reference only — the EnKF has no GPU path)")
    # Which trained model to time. The prior's size shows up here: it is evaluated inside every
    # one of the solver's 20 iterations, so a wider psi is not free even though it is a small
    # fraction of the parameters. Runs must therefore be timed individually, not substituted
    # for each other.
    ap.add_argument("--run", default="a2",
                    help="run-name stem under runs/: times runs/varnet_<run>_k1 and _k4")
    ap.add_argument("--ens-fmt", default="",
                    help="run-name format for a deep ensemble, e.g. runs/varnet_vsb0_s{}. When "
                         "given, two extra rows are timed: one member alone and all members "
                         "together. Uses the k=1 observation file")
    ap.add_argument("--ens-members", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="")
    ap.add_argument("--dincae-ckpt", default="ckpt_00070.pt",
                    help="which DINCAE checkpoint to use -- defaults to the same "
                         "one as the accuracy table (ep70, picked on the validation set)")
    args = ap.parse_args()
    out = args.out or f"check_outputs/eval/bench_speed_{args.run}.json"

    # The EnKF's exports moved from methods/varnet/check_outputs/ to under
    # methods/enkf/ on 2026-09-03 (they are the EnKF's output). Go through
    # paths.enkf_export(), don't reassemble it from ROOT -- hardcoding the old
    # path here made the job FileNotFoundError during setup after the refactor.
    k1 = os.path.join(paths.enkf_export("enkf_k1_full"), f"obs_{args.day}.npz")
    k4 = os.path.join(paths.enkf_export("enkf_k4_full"), f"obs_{args.day}.npz")
    ck = lambda k: f"{ROOT}/runs/varnet_{args.run}_k{k}/varnet_best.pt"
    have_k = all(os.path.exists(ck(k)) for k in (1, 4))
    ens_ck = [f"{ROOT}/{args.ens_fmt.format(i)}/varnet_best.pt" for i in args.ens_members] \
        if args.ens_fmt else []
    for c in ens_ck:
        if not os.path.exists(c):
            raise SystemExit(f"no checkpoint at {c} — check --ens-fmt/--ens-members")
    if not have_k and not ens_ck:
        raise SystemExit(f"no checkpoint at {ck(1)} — pass --run or --ens-fmt")
    day_h5 = om.day_path(args.day) if hasattr(om, "day_path") else os.path.join(
        om.GRID_CACHE, f"{args.day}_corridor_1.0s.h5")
    hw = hw_info()
    print(f"[hw] {hw['cpu']} · {hw['cores_avail']} cores · node={hw['node']} · "
          f"OMP={hw['omp_threads']} torch={hw['torch_threads']}", flush=True)
    print(f"[gpu] {hw['gpu'] or 'none'}", flush=True)
    print(f"[cfg] {args.frames} frames, {args.repeats} timed repeats (+1 warm-up each), "
          f"4DVarNet = runs/varnet_{args.run}_k*\n", flush=True)

    res = {"hw": hw, "frames": args.frames, "repeats": args.repeats, "day": args.day,
           "varnet_run": args.run, "runs": {}}
    jobs = [
        ("enkf_k1", "EnKF k=1, optimised (bit-identical to the original)",
         lambda: bench_enkf("opt", k1, args.frames)),
        ("enkf_k4", "EnKF k=4, optimised",
         lambda: bench_enkf("opt", k4, args.frames)),
    ]
    if have_k:
        jobs += [
            ("varnet_k1", f"4DVarNet k=1 ({args.run})",
             lambda: bench_varnet(ck(1), k1, args.frames, "cpu")),
            ("varnet_k4", f"4DVarNet k=4 ({args.run})",
             lambda: bench_varnet(ck(4), k4, args.frames, "cpu")),
        ]
    if ens_ck:
        jobs += [
            ("varnet_single", f"4DVarNet, 1 member ({args.ens_fmt.format(args.ens_members[0])})",
             lambda: bench_varnet(ens_ck[0], k1, args.frames, "cpu")),
            (f"varnet_ens{len(ens_ck)}",
             f"4DVarNet deep ensemble, {len(ens_ck)} members + Sec. 2.4 combination",
             lambda: bench_varnet_ensemble(ens_ck, k1, args.frames, "cpu")),
        ]
    # DINCAE and Senseiver: per-frame methods, completing the four-way
    # comparison. If a checkpoint is missing, skip it with a note rather than
    # failing the whole benchmark over one missing file.
    dinc_dir = os.path.join(paths.method(paths.DINCAE), "runs", "dincae_full")
    dinc_ck = os.path.join(dinc_dir, args.dincae_ckpt)
    if os.path.exists(dinc_ck):
        jobs += [("dincae", f"DINCAE, 1 checkpoint ({args.dincae_ckpt})",
                  lambda: bench_dincae(dinc_dir, dinc_ck, day_h5, args.frames, "cpu"))]
    else:
        print(f"[skip] dincae: no {dinc_ck}", flush=True)
    sens_ck = os.path.join(paths.method(paths.SENSEIVER), "runs", "senseiver_A", "best.pt")
    if os.path.exists(sens_ck):
        jobs += [("senseiver", "Senseiver, 1 model (best.pt)",
                  lambda: bench_senseiver(sens_ck, day_h5, args.frames, "cpu"))]
    else:
        print(f"[skip] senseiver: no {sens_ck}", flush=True)

    if args.gpu and torch.cuda.is_available() and have_k:
        # Reported separately, never as the head-to-head: the EnKF has no GPU path (its Kalman
        # update is host numpy), so a GPU-vs-CPU ratio would measure hardware, not method.
        jobs += [("varnet_k1_gpu", f"4DVarNet k=1 ({args.run}), GPU (reference only)",
                  lambda: bench_varnet(ck(1), k1, args.frames, "cuda"))]

    # Build every measurement first (model loading, data prep), then warm each one up.
    print("[setup] preparing + warming up ...", flush=True)
    prepared = {}
    for name, desc, fn in jobs:
        run_once, warm, nf, win = fn()
        warm()
        prepared[name] = (run_once, nf, desc, win)
        print(f"        {name}: ready ({nf} frames, window {win})", flush=True)

    # INTERLEAVED, not blocked: round 1 runs every method, then round 2, then round 3.
    # Blocking (AAA BBB CCC) would let a change in the node's background load land entirely
    # on one method and be read as a difference between methods. Interleaving (ABC ABC ABC)
    # spreads any drift across all of them, and what is left shows up as std rather than
    # hiding as bias. This is what makes the numbers usable without an exclusive node —
    # every batch-csl and H200 node is currently shared, and none was going to free up.
    walls = {n: [] for n in prepared}
    cpus = {n: [] for n in prepared}
    for rep in range(args.repeats):
        print(f"\n[round {rep + 1}/{args.repeats}]", flush=True)
        for name, (run_once, nf, _, _w) in prepared.items():
            w, c = measure(run_once)
            walls[name].append(w); cpus[name].append(c)
            print(f"  {name:16s} {w:8.2f} s   {w / nf * 1000:8.2f} ms/frame", flush=True)

    for name, (_, nf, desc, win) in prepared.items():
        # window is reported by each bench function itself (per-frame=1,
        # windowed solver=dT). Used to be guessed from the name prefix, which
        # broke once DINCAE / Senseiver were added.
        res["runs"][name] = summarise(walls[name], cpus[name], nf, win) | {"desc": desc}

    print("\n" + "=" * 92)
    print(f"{'method':16s} {'throughput ms/frame':>22s} {'latency to one frame':>22s} "
          f"{'cores':>6s}  vs fastest")
    fastest = min(r["per_frame_s"] for r in res["runs"].values())
    for n, r in res["runs"].items():
        lat = (f"{r['latency_s'] * 1000:.2f} ms" if r["per_frame_is_latency"]
               else f"{r['latency_s']:.2f} s (whole window)")
        print(f"{n:16s} {r['per_frame_s'] * 1000:>13.2f}±{r['per_frame_std_s'] * 1000:<7.3f} "
              f"{lat:>22s} {r['cores_used']:>6.1f}  {r['per_frame_s'] / fastest:>8.1f}x")
    print("\nthroughput = amortised cost per frame over the batch. latency = delay until a given\n"
          "frame's estimate exists: one frame for the sequential filter, one whole window solve\n"
          "for the windowed method, which is not an online/causal estimator.")
    # Spread relative to the effect: if the largest std is small next to the gaps between
    # methods, contention from co-tenant jobs did not change the ranking.
    worst = max(r["wall_std"] / r["wall_mean"] for r in res["runs"].values())
    print(f"\nlargest run-to-run spread: {worst * 100:.1f}% of the mean "
          f"(node was shared — see hw/node above)")
    p = os.path.join(ROOT, out)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, "w"), indent=2)
    print(f"\n[out] {p}")


if __name__ == "__main__":
    main()
