"""
run_enkf_surrogate.py — the EnKF with the retrained PedPred3 deep ensemble as its forecast model
==================================================================================================

Arms, all on the same exported observations:

  E0  shipped      vendored PedPred3 (apt-ibex), 0.01 x Q injected, published localisation
  E1  mean         each member, each step: a random pair's surrogate_mean, 0.01 x Q injected
  E2  mean+sigma   surrogate_mean + sigma * eps of a random pair, no injected noise
  E3  E2 + fix_localization
  E4  E1 + fix_localization
  E5  E2 with sigma x 0.3        E7  E5 + fix_localization
  E6  E2 with sigma x 0.1        E8  E6 + fix_localization
  E9 / E10 / E11   sigma x 0.3 + fix_localization, radius 2 / 3 / 5
  E12              sigma x 0.5, published localisation
  E13 / E14        sigma x 0.5 + fix_localization, radius 3 / 5

surrogate_mean = the mean network's forecast with vx/vy/var zeroed where the forecast density is
below 0.01, clipped (methods/enkf/surrogate/model.py). Pairs are runs/surrogate_{mean,sigma}_s*/best.pt.

Everything else is enkf_opt's LocalizedEnsembleKalmanFilter as it stands -- bias correction on,
inflation 1.02, radius 7, pinv gain. The subclass overrides only _f_model(), which forecast()
calls before it adds the injected noise, subtracts the bias estimate and clips, so E0 runs the
identical code path as checks/run_enkf_baseline.py --enkf-src opt. The cold start (random
Gaussian ensemble, std = PROC_STD, RandomState(0)) and the frame loop are copied from that driver:
it cannot be imported, because it imports pedpred at module level from whichever copy argv names.

enkf_opt/pedpred/ENKF.py carries uncommitted switches (proc_noise_scale, fix_localization) that
E2/E3 rely on; run_config.json in each output directory records its sha256 and git state.
pedpred/models.py is imported from enkf_opt here; it is byte-identical to enkf_lab's.

    source sbatch/_env.sh && cd methods/enkf
    python3 -u -m methods.enkf.checks.run_enkf_surrogate --arms E2 --only atc-20130616 --frames 300
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

from crowdcore import paths

OPT = paths.enkf_vendor("enkf_opt")
sys.path.insert(0, OPT)
from pedpred.ENKF import LocalizedEnsembleKalmanFilter                                   # noqa: E402
from pedpred.utils import load_model                                                      # noqa: E402

from methods.enkf.surrogate.model import EMPTY_DENSITY, load_pedpred3, raw_forecast, surrogate_mean  # noqa: E402
from methods.enkf.surrogate.train import LOGVAR_MAX, LOGVAR_MIN                          # noqa: E402

H, W, F = 36, 12, 4
TOTAL = H * W
STATE_DIM = F * TOTAL
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # methods/enkf
# run_enkf_baseline.py: calibrated values from the original ENKF.py main()
PROC_STD = (0.02829307, 0.31263075, 0.12325809, 0.41680932)
INIT_STD = (0.2290, 1.2660, 0.3429, 0.0259)

ARMS = {
    "E0": dict(surrogate=None, proc_noise_scale=0.01, fix_localization=False, sigma_scale=1.0),
    "E1": dict(surrogate="mean", proc_noise_scale=0.01, fix_localization=False, sigma_scale=1.0),
    "E2": dict(surrogate="mean+sigma", proc_noise_scale=0.0, fix_localization=False, sigma_scale=1.0),
    "E3": dict(surrogate="mean+sigma", proc_noise_scale=0.0, fix_localization=True, sigma_scale=1.0),
    # Round 2, after E2/E3 over-dispersed ~3x: separate the localisation fix from the spread
    # (E4), and shrink the sampled sigma (E5-E8). sigma is calibrated for ONE step from a true
    # state; sampled every step it compounds through the model.
    "E4": dict(surrogate="mean", proc_noise_scale=0.01, fix_localization=True, sigma_scale=1.0),
    "E5": dict(surrogate="mean+sigma", proc_noise_scale=0.0, fix_localization=False, sigma_scale=0.3),
    "E6": dict(surrogate="mean+sigma", proc_noise_scale=0.0, fix_localization=False, sigma_scale=0.1),
    "E7": dict(surrogate="mean+sigma", proc_noise_scale=0.0, fix_localization=True, sigma_scale=0.3),
    "E8": dict(surrogate="mean+sigma", proc_noise_scale=0.0, fix_localization=True, sigma_scale=0.1),
    # Round 3: every fix_localization arm above did worse than its pair without the fix. Keep the
    # fix and shrink the radius (7 cells spans most of the 12-cell-wide grid, so spurious
    # long-range covariances reach far), and try sigma x 0.5 between the still under-dispersed
    # 0.3 and the runaway 1.0. Arms without "radius" use --radius (default 7).
    "E9": dict(surrogate="mean+sigma", proc_noise_scale=0.0, fix_localization=True, sigma_scale=0.3, radius=2),
    "E10": dict(surrogate="mean+sigma", proc_noise_scale=0.0, fix_localization=True, sigma_scale=0.3, radius=3),
    "E11": dict(surrogate="mean+sigma", proc_noise_scale=0.0, fix_localization=True, sigma_scale=0.3, radius=5),
    "E12": dict(surrogate="mean+sigma", proc_noise_scale=0.0, fix_localization=False, sigma_scale=0.5),
    "E13": dict(surrogate="mean+sigma", proc_noise_scale=0.0, fix_localization=True, sigma_scale=0.5, radius=3),
    "E14": dict(surrogate="mean+sigma", proc_noise_scale=0.0, fix_localization=True, sigma_scale=0.5, radius=5),
}


class SurrogateEnKF(LocalizedEnsembleKalmanFilter):
    """enkf_opt's filter with the forecast model call replaced; forecast() itself is inherited."""

    def __init__(self, *args, pairs, use_sigma, sigma_scale=1.0, seed=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.pairs, self.use_sigma, self.sigma_scale = pairs, use_sigma, sigma_scale
        self.gen = torch.Generator().manual_seed(seed)

    def _f_model(self, x_np_batch, model):
        N = x_np_batch.shape[0]
        x = torch.from_numpy(np.ascontiguousarray(x_np_batch, dtype=np.float32).reshape(N, 1, *self.state_shape))
        pick = torch.randint(len(self.pairs), (N,), generator=self.gen)
        out = torch.empty_like(x)
        with torch.no_grad():
            for j, (mean_net, sigma_net) in enumerate(self.pairs):
                idx = (pick == j).nonzero().squeeze(1)
                if idx.numel() == 0:
                    continue
                xe = x[idx]
                y = surrogate_mean(mean_net, xe)
                if self.use_sigma:
                    sd = torch.exp(0.5 * raw_forecast(sigma_net, xe).clamp(LOGVAR_MIN, LOGVAR_MAX))
                    y = y + self.sigma_scale * sd * torch.randn(y.shape, generator=self.gen)
                out[idx] = y
        return out.numpy().reshape(N, -1)


def build_C(obs_cells):
    C = np.zeros((F * len(obs_cells), STATE_DIM))
    for i, (r, c) in enumerate(obs_cells):
        for f in range(F):
            C[i * F + f, f * TOTAL + r * W + c] = 1.0
    return C


def run_day(npz, arm, model, pairs, ensemble, radius, frames):
    z = np.load(npz)
    X_true, Y, Omega, obs_std = z["X_true"], z["Y"], z["Omega"], z["obs_std"]
    T = min(frames, X_true.shape[0]) if frames > 0 else X_true.shape[0]
    cfg = ARMS[arm]
    kw = dict(grid_size=(H, W), state_shape=(F, H, W), ensemble_size=ensemble,
              proc_noise_std=PROC_STD, obs_noise_std=tuple(float(s) for s in obs_std),
              init_perturb_std=INIT_STD, inflation=1.02, localization_radius=cfg.get("radius", radius),
              proc_noise_scale=cfg["proc_noise_scale"], fix_localization=cfg["fix_localization"])
    if cfg["surrogate"] is None:
        enkf = LocalizedEnsembleKalmanFilter(**kw)
    else:
        enkf = SurrogateEnKF(pairs=pairs, use_sigma=cfg["surrogate"] == "mean+sigma",
                             sigma_scale=cfg["sigma_scale"], **kw)
        model = "surrogate"          # forecast() only checks `model is None`
    # ORIGINAL cold start: random Gaussian ensemble per channel (no truth, no observations)
    rng = np.random.RandomState(0)
    X_init = np.zeros((ensemble, STATE_DIM))
    for f, s in enumerate(PROC_STD):
        X_init[:, f * TOTAL:(f + 1) * TOTAL] = rng.normal(0, s, size=(ensemble, TOTAL))
    enkf.X = X_init

    Est = np.zeros((T, F, H, W), np.float32)
    Spread = np.zeros((T, F, H, W), np.float32)
    n_bad = 0
    t0 = time.time()
    for t in range(T):
        cells = list(zip(*np.where(Omega[t])))
        ok = False
        if cells:
            C = build_C(cells)
            y_obs = C @ Y[t].reshape(-1)
            try:
                est = enkf.step(C, y_obs, model=model); ok = True
            except np.linalg.LinAlgError:
                n_bad += 1
        if not ok:
            Xf = enkf.forecast(model); enkf.X = enkf._clip_bounds(Xf)
            est = enkf.X.mean(axis=0).reshape(F, H, W)
        Est[t] = est
        Spread[t] = enkf.get_std().reshape(F, H, W)
        if t in (9, 99, 999) or (t + 1) % 5000 == 0:
            print(f"    frame {t + 1}/{T}  {(time.time() - t0) / (t + 1) * 1e3:.0f} ms/frame", flush=True)
    return Est, Spread, T, time.time() - t0, n_bad


def provenance(pairs_files):
    enkf_py = os.path.join(OPT, "pedpred", "ENKF.py")
    git = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True, cwd=HERE).stdout.strip()
    return {"enkf_src": OPT, "enkf_py_sha256": hashlib.sha256(open(enkf_py, "rb").read()).hexdigest(),
            "enkf_py_uncommitted": bool(git("status", "--porcelain", "--", enkf_py)),
            "git_head": git("rev-parse", "HEAD"), "pairs": pairs_files, "empty_density": EMPTY_DENSITY,
            "logvar_clamp": [LOGVAR_MIN, LOGVAR_MAX]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--obs-dir", default="check_outputs/enkf_valid_k1", help="obs_*.npz from export_obs_for_enkf")
    ap.add_argument("--out-fmt", default="check_outputs/enkf_valid_{arm}", help="est_*.npz written here")
    ap.add_argument("--only", default="")
    ap.add_argument("--frames", type=int, default=0, help="0 = the whole day")
    ap.add_argument("--ensemble", type=int, default=100)
    ap.add_argument("--radius", type=int, default=7)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))

    files = sorted(glob.glob(os.path.join(a.obs_dir, "obs_*.npz")))
    files = [f for f in files if a.only in f]
    if not files:
        raise SystemExit(f"no obs_*.npz matching {a.only!r} in {a.obs_dir}")
    runs = os.path.join(HERE, "runs")
    pair_files = [(f"{runs}/surrogate_mean_s{s}/best.pt", f"{runs}/surrogate_sigma_s{s}/best.pt") for s in a.seeds]
    model = pairs = None
    if any(ARMS[arm]["surrogate"] is None for arm in a.arms):
        model = load_model(os.path.join(OPT, "apt-ibex_train_model_28D.pth"), torch.device("cpu"))
    if any(ARMS[arm]["surrogate"] for arm in a.arms):
        pairs = [(load_pedpred3(m).eval(), load_pedpred3(s).eval()) for m, s in pair_files]

    for arm in a.arms:
        out = a.out_fmt.format(arm=arm)
        os.makedirs(out, exist_ok=True)
        json.dump({"arm": arm, **ARMS[arm], "ensemble": a.ensemble, "radius": ARMS[arm].get("radius", a.radius), "frames": a.frames,
                   "obs_dir": a.obs_dir, **provenance(pair_files if ARMS[arm]["surrogate"] else None)},
                  open(os.path.join(out, "run_config.json"), "w"), indent=2)
        for npz in files:
            stem = os.path.basename(npz)[4:-4]
            print(f"[{arm}] {stem} -> {out}", flush=True)
            Est, Spread, T, dt, n_bad = run_day(npz, arm, model, pairs, a.ensemble, a.radius, a.frames)
            np.savez_compressed(os.path.join(out, f"est_{stem}.npz"), Est=Est, Spread=Spread)
            if not os.path.exists(os.path.join(out, f"obs_{stem}.npz")):
                os.symlink(os.path.relpath(npz, out), os.path.join(out, f"obs_{stem}.npz"))
            json.dump({"arm": arm, "day": stem, "n_frames": int(T), "total_s": dt, "per_frame_s": dt / max(T, 1),
                       "svd_skips": n_bad, "node": os.environ.get("SLURMD_NODENAME", ""),
                       "cores": len(os.sched_getaffinity(0))},
                      open(os.path.join(out, f"timing_{stem}.json"), "w"), indent=2)
            print(f"[{arm}] {stem}: {T} frames in {dt:.0f}s ({dt / max(T, 1) * 1e3:.0f} ms/frame), "
                  f"{n_bad} SVD-skips", flush=True)
    print("[done]")


if __name__ == "__main__":
    main()
