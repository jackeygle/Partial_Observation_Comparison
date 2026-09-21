"""
plot_reconstruction_sequence.py — N consecutive frames, all 4 methods, one PNG per frame
==========================================================================================

Generalizes `methods/varnet/checks/plot_reconstruction_sequence.py` (4DVarNet
+ EnKF only) to include DINCAE and Senseiver too, reusing the per-method
reconstruction logic from `compare/plot_reconstruction.py`.

Each method's reconstruction for the WHOLE [start, start+N) block is computed
ONCE (batched) before the plotting loop starts:

  - EnKF: already exported, just slice `est_<day>.npz`
  - DINCAE: one `predict_day()` call (its own internal batching)
  - Senseiver: chunked forward passes (`--batch` frames at a time)
  - 4DVarNet: `dT`-frame windows (its native chunk), one solver pass per window

For 5000 frames the model-inference above takes minutes; the dominant cost is
the N `savefig` calls after it, one full-resolution multi-panel figure each
(the figure and axes are reused across frames via `ax.cla()`, not recreated,
which is what keeps this from being much slower than it already is).

**Output is intentionally NOT tracked in git.** Default `--outdir` matches the
`**/seq_ppt_*/` rule already in `.gitignore` (extended to cover `compare/
results/` too) -- 5000 PNGs is 100-500MB depending on panel count, cheap to
regenerate, not something to commit.

The "partial obs" panel is illustrative when several methods are shown (one
shared simulated observation, not every method's own input).  For a
Senseiver-only run it is the exact noisy observation and mask fed to Senseiver,
including the robot trajectory advanced from frame zero.

Usage (GPU node):
    python3 -m compare.plot_reconstruction_sequence --day atc-20130811 --n 5000
    # fewer methods / a specific block instead of auto-picking the busiest one:
    python3 -m compare.plot_reconstruction_sequence --methods dincae,senseiver --day atc-20130811 --n 500 --start 10000

"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from crowdcore import config
from crowdcore import navigation as nav
from crowdcore import observation_model as om
from crowdcore import paths

# Set from --trajectory-mode. None follows crowdcore's configured default; the
# override exists because the EnKF-style exports this script reads were produced
# before per_day routes, so matching them takes an explicit "fixed".
TRAJECTORY_MODE = None


def day_seed(day_file):
    return om.day_seed(day_file, trajectory_mode=TRAJECTORY_MODE)

from methods.varnet.checks.model_io import load_solver, baseline_ckpt
from methods.dincae.checks.evaluate import load_models as dincae_load_models, \
    predict_day as dincae_predict_day, clip_bounds as dincae_clip_bounds
from methods.senseiver.checks.evaluate import load_model as senseiver_load_model, \
    clip_bounds as senseiver_clip_bounds
from methods.senseiver import dataset as sds
from methods.senseiver import sensors as ssensors

CLIP = ((0.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (0.0, 2.0))


def clip_np(x):                             # (n,4,H,W)
    x = x.copy()
    for c, (lo, hi) in enumerate(CLIP):
        np.clip(x[:, c], lo, hi, out=x[:, c])
    return x


def panel(ax, state, vmax, title, speed_scale):
    """Density as the colour, velocity as arrows whose LENGTH is the speed.

    The arrows used to be unit vectors, which showed heading but said nothing
    about how fast anyone was moving; a still crowd and a rushing one drew the
    same picture.  ``speed_scale`` is fixed for the whole sequence so a long
    arrow means the same speed in every frame and in every panel.
    """
    dens, vx, vy = state[0], state[1], state[2]
    x, y = np.meshgrid(np.arange(dens.shape[1]), np.arange(dens.shape[0]))
    m = np.isnan(dens)
    u = np.where(m, np.nan, vx); v = np.where(m, np.nan, vy)
    im = ax.imshow(dens, cmap="Blues", origin="upper", aspect="equal", vmin=0, vmax=vmax)
    ax.quiver(x, y, u, v, color="black", scale=speed_scale, scale_units="width",
              headwidth=3, headlength=4)
    ax.set_title(title, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
    return im


# --------------------------------------------------------------------------- #
# Per-method batched reconstruction over [start, start+N)
# --------------------------------------------------------------------------- #

def block_varnet(ckpt, X, start, N, dev, day_file):
    """X is the FULL day array; reconstructs [start, start+N) in dT windows."""
    solver, a, _ = load_solver(ckpt, dev)
    dT = a["dT"]
    assert N % dT == 0, f"--n must be a multiple of 4DVarNet's dT={dT} (got {N})"
    Xb = X[start:start + N]
    out = om.generate_observations(Xb, add_noise=True, seed=day_seed(day_file),
                                   valid_mask=nav.build_valid_mask_from_config(Xb))
    x0 = om.fill_missing_state(out["Y"], out["Omega_c"], method=config.get("observation", "init_method"))
    win = lambda arr: om.to_windows(arr, dT)
    yb, mb, x0b = torch.from_numpy(win(out["Y"])).float(), \
        torch.from_numpy(win(out["Omega_c"].astype(np.float32))).float(), \
        torch.from_numpy(win(x0)).float()
    recs = []
    with torch.enable_grad():
        for i in range(x0b.shape[0]):
            r = solver(x0b[i:i + 1].to(dev), yb[i:i + 1].to(dev), mb[i:i + 1].to(dev)).detach().cpu()
            recs.append(r)
    xr = om.from_windows(torch.cat(recs, 0).numpy())[:N]
    return clip_np(xr)


def block_dincae(run_dir, ckpt, day_file, start, N, dev):
    from methods.dincae.state import StateStats
    from methods.dincae.encoding import FRESH_OFFSETS
    stats = StateStats()
    # A named checkpoint, never the ckpt_*.pt glob -- the glob averages every
    # checkpoint's outputs, a different configuration from every reported
    # DINCAE number (see evaluate.PUBLISHED_CKPT).
    models, epochs, _ = dincae_load_models(run_dir, os.path.join(run_dir, ckpt), dev)
    lo, hi_margin = -min(FRESH_OFFSETS), max(FRESH_OFFSETS)
    # predict_day's `frames` truncates the day to T = min(T_all, frames); it also
    # needs hi_margin frames of FUTURE context past the last predicted frame, or
    # its own idx_all range comes up short and every array below is silently
    # truncated (caught by testing this exact off-by-one against a real run).
    _, rec, _, _, _ = dincae_predict_day(models, stats, day_file, dev, frames=start + N + hi_margin)
    lo_i, hi_i = start - lo, start - lo + N
    if lo_i < 0:
        raise ValueError(f"--start must be >= {lo} for DINCAE (its FRESH_OFFSETS need frames before start)")
    return dincae_clip_bounds(rec[lo_i:hi_i])


def block_senseiver(ckpt, day_file, start, N, batch, dev):
    model, ck = senseiver_load_model(ckpt, dev)
    k = ck["args"].get("obs_every_k") or sds.obs_config()["obs_every_k"]
    X, Y, Om = sds.load_day(day_file, stride=1, seed=day_seed(day_file), frames=start + N, obs_every_k=k)
    pe = model.pos_enc.detach().cpu().numpy()
    mean = model.in_mean.detach().cpu().numpy()
    std = model.in_std.detach().cpu().numpy()
    out = []
    with torch.no_grad():
        for i in range(start, start + N, batch):
            sl = slice(i, min(i + batch, start + N))
            if getattr(model, "time_window", 1) > 1:
                time_window = model.time_window
                windows = [
                    (
                        Y[max(0, j - time_window + 1):j + 1],
                        Om[max(0, j - time_window + 1):j + 1],
                    )
                    for j in range(sl.start, sl.stop)
                ]
                tok, pad, dt, _, cell_idx = ssensors.build_batch_temporal(
                    windows, pe, mean, std, return_cell_idx=True
                )
                xr = senseiver_clip_bounds(
                    model.reconstruct(tok.to(dev), pad.to(dev), dt.to(dev),
                                      cell_idx.to(dev))
                )
                out.append(xr.cpu().numpy())
                continue
            tok, pad, _ = ssensors.build_batch(Y[sl], Om[sl], pe, mean, std)
            xr = senseiver_clip_bounds(model.reconstruct(tok.to(dev), pad.to(dev)))
            out.append(xr.cpu().numpy())
    recon = np.concatenate(out, 0)
    c, h, w = sds.state_shape()
    exact_obs = Y[start:start + N].reshape(N, c, h, w)
    exact_omega = Om[start:start + N].reshape(N, h, w)
    return recon, exact_obs, exact_omega


def block_enkf(enkf_dir, day, start, N):
    z = np.load(os.path.join(enkf_dir, f"est_{day}.npz"))
    return z["Est"][start:start + N], z["Spread"][start:start + N, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="dincae,senseiver,varnet,enkf")
    ap.add_argument("--day", default="atc-20130811")
    ap.add_argument("--n", type=int, default=5000, help="number of consecutive frames")
    ap.add_argument("--start", type=int, default=-1, help="start frame; -1 = auto-pick the busiest N-frame block")
    ap.add_argument("--batch", type=int, default=256, help="Senseiver forward-pass chunk size")
    ap.add_argument("--varnet-ckpt", default=None,
                    help="default: the reported 4DVarNet model (model_io.baseline_ckpt) -- it used to be "
                         "runs/varnet_mse5_s0/varnet_best.pt, a hidden=32 run at a train-split-selected epoch")
    ap.add_argument("--dincae-run-dir", default=os.path.join(paths.method(paths.DINCAE), "runs", "dincae_ff"))
    ap.add_argument("--dincae-ckpt", default="ckpt_00060.pt",
                    help="the checkpoint every reported DINCAE number uses (evaluate.PUBLISHED_CKPT)")
    ap.add_argument("--senseiver-ckpt", default=os.path.join(paths.method(paths.SENSEIVER), "runs", "senseiver_A", "best.pt"))
    ap.add_argument("--enkf-dir", default=paths.enkf_export("enkf_k1_full"),
                    help="comma-separated export directories; one panel per directory")
    ap.add_argument("--enkf-label", default="EnKF",
                    help="panel title for whatever --enkf-dir holds; set it when the "
                         "export is not the plain EnKF filter")
    ap.add_argument("--spread-vmax", type=float, default=0.0,
                    help="upper end of the spread panel's colour scale; 0 takes the "
                         "sequence's 99th percentile. Set it explicitly to the same "
                         "value for every model being compared")
    ap.add_argument("--trajectory-mode", default=None, choices=("fixed", "per_day"),
                    help="robot-route seeding for the observations this script "
                         "simulates; default follows crowdcore's config")
    ap.add_argument("--outdir", default="")
    args = ap.parse_args()

    global TRAJECTORY_MODE
    TRAJECTORY_MODE = args.trajectory_mode

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dT = 200

    day_file = [d for d in om.split_files("test") if args.day in d][0]
    Xall, _ = om.load_state(day_file); Xall = np.asarray(Xall)
    N = (args.n // dT) * dT if "varnet" in methods else args.n
    if N != args.n:
        print(f"[seq] --n rounded {args.n} -> {N} (whole number of 4DVarNet's dT={dT} windows)")

    if args.start < 0:
        dens = Xall[:, 0].reshape(Xall.shape[0], -1).sum(1)
        csum = np.concatenate([[0], np.cumsum(dens)])
        block = csum[N:] - csum[:-N]
        start = int(np.argmax(block))
        if "varnet" in methods:
            start = (start // dT) * dT
    else:
        start = (args.start // dT) * dT if "varnet" in methods else args.start
    start = min(start, Xall.shape[0] - N)
    print(f"[seq] day={args.day} methods={methods} frames [{start}, {start+N}) = {N} frames", flush=True)

    X = Xall[start:start + N]
    out = om.generate_observations(X, add_noise=True, seed=day_seed(day_file),
                                   valid_mask=nav.build_valid_mask_from_config(X))
    Omega, obs = out["Omega"], out["Y"].copy()
    vmax = max(0.3, float(np.percentile(X[:, 0], 99)))

    recon, spread, spreads = {}, None, {}
    if "dincae" in methods:
        print("[seq] running DINCAE ...", flush=True)
        recon["DINCAE"] = block_dincae(args.dincae_run_dir, args.dincae_ckpt, day_file, start, N, dev)
    if "senseiver" in methods:
        print("[seq] running Senseiver ...", flush=True)
        senseiver_rec, senseiver_obs, senseiver_omega = block_senseiver(
            args.senseiver_ckpt, day_file, start, N, args.batch, dev)
        recon["Senseiver"] = senseiver_rec
        if methods == ["senseiver"]:
            # For a Senseiver-only diagnostic, show the exact observations fed
            # to the model.  Re-initialising the robot path at `start` produces
            # a different mask from advancing it from frame zero.
            obs, Omega = senseiver_obs, senseiver_omega
    if "varnet" in methods:
        print("[seq] running 4DVarNet ...", flush=True)
        recon["4DVarNet"] = block_varnet((args.varnet_ckpt or baseline_ckpt()), Xall, start, N, dev, day_file)
    if "enkf" in methods:
        print("[seq] loading EnKF export ...", flush=True)
        dirs = [d.strip() for d in args.enkf_dir.split(",") if d.strip()]
        labels = [l.strip() for l in args.enkf_label.split(",") if l.strip()]
        if len(labels) != len(dirs):
            raise SystemExit(f"--enkf-label needs {len(dirs)} comma-separated names")
        for d, lab in zip(dirs, labels):
            est, spread = block_enkf(d, args.day, start, N)
            recon[lab] = clip_np(est)                    # est is already (N,4,H,W)
            spreads[lab] = spread
        spread = spreads[labels[0]]
    # One scale for the whole sequence, never per frame: matplotlib's autoscale
    # would rescale every frame to its own maximum, so a quiet frame and a busy
    # one look equally red (measured 4.3x between the quietest and busiest frame)
    # and the panel stops saying anything about the absolute spread. Pass
    # --spread-vmax to hold it fixed across models, which a comparison needs.
    if spread is not None:
        # One number across every frame AND every model: two panels that share a
        # colour map but not its limits look alike while differing several-fold.
        pooled = np.concatenate([np.nan_to_num(s).ravel() for s in spreads.values()])
        spread_vmax = args.spread_vmax or float(np.percentile(pooled, 99))
        print(f"[seq] spread colour scale: vmin=0 vmax={spread_vmax:.4f}"
              f"{' (given)' if args.spread_vmax else ' (p99 over all models and frames)'}"
              f"; density scale vmax={vmax:.4f}", flush=True)
    speed_scale = max(1e-6, 7.0 * float(np.percentile(np.abs(X[:, 1:3]), 99)))

    outdir = args.outdir or os.path.join(paths.COMPARE, "results", f"seq_ppt_{args.day}")
    os.makedirs(outdir, exist_ok=True)
    n_spread = len(spreads) if spread is not None else 0
    ncol = 2 + len(recon) + n_spread
    fig, axes = plt.subplots(1, ncol, figsize=(3.1 * ncol + 1.1, 4.7))
    fig.subplots_adjust(right=0.88, bottom=0.14)
    print(f"[seq] rendering {N} frames ({ncol} panels each) -> {outdir}", flush=True)
    for t in range(N):
        obs_t = obs[t].copy(); obs_t[:, ~Omega[t]] = np.nan
        cells = [("partial obs", obs_t), ("true state", X[t])] + [(name, arr[t]) for name, arr in recon.items()]
        im_d = None
        for ax, (ti, s) in zip(axes, cells):
            ax.cla(); im_d = panel(ax, s, vmax, ti, speed_scale)
        im_s = None
        for ax, lab in zip(axes[len(cells):], labels):
            ax.cla()
            im_s = ax.imshow(np.nan_to_num(spreads[lab][t]), cmap="Reds", origin="upper",
                             aspect="equal", vmin=0, vmax=spread_vmax)
            ax.set_title(f"{lab}\ndensity sigma", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        if t == 0:
            cb1 = fig.colorbar(im_d, ax=axes[:len(cells)].tolist(), fraction=0.020, pad=0.012)
            cb1.set_label("density", fontsize=9); cb1.ax.tick_params(labelsize=8)
            if im_s is not None:
                cb2 = fig.colorbar(im_s, ax=axes[len(cells):].tolist(), fraction=0.045, pad=0.012)
                cb2.set_label("density sigma-hat", fontsize=9); cb2.ax.tick_params(labelsize=8)
        seen = int(Omega[t].sum())
        fig.suptitle(f"{args.day}   frame {start+t}   (seq {t:05d}/{N})   "
                     f"observed {seen}/{Omega[t].size} cells", fontsize=11)
        fig.savefig(os.path.join(outdir, f"frame_{t:05d}.png"), dpi=100, bbox_inches="tight")
        if t % 200 == 0:
            print(f"  rendered {t}/{N}", flush=True)
    plt.close(fig)
    print(f"[done] {N} PNGs ({ncol} panels) -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
