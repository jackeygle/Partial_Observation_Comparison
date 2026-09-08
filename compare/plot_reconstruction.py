"""
plot_reconstruction.py — one reconstructed field, side by side across ALL FOUR methods
========================================================================================

`plot_reconstruction_enkf.py` only ever compares 4DVarNet + EnKF (it loads a
4DVarNet solver directly). This is the general version: pick one busy
held-out frame and show as many methods as asked for, in the same
density(Blues) + velocity-heading-arrow style used throughout this repo, so
DINCAE's and Senseiver's reconstructions can finally be looked at as images,
not just error numbers.

Panel layout: true state | partial obs (reference only) | <method 1> |
<method 2> | ... | EnKF spread (only if `enkf` was requested).

**The "partial obs" panel is illustrative, not per-method-exact.** It is
generated once from `config.yaml`'s defaults (matches all four current
headline checkpoints, which all use `obs_every_k=1`); each method's own
reconstruction below still uses *its own* checkpoint's actual training
config (its own `dT`/`obs_every_k` read from the checkpoint, not the
illustrative panel). This script is for eyeballing reconstructions, not a
substitute for `compare/compare5.py`'s controlled numeric comparison.

The "busy frame" is chosen once, within the first `dT` frames of the day
(4DVarNet only ever reconstructs its first window -- same constraint
`plot_reconstruction_enkf.py` has), as the observed frame with the most
total density. DINCAE and Senseiver are not windowed the same way and could
show any frame; they're evaluated at the same absolute frame index so all
panels depict the same instant.

Usage (GPU node; EnKF alone needs no GPU, it's a numpy file read):
    python3 -m compare.plot_reconstruction --methods dincae,senseiver,varnet,enkf --day atc-20130811
    python3 -m compare.plot_reconstruction --methods dincae --day atc-20130818 --out my_dincae_look.png
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

from methods.varnet.checks.model_io import load_solver
from methods.dincae.checks.evaluate import load_models as dincae_load_models, \
    predict_day as dincae_predict_day, clip_bounds as dincae_clip_bounds
from methods.senseiver.checks.evaluate import load_model as senseiver_load_model, \
    clip_bounds as senseiver_clip_bounds
from methods.senseiver import dataset as sds
from methods.senseiver import sensors as ssensors

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIP = ((0.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (0.0, 2.0))


def clip_np(x):
    x = x.copy()
    for c, (lo, hi) in enumerate(CLIP):
        np.clip(x[c], lo, hi, out=x[c])
    return x


def panel(ax, state, vmax, title):
    """density (Blues) + unit heading arrows -- same style as plot_reconstruction_enkf.py."""
    density, vx, vy = state[0], state[1], state[2]
    heading = np.arctan2(vy, vx)
    x, y = np.meshgrid(np.arange(density.shape[1]), np.arange(density.shape[0]))
    u, v = np.cos(heading), np.sin(heading)
    mask = np.isnan(density)
    u = np.where(mask, np.nan, u); v = np.where(mask, np.nan, v)
    im = ax.imshow(density, cmap="Blues", origin="upper", aspect="equal", vmin=0, vmax=vmax)
    ax.quiver(x, y, u, v, color="black", scale=30, headwidth=3, headlength=4)
    ax.set_title(title, fontsize=13); ax.set_xticks([]); ax.set_yticks([])
    return im


def pick_busy_frame(X, dT):
    """Busiest OBSERVED frame within the first dT frames -- matches
    plot_reconstruction_enkf.py's heuristic exactly, so the two scripts agree
    on which frame to show when both are pointed at the same day."""
    out = om.generate_observations(X[:dT], add_noise=True, valid_mask=nav.build_valid_mask_from_config(X))
    tdens = X[:dT, 0].reshape(dT, -1).sum(1)
    observed = out["Omega"].reshape(dT, -1).sum(1) > 0
    cand = np.where(observed)[0]
    t = int(cand[np.argmax(tdens[cand])]) if len(cand) else int(np.argmax(tdens))
    return t, out


def recon_varnet(ckpt, day_file, t, dev):
    solver, a, _ = load_solver(ckpt, dev)
    dT = a["dT"]
    X, _ = om.load_state(day_file); X = np.asarray(X)
    out = om.generate_observations(X[:dT], add_noise=True, valid_mask=nav.build_valid_mask_from_config(X))
    win = lambda arr: om.to_windows(arr[:dT], dT)
    x0 = om.fill_missing_state(out["Y"], out["Omega_c"], method=config.get("observation", "init_method"))
    with torch.enable_grad():
        xr = solver(torch.from_numpy(win(x0)).float().to(dev),
                    torch.from_numpy(win(out["Y"])).float().to(dev),
                    torch.from_numpy(win(out["Omega_c"].astype(np.float32))).float().to(dev)
                    ).detach().cpu().numpy()[0]
    return clip_np(xr[:, t])


def recon_dincae(run_dir, ckpt, day_file, t, dev):
    from methods.dincae.state import StateStats
    from methods.dincae.encoding import FRESH_OFFSETS
    stats = StateStats()
    # One named checkpoint, not the ckpt_*.pt glob: the glob averages every
    # checkpoint's outputs, which is not the configuration behind any reported
    # DINCAE number (see evaluate.PUBLISHED_CKPT), so the panel would not show
    # the model the tables describe.
    models, epochs, ckpt_paths = dincae_load_models(run_dir, os.path.join(run_dir, ckpt), dev)
    Xt, rec, _, _, _ = dincae_predict_day(models, stats, day_file, dev)
    # predict_day's arrays start at frame `lo` (the smallest FRESH_OFFSETS margin), not 0
    lo = -min(FRESH_OFFSETS)
    idx = t - lo
    if not (0 <= idx < rec.shape[0]):
        raise ValueError(f"frame {t} is outside DINCAE's predictable range for this day "
                          f"(needs {lo} <= t < {lo + rec.shape[0]})")
    return dincae_clip_bounds(rec[idx:idx + 1])[0]


def recon_senseiver(ckpt, day_file, t, dev):
    model, ck = senseiver_load_model(ckpt, dev)
    k = ck["args"].get("obs_every_k") or sds.obs_config()["obs_every_k"]
    C, H, W = sds.state_shape()
    X, Y, Om = sds.load_day(day_file, stride=1, seed=0, frames=t + 1, obs_every_k=k)
    pe = model.pos_enc.detach().cpu().numpy()
    mean = model.in_mean.detach().cpu().numpy()
    std = model.in_std.detach().cpu().numpy()
    tok, pad, _ = ssensors.build_batch(Y[t:t + 1], Om[t:t + 1], pe, mean, std)
    with torch.no_grad():
        xr = model.reconstruct(tok.to(dev), pad.to(dev))
        xr = senseiver_clip_bounds(xr)
    return xr[0].cpu().numpy()


def recon_enkf(enkf_dir, day, t):
    z = np.load(os.path.join(enkf_dir, f"est_{day}.npz"))
    return z["Est"][t], z["Spread"][t, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="dincae,senseiver,varnet,enkf",
                     help="comma-separated, any of: dincae,senseiver,varnet,enkf")
    ap.add_argument("--day", default="atc-20130811")
    ap.add_argument("--frame", type=int, default=-1, help="-1 = auto-pick the busiest observed frame")
    ap.add_argument("--varnet-ckpt", default=os.path.join(paths.method(paths.VARNET), "runs", "varnet_mse5_s0", "varnet_best.pt"))
    ap.add_argument("--dincae-run-dir", default=os.path.join(paths.method(paths.DINCAE), "runs", "dincae_full"))
    ap.add_argument("--dincae-ckpt", default="ckpt_00070.pt",
                    help="the checkpoint every reported DINCAE number uses (evaluate.PUBLISHED_CKPT)")
    ap.add_argument("--senseiver-ckpt", default=os.path.join(paths.method(paths.SENSEIVER), "runs", "senseiver_A", "best.pt"))
    ap.add_argument("--enkf-dir", default=paths.enkf_export("enkf_k1_full"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    day_file = [d for d in om.split_files("test") if args.day in d][0]
    X, _ = om.load_state(day_file); X = np.asarray(X)
    dT = 200                                          # the paper's window; only used to pick a busy frame + drive varnet
    if args.frame >= 0:
        t = args.frame
        obs_out = om.generate_observations(X[:dT], add_noise=True, valid_mask=nav.build_valid_mask_from_config(X))
    else:
        t, obs_out = pick_busy_frame(X, dT)
    true_s = X[t]
    obs_s = obs_out["Y"][t].copy(); obs_s[:, ~obs_out["Omega"][t]] = np.nan

    panels = [("true state", true_s), ("partial obs", obs_s)]
    spread = None
    for m in methods:
        if m == "varnet":
            panels.append(("4DVarNet", recon_varnet(args.varnet_ckpt, day_file, t, dev)))
        elif m == "dincae":
            panels.append(("DINCAE", recon_dincae(args.dincae_run_dir, args.dincae_ckpt, day_file, t, dev)))
        elif m == "senseiver":
            panels.append(("Senseiver", recon_senseiver(args.senseiver_ckpt, day_file, t, dev)))
        elif m == "enkf":
            est, spread = recon_enkf(args.enkf_dir, args.day, t)
            panels.append(("EnKF", est))
        else:
            raise ValueError(f"unknown method {m!r}")

    ncol = len(panels) + (1 if spread is not None else 0)
    vmax = max(0.3, float(np.percentile(true_s[0], 98)))
    fig, axes = plt.subplots(1, ncol, figsize=(3.8 * ncol, 4.2))
    axes = np.atleast_1d(axes)
    for ax, (title, state) in zip(axes, panels):
        im = panel(ax, state, vmax, title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if spread is not None:
        sp = spread.copy(); sp[np.isnan(sp)] = 0
        im = axes[-1].imshow(sp, cmap="Reds", origin="upper", aspect="equal")
        axes[-1].set_title("EnKF spread", fontsize=13); axes[-1].set_xticks([]); axes[-1].set_yticks([])
        fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)

    fig.suptitle(f"{args.day}  frame {t}  —  density (blue) + velocity heading (arrows)", fontsize=13)
    out_p = args.out or os.path.join(paths.COMPARE, "results", f"reconstruction_{args.day}_{'-'.join(methods)}.png")
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    fig.savefig(out_p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[figure] {out_p}  (frame {t})")


if __name__ == "__main__":
    main()
