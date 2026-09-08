"""
plot_velocity_enkf.py  —  velocity-focused figure: speed magnitude + direction
===============================================================================

A companion to plot_reconstruction_enkf.py (which is density-focused). Here the
BACKGROUND colour is the velocity MAGNITUDE |v| = sqrt(vx^2 + vy^2) and the black
arrows show DIRECTION (heading). So both the speed (colour) and the direction
(arrow) of the velocity can be read off and compared method by method — you can
see whether the reconstructed velocity is accurate in magnitude AND direction,
which the density-background figure (unit arrows) cannot show.

    partial obs | true | 4DVarNet | EnKF | EnKF velocity-spread

One busy held-out frame; a shared speed colour scale across the four field panels.

Run on a GPU node:
    python3 checks/plot_velocity_enkf.py --day atc-20130811
"""
from __future__ import annotations
import argparse, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch
from crowdcore import config
from crowdcore import navigation as nav
from crowdcore import observation_model as om
from crowdcore import paths
from methods.varnet.checks.model_io import load_solver

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENKFDIR = paths.enkf_export()   # default: enkf_k1_full, the model of record
ARROW_MIN = 0.1                                           # arrows only where there is real crowd


def speed_panel(ax, state, vmax):
    """EnKF-style: speed |v| as a Blues background + black unit heading arrows.

    Identical style to plot_reconstruction_enkf.py (Blues + black arrows); the only
    difference is the background colour encodes speed magnitude, not density.
    """
    dens, vx, vy = state[0], state[1], state[2]
    speed = np.sqrt(vx ** 2 + vy ** 2)
    speed_m = np.where(np.isnan(dens), np.nan, speed)
    heading = np.arctan2(vy, vx)
    x, y = np.meshgrid(np.arange(dens.shape[1]), np.arange(dens.shape[0]))
    u, v = np.cos(heading), np.sin(heading)
    mask = np.isnan(dens)                                # only skip cells with NO data; show ALL raw predictions
    u = np.where(mask, np.nan, u); v = np.where(mask, np.nan, v)
    im = ax.imshow(speed_m, cmap="Blues", origin="upper", aspect="equal", vmin=0, vmax=vmax)
    ax.quiver(x, y, u, v, color="black", scale=30, headwidth=3, headlength=4)
    ax.set_xticks([]); ax.set_yticks([])
    return im


def build_solver(ckpt, dev):
    s, a, _ = load_solver(ckpt, dev)
    return s, a


def clip_np(x):
    x = x.copy(); x[0] = np.clip(x[0], 0, 5); x[1] = np.clip(x[1], -5, 5); x[2] = np.clip(x[2], -5, 5); x[3] = np.clip(x[3], 0, 2); return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/varnet_b0_k1/varnet_best.pt")
    ap.add_argument("--day", default="atc-20130811")
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--outdir", default="check_outputs/eval")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    solver, a = build_solver(args.ckpt, dev); dT = a["dT"]

    day = [d for d in om.split_files("test") if args.day in d][0]
    X, _ = om.load_state(day); X = np.asarray(X)[:args.frames]
    out = om.generate_observations(X, add_noise=True, valid_mask=nav.build_valid_mask_from_config(X))
    Omega = out["Omega"]
    Est = np.load(os.path.join(ENKFDIR, f"est_{args.day}.npz"))["Est"]   # spread not shown (matches original: density-only)

    tdens = X[:dT, 0].reshape(dT, -1).sum(1); obs = Omega[:dT].reshape(dT, -1).sum(1) > 0
    cand = np.where(obs)[0]; t = int(cand[np.argmax(tdens[cand])]) if len(cand) else int(np.argmax(tdens))

    win = lambda arr: om.to_windows(arr[:dT], dT)           # first window only
    x0 = om.fill_missing_state(out["Y"], out["Omega_c"], method=config.get("observation", "init_method"))
    with torch.enable_grad():
        xr = solver(torch.from_numpy(win(x0)).float().to(dev), torch.from_numpy(win(out["Y"])).float().to(dev),
                    torch.from_numpy(win(out["Omega_c"].astype(np.float32))).float().to(dev)).detach().cpu().numpy()[0]
    var_s = clip_np(xr[:, t]); true_s, enkf_s = X[t], Est[t]
    obs_s = out["Y"][t].copy(); obs_s[:, ~Omega[t]] = np.nan

    # shared speed scale from true / 4DVarNet / EnKF
    sp_all = [np.sqrt(s[1] ** 2 + s[2] ** 2) for s in (true_s, var_s, enkf_s)]
    vmax = max(np.percentile(np.concatenate([s.ravel() for s in sp_all]), 98), 0.2)

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 5))
    for ax, (ti, s) in zip(axes, [("partial obs", obs_s), ("true state", true_s),
                                  ("4DVarNet", var_s), ("EnKF", enkf_s)]):
        im = speed_panel(ax, s, vmax); ax.set_title(ti, fontsize=13)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("speed |v|")
    fig.suptitle(f"{args.day}  frame {t}  —  velocity: colour = speed |v|, arrows = direction",
                 fontsize=13)
    out_p = os.path.join(ROOT, args.outdir, f"velocity_enkf_{args.day}.png")
    fig.savefig(out_p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[figure] {out_p}  (frame {t})")


if __name__ == "__main__":
    main()
