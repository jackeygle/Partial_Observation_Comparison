"""
plot_reconstruction_enkf.py  —  reconstruction in the EnKF project's own style
==============================================================================

Matches Partial_observation/utils.py:plot_generated_matrix_on_ax / save_step_plot so
our figures sit next to the EnKF baseline's with no visual mismatch:

    partial obs | true | 4DVarNet | EnKF | EnKF spread

Each of the first four panels: DENSITY as a Blues background + VELOCITY as black
heading arrows (unit vectors from vx, vy — direction only, exactly like the EnKF
project). The spread panel: EnKF density-channel ensemble spread (Reds). One busy
held-out frame; per-panel colourbars.

Run on a GPU node:
    python3 checks/plot_reconstruction_enkf.py --day atc-20130811
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

# 这个脚本原来住在 4dvarnet_enkf/ 下，ROOT 一直指那个目录（runs/、check_outputs/
# 都挂在它下面）。2026-09-03 重构后它搬到了顶层，dirname(dirname(__file__)) 会变成
# 仓库根，于是每一条 os.path.join(ROOT, ...) 都会静默指错地方 —— 所以显式绑定。
ROOT = paths.method(paths.VARNET)
ENKFDIR = paths.enkf_export("enkf")
ARROW_MIN = 0.1                                           # draw a heading arrow only where there is real crowd


def plot_matrix(ax, state, vmax):
    """EnKF-style: density (Blues) + unit heading arrows (direction only)."""
    density, vx, vy = state[0], state[1], state[2]
    heading = np.arctan2(vy, vx)
    x, y = np.meshgrid(np.arange(density.shape[1]), np.arange(density.shape[0]))
    u, v = np.cos(heading), np.sin(heading)
    mask = np.isnan(density)                              # only skip cells with NO data (unobserved); show ALL raw predictions
    u = np.where(mask, np.nan, u); v = np.where(mask, np.nan, v)
    im = ax.imshow(density, cmap="Blues", origin="upper", aspect="equal", vmin=0, vmax=vmax)
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
    z = np.load(os.path.join(ENKFDIR, f"est_{args.day}.npz")); Est, Spread = z["Est"], z["Spread"]

    tdens = X[:dT, 0].reshape(dT, -1).sum(1); obs = Omega[:dT].reshape(dT, -1).sum(1) > 0
    cand = np.where(obs)[0]; t = int(cand[np.argmax(tdens[cand])]) if len(cand) else int(np.argmax(tdens))

    win = lambda arr: om.to_windows(arr[:dT], dT)           # first window only
    x0 = om.fill_missing_state(out["Y"], out["Omega_c"], method=config.get("observation", "init_method"))
    with torch.enable_grad():
        xr = solver(torch.from_numpy(win(x0)).float().to(dev), torch.from_numpy(win(out["Y"])).float().to(dev),
                    torch.from_numpy(win(out["Omega_c"].astype(np.float32))).float().to(dev)).detach().cpu().numpy()[0]
    var_s = clip_np(xr[:, t]); true_s, enkf_s = X[t], Est[t]
    obs_s = out["Y"][t].copy(); obs_s[:, ~Omega[t]] = np.nan
    vmax = max(0.3, float(np.percentile(true_s[0], 98)))

    fig, axes = plt.subplots(1, 5, figsize=(19, 5))
    for ax, (ti, s) in zip(axes[:4], [("partial obs", obs_s), ("true state", true_s),
                                      ("4DVarNet", var_s), ("EnKF", enkf_s)]):
        im = plot_matrix(ax, s, vmax); ax.set_title(ti, fontsize=13)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    sp = Spread[t, 0].copy(); sp[np.isnan(sp)] = 0
    im4 = axes[4].imshow(sp, cmap="Reds", origin="upper", aspect="equal")
    axes[4].set_title("EnKF spread", fontsize=13); axes[4].set_xticks([]); axes[4].set_yticks([])
    fig.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)
    fig.suptitle(f"{args.day}  frame {t}  —  density (blue) + velocity heading (arrows);  EnKF-baseline style",
                 fontsize=13)
    out_p = os.path.join(ROOT, args.outdir, f"reconstruction_enkf_{args.day}.png")
    fig.savefig(out_p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[figure] {out_p}  (frame {t})")


if __name__ == "__main__":
    main()
