"""
plot_reconstruction_sequence.py  —  N consecutive reconstructed frames as PNGs
==============================================================================

Reconstruct a block of N CONSECUTIVE frames from one test day with 4DVarNet and
save one PNG per frame: three panels  obs | true | 4DVarNet  (density Blues +
heading arrows, same style as plot_reconstruction_enkf.py, minus the EnKF).

The block is split into dT-frame windows (the model's native window), each
reconstructed independently, then concatenated back to N frames.

Run on a GPU node:
    python3 checks/plot_reconstruction_sequence.py --day atc-20130811 --n 1000
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch
import config, navigation as nav, observation_model as om
from model_io import load_solver

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARROW_MIN = 0.1


def build_solver(ckpt, dev):
    s, a, _ = load_solver(ckpt, dev)
    return s, a


def clip_np(x):                                          # EnKF-consistent physical bounds, on (n,4,H,W)
    x = x.copy()
    x[:, 0] = np.clip(x[:, 0], 0, 5); x[:, 1] = np.clip(x[:, 1], -5, 5)
    x[:, 2] = np.clip(x[:, 2], -5, 5); x[:, 3] = np.clip(x[:, 3], 0, 2)
    return x


def panel(ax, state, vmax):
    """density (Blues) + unit heading arrows (direction), gated on real crowd."""
    dens, vx, vy = state[0], state[1], state[2]
    heading = np.arctan2(vy, vx)
    x, y = np.meshgrid(np.arange(dens.shape[1]), np.arange(dens.shape[0]))
    u, v = np.cos(heading), np.sin(heading)
    m = np.isnan(dens)                                   # only skip cells with NO data; show ALL raw predictions
    u = np.where(m, np.nan, u); v = np.where(m, np.nan, v)
    im = ax.imshow(dens, cmap="Blues", origin="upper", aspect="equal", vmin=0, vmax=vmax)
    ax.quiver(x, y, u, v, color="black", scale=30, headwidth=3, headlength=4)
    ax.set_xticks([]); ax.set_yticks([])
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/varnet_b0_k1/varnet_best.pt")
    ap.add_argument("--day", default="atc-20130811")
    ap.add_argument("--n", type=int, default=1000, help="number of consecutive frames")
    ap.add_argument("--start", type=int, default=-1, help="start frame; -1 = auto-pick busiest block")
    ap.add_argument("--enkf-dir", default="", help="if set, load est_<day>.npz from here and add EnKF + spread panels (PPT style)")
    ap.add_argument("--outdir", default="")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    solver, a = build_solver(args.ckpt, dev); dT = a["dT"]
    N = (args.n // dT) * dT                               # whole number of windows
    outdir = args.outdir or os.path.join(ROOT, "check_outputs", "eval", f"seq_{args.day}")
    os.makedirs(outdir, exist_ok=True)

    day = [d for d in om.split_files("test") if args.day in d][0]
    Xall, _ = om.load_state(day); Xall = np.asarray(Xall)

    # pick the busiest N-frame block (max total density) if start not given
    if args.start < 0:
        dens = Xall[:, 0].reshape(Xall.shape[0], -1).sum(1)
        csum = np.concatenate([[0], np.cumsum(dens)])
        block = csum[N:] - csum[:-N]                      # sliding N-sum
        start = int(np.argmax(block)); start = (start // dT) * dT
    else:
        start = (args.start // dT) * dT
    start = min(start, Xall.shape[0] - N)
    X = Xall[start:start + N]
    print(f"[seq] day={args.day} frames [{start}, {start+N}) = {N} frames, {N//dT} windows", flush=True)

    out = om.generate_observations(X, add_noise=True, valid_mask=nav.build_valid_mask_from_config(X))
    Omega = out["Omega"]
    x0 = om.fill_missing_state(out["Y"], out["Omega_c"], method=config.get("observation", "init_method"))

    win = lambda arr: om.to_windows(arr[:N], dT)            # (N,C,H,W) -> (n_win, C, dT, H, W)
    yb = torch.from_numpy(win(out["Y"])).float()
    mb = torch.from_numpy(win(out["Omega_c"].astype(np.float32))).float()
    x0b = torch.from_numpy(win(x0)).float()
    recs = []
    with torch.enable_grad():
        for i in range(x0b.shape[0]):                    # one window at a time (memory-safe)
            r = solver(x0b[i:i+1].to(dev), yb[i:i+1].to(dev), mb[i:i+1].to(dev)).detach().cpu()
            recs.append(r)
    xr = torch.cat(recs, 0).numpy()                      # (n_win, C, dT, H, W)
    xr = clip_np(om.from_windows(xr)[:N])                   # (N,4,36,12)

    vmax = max(0.3, float(np.percentile(X[:, 0], 99)))   # shared density scale across all frames
    obs = out["Y"].copy()

    # optional EnKF panels (PPT-style): load est_<day>.npz for the SAME block
    Est = Spread = None
    if args.enkf_dir:
        z = np.load(os.path.join(args.enkf_dir, f"est_{args.day}.npz"))
        Est, Spread = z["Est"], z["Spread"]
        nE = min(N, Est.shape[0]); print(f"[seq] EnKF panels from {args.enkf_dir} ({Est.shape[0]} frames)", flush=True)

    ncol = 5 if Est is not None else 3
    fig, axes = plt.subplots(1, ncol, figsize=(3.1 * ncol, 4.4))
    for t in range(N):
        obs_t = obs[t].copy(); obs_t[:, ~Omega[t]] = np.nan
        cells = [("partial obs", obs_t), ("true state", X[t]), ("4DVarNet", xr[t])]
        if Est is not None and t < Est.shape[0]:
            cells.append(("EnKF", clip_np(Est[t:t+1])[0]))
        for ax, (ti, s) in zip(axes, cells):
            ax.cla(); panel(ax, s, vmax); ax.set_title(ti, fontsize=11)
        if Est is not None:                              # 5th panel: EnKF density spread (Reds)
            axes[4].cla()
            sp = Spread[t, 0].copy() if t < Spread.shape[0] else np.zeros((36, 12))
            sp[np.isnan(sp)] = 0
            axes[4].imshow(sp, cmap="Reds", origin="upper", aspect="equal")
            axes[4].set_title("EnKF spread", fontsize=11); axes[4].set_xticks([]); axes[4].set_yticks([])
        obs_flag = "OBS" if Omega[t].any() else "no obs (t%4≠0)"
        fig.suptitle(f"{args.day}  frame {start+t}  (seq {t:04d}/{N})   [{obs_flag}]", fontsize=11)
        fig.savefig(os.path.join(outdir, f"frame_{t:04d}.png"), dpi=100, bbox_inches="tight")
        if t % 100 == 0:
            print(f"  rendered {t}/{N}", flush=True)
    plt.close(fig)
    print(f"[done] {N} PNGs ({ncol} panels) -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
