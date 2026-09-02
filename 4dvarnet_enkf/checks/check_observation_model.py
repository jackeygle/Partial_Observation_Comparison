"""
check_observation_model.py  —  verification, diagnostics & figures for Component 1
==================================================================================

This is NOT part of the data pipeline. It only *exercises* `observation_model.py`:
  - runs numerical assertions ("verify it works"),
  - prints/writes the I/O report,
  - renders the inspection figures (per-frame triptych + coverage-vs-range curve).

The functional module `observation_model.py` has no dependency on this file.

Run:
    module load scicomp-pytorch-env/2026.1
    python3 checks/check_observation_model.py --file first --frames 60 --init prev
"""

from __future__ import annotations

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make parent-level modules importable

import numpy as np

import navigation
from observation_model import (
    CHANNEL_NAMES, NUM_AGENTS, SENSING_RANGE,
    fill_missing_state, generate_observations, load_state, resolve_file,
)


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def coverage_stats(Omega, valid_mask=None):
    """Fraction of WALKABLE cells observed: overall mean and per-frame min/max.

    Coverage is measured over the walkable corridor (valid_mask), not the whole
    36x12 box — wall / outside-corridor cells are never observable, so including
    them would understate coverage. If valid_mask is None, falls back to all cells.
    """
    T = Omega.shape[0]
    if valid_mask is None:
        valid_mask = np.ones(Omega.shape[1:], dtype=bool)
    n_valid = int(valid_mask.sum())
    # observed walkable cells per frame / total walkable cells
    obs_in_valid = (Omega & valid_mask[None]).reshape(T, -1).sum(axis=1)
    per_frame = obs_in_valid / max(n_valid, 1)
    return {
        "mean_coverage": float(per_frame.mean()),
        "min_coverage": float(per_frame.min()),
        "max_coverage": float(per_frame.max()),
        "n_cells": n_valid,
    }


def io_report(X, out):
    """Human-readable summary of shapes/dtypes/ranges — the I/O documentation."""
    vm = out["valid_mask"]
    cov = coverage_stats(out["Omega"], vm)
    lines = ["=== observation_model I/O report ===",
             f"X        {X.shape} {X.dtype}   full state (T,C,H,W)",
             f"Y        {out['Y'].shape} {out['Y'].dtype}   noisy partial observation",
             f"Y_clean  {out['Y_clean'].shape} {out['Y_clean'].dtype}",
             f"Omega    {out['Omega'].shape} {out['Omega'].dtype}   observation mask (spatial)",
             f"Omega_c  {out['Omega_c'].shape} {out['Omega_c'].dtype}   per-channel mask",
             f"valid    {vm.shape} {vm.dtype}   walkable mask ({int(vm.sum())}/{vm.size} cells)",
             f"positions{out['positions'].shape} {out['positions'].dtype}   robot (row,col)/sec",
             "",
             f"channels = {CHANNEL_NAMES}",
             f"coverage : mean={cov['mean_coverage']:.3%}  "
             f"min={cov['min_coverage']:.3%}  max={cov['max_coverage']:.3%}  "
             f"(of {cov['n_cells']} walkable cells)",
             "per-channel value range (full state X):"]
    for c, name in enumerate(CHANNEL_NAMES):
        lines.append(f"    {name:8s} [{X[:, c].min():+.4f}, {X[:, c].max():+.4f}]")
    return "\n".join(lines), cov


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
def plot_frame(step_idx, x_gt, obs, obs_mask, positions, valid_mask, predictions, save_path):
    """Plot one frame: GT | robots+observed area | partial obs | each method.

    Density-panel style (Blues + velocity quiver + colorbar) is taken from the
    original project (Partial_observation_new/my_solution/run_comparison.py).
    Added on top: a panel showing the robot positions and the cells they observe
    (the mask Ω), so the partial-observation process is visible.
    `positions` is (num_agents, 2) int (row, col); `predictions` is an ordered
    dict {label: state (C,H,W)} — here we pass the init X0.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n_methods = len(predictions)
    n_cols = n_methods + 3                                # GT + Robots + PartialObs + methods
    fig = plt.figure(figsize=(5 * n_cols, 5.5))
    gs = gridspec.GridSpec(1, n_cols, figure=fig, wspace=0.25)

    # Choose vmax from GT density (so all density panels share the same scale)
    vmax = max(float(np.nanmax(x_gt[0])), 0.1)

    def _plot_state(ax, state, title):
        density = state[0].copy()
        vx, vy = state[1], state[2]
        speed = np.sqrt(vx**2 + vy**2)
        H, W = density.shape
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        u = np.where(speed < 1e-6, np.nan, vx / (speed + 1e-9))
        v = np.where(speed < 1e-6, np.nan, vy / (speed + 1e-9))
        nan_mask = np.isnan(density)
        u[nan_mask] = np.nan
        v[nan_mask] = np.nan
        density[nan_mask] = 0.0
        im = ax.imshow(density, cmap="Blues", aspect="equal", origin="upper",
                       vmin=0, vmax=vmax)
        ax.quiver(x, y, u, v, color="black", scale=30, headwidth=3, headlength=4)
        rho_min = float(np.nanmin(state[0])) if np.any(~np.isnan(state[0])) else 0.0
        rho_max = float(np.nanmax(state[0])) if np.any(~np.isnan(state[0])) else 0.0
        ax.set_title(f"{title}\nρ∈[{rho_min:.2f}, {rho_max:.2f}]", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("ρ", fontsize=9)
        return im

    def _plot_robots(ax, obs2d, valid2d, pos, title):
        """Obstacle(gray) / walkable-unobserved(white) / observed(green) + robots(red)."""
        from matplotlib.colors import ListedColormap
        cat = np.where(~valid2d, 0, np.where(obs2d, 2, 1))     # 0=wall 1=walkable-unobserved 2=observed
        ax.imshow(cat, cmap=ListedColormap(["#9e9e9e", "#ffffff", "#2e7d32"]),
                  aspect="equal", origin="upper", vmin=0, vmax=2)
        ax.scatter(pos[:, 1], pos[:, 0], c="red", s=120, marker="X",
                   edgecolors="black", linewidths=0.8, zorder=3)
        n_valid = int(valid2d.sum())
        cov = 100.0 * (obs2d & valid2d).sum() / max(n_valid, 1)
        ax.set_title(f"{title}\nobserved {cov:.0f}% of corridor", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # 1) Ground truth (full field)
    ax_gt = fig.add_subplot(gs[0])
    _plot_state(ax_gt, x_gt, f"Step {step_idx}\nGround Truth")

    # 2) Robots + observed area (obstacle-aware partial-observation process)
    ax_rob = fig.add_subplot(gs[1])
    _plot_robots(ax_rob, obs_mask[0] >= 0.5, valid_mask, positions, "Robots + observed area")

    # 3) Partial observation (what the robots actually measure)
    ax_obs = fig.add_subplot(gs[2])
    obs_vis = obs.copy()
    obs_vis[:, obs_mask[0] < 0.5] = np.nan
    _plot_state(ax_obs, obs_vis, "Partial Obs")

    # 4) Each method (here: the init X0)
    for i, (name, x_hat) in enumerate(predictions.items()):
        ax = fig.add_subplot(gs[i + 3])
        _plot_state(ax, x_hat, name)

    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_coverage_curve(results_by_range, save_path):
    """Mean coverage vs sensing range — the number the supervisor asked about."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ranges = sorted(results_by_range)
    cov = [results_by_range[r] for r in ranges]
    fig, axp = plt.subplots(figsize=(5, 3.2))
    axp.plot(ranges, [c * 100 for c in cov], "o-")
    axp.axhline(50, ls="--", c="gray", lw=1)
    axp.set_xlabel("sensing range (cells)"); axp.set_ylabel("mean coverage (%)")
    axp.set_title("coverage vs sensing range (3 robots)")
    for r, c in zip(ranges, cov):
        axp.annotate(f"{c:.0%}", (r, c * 100), textcoords="offset points", xytext=(0, 6), ha="center")
    fig.tight_layout(); fig.savefig(save_path, dpi=110); plt.close(fig)


# --------------------------------------------------------------------------- #
# Self-check
# --------------------------------------------------------------------------- #
def self_check(path, frames, sensing_range, num_agents, init, outdir, seed,
               plot_start=None, plot_count=6):
    # Clear only THIS script's own stale outputs (frame_*.png) — do NOT rmtree the whole
    # dir, in case other scripts share it and would lose their figures.
    os.makedirs(outdir, exist_ok=True)
    import glob as _glob
    for _f in _glob.glob(os.path.join(outdir, "frame_*.png")):
        os.remove(_f)
    X, time = load_state(path)                            # load the full state
    # Derive the walkable region from the FULL day (a fixed property of the corridor,
    # independent of how many frames are loaded), then slice for the demo
    valid_mask = navigation.build_valid_mask_from_config(X)   # hybrid: data AND real map (config)
    if frames:
        X = X[:frames]                                    # keep only the first N frames to speed up the self-check
    print(f"[load] {os.path.basename(path)}  X={X.shape}  ({time[-1] - time[0]:.0f}s span)  "
          f"walkable {int(valid_mask.sum())}/{valid_mask.size}")

    out = generate_observations(X, sensing_range, num_agents, add_noise=True, seed=seed,
                                valid_mask=valid_mask)
    X0 = fill_missing_state(out["Y"], out["Omega_c"], method=init)

    # --- Numeric assertions: this is the "verify it is actually correct" step;
    #     any failure raises and stops the run -----------------------------------
    mc = out["Omega_c"]                                   # (T,C,H,W) per-channel observation mask
    # Assertion 1: noise-free observations must equal the ground truth X on observed (channel, cell) slots
    assert np.allclose(out["Y_clean"][mc], X[mc]), "Y_clean must equal X on observed cells"
    # Assertion 2: unobserved (channel, cell) slots must all be 0
    assert (out["Y_clean"][~mc] == 0).all(), "Y must be 0 off the mask"
    # Assertion 3: the init X0 must keep observed values on observed slots (only fill unobserved ones)
    assert np.allclose(X0[mc], out["Y"][mc]), "X0 must keep observed values"
    n_obs = int(out["Omega"].sum())                       # total observed cells (spatial, across all frames)
    print(f"[verify] all assertions passed ({n_obs} observed cell-slots across {X.shape[0]} frames)")

    report, cov = io_report(X, out)
    print("\n" + report)
    with open(os.path.join(outdir, "io_report.txt"), "w") as f:
        f.write(report + "\n")

    # --- figures -------------------------------------------------------------
    try:
        # Plot a run of CONSECUTIVE frames (default: middle of the sequence, where crowd
        # activity is high and the 'prev' history has accumulated enough)
        T = X.shape[0]
        start = plot_start if plot_start is not None else max(0, T // 2 - plot_count // 2)
        start = max(0, min(start, T - plot_count))        # clamp to bounds
        for t in range(start, min(start + plot_count, T)):
            plot_frame(t, X[t], out["Y_clean"][t], out["Omega"][t][None],
                       out["positions"][t], out["valid_mask"],
                       {f"Init X0 ({init})": X0[t]},
                       os.path.join(outdir, f"frame_{t:03d}.png"))
        # coverage vs range sweep (5 vs 10 etc. — what the supervisor asked)
        sweep = {}
        for r in (3, 5, 7, 10):
            o = generate_observations(X, r, num_agents, add_noise=False, seed=seed,
                                      valid_mask=valid_mask)
            sweep[r] = coverage_stats(o["Omega"], valid_mask)["mean_coverage"]
        plot_coverage_curve(sweep, os.path.join(outdir, "coverage_vs_range.png"))
        print(f"\n[figures] written to {outdir}/  "
              f"(coverage sweep: " + ", ".join(f"r{r}={c:.0%}" for r, c in sweep.items()) + ")")
        cov["sweep"] = sweep
    except Exception as e:                               # never block the numeric check on plotting
        print(f"[figures] skipped ({type(e).__name__}: {e})")

    with open(os.path.join(outdir, "stats.json"), "w") as f:
        json.dump(cov, f, indent=2)
    return out, X0


def main():
    ap = argparse.ArgumentParser(description="Verify & visualise Component 1 (partial-observation generation)")
    ap.add_argument("--file", default="first", help="grid_cache path, date stem, or 'first'")
    ap.add_argument("--frames", type=int, default=60, help="limit frames for the check (0 = all)")
    ap.add_argument("--sensing-range", type=int, default=SENSING_RANGE)
    ap.add_argument("--num-agents", type=int, default=NUM_AGENTS)
    ap.add_argument("--init", default="prev", choices=["zeros", "prev", "interp"])
    ap.add_argument("--outdir", default="check_outputs/observation_model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot-start", type=int, default=None,
                    help="first frame to plot (consecutive); default = middle of the sequence")
    ap.add_argument("--plot-count", type=int, default=6,
                    help="how many consecutive frames to plot")
    args = ap.parse_args()
    path = resolve_file(args.file)
    self_check(path, args.frames or None, args.sensing_range, args.num_agents,
               args.init, args.outdir, args.seed,
               plot_start=args.plot_start, plot_count=args.plot_count)


if __name__ == "__main__":
    main()
