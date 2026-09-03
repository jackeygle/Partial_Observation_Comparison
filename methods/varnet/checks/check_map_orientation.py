"""
check_map_orientation.py  —  is the real map aligned with the data, or flipped/rotated?
=======================================================================================

Decisive test for the orientation bug the supervisor flagged. Pedestrians walk in
FREE space, so the all-day accumulated crowd density (from the data) must land on
the FREE part of the real laser map, NOT on obstacles. If the coordinate transform
(theta / origin / row-flip) is wrong, density will sit on walls -> orientation bug.

Two checks:
  (1) numeric — Pearson correlation between per-cell density and per-cell obstacle
      fraction over all 432 cells. Correct alignment => NEGATIVE (people avoid walls).
  (2) visual — overlay the accumulated density on the real map image at the pixel
      positions our transform computes; density should light up the free corridor.

Run:
    module load scicomp-pytorch-env/2026.1
    python3 checks/check_map_orientation.py     # -> check_outputs/navigation/
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import navigation as nav
from navigation import load_atc_map, map_obstacle_fraction, world_to_map_pixel
from data_pipeline.h5_to_grid import SUBSETS, rotation_matrix
from observation_model import load_state, resolve_file


def main():
    ap = argparse.ArgumentParser(description="Check real-map orientation vs data density")
    ap.add_argument("--file", default="first")
    ap.add_argument("--outdir", default="check_outputs/navigation")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # all-day accumulated crowd density in the 36x12 grid frame
    X, _ = load_state(resolve_file(args.file))
    density = X[:, 0].sum(axis=0)                          # (H, W) — where people were, over the whole day
    occ = map_obstacle_fraction()                          # (H, W) — obstacle fraction per cell (real map)

    # (1) numeric: correlation between density and obstacle fraction
    d, o = density.ravel(), occ.ravel()
    corr = float(np.corrcoef(d, o)[0, 1])
    # mean obstacle fraction weighted by density: "on average, how blocked is a cell
    # where people actually are?" — should be LOW if aligned
    wocc = float((d * o).sum() / d.sum())
    print(f"[numeric] corr(density, obstacle_fraction) = {corr:+.3f}  "
          f"(expect NEGATIVE if aligned: people avoid walls)")
    print(f"[numeric] density-weighted mean obstacle fraction = {wocc:.1%}  "
          f"(expect LOW if aligned)")
    print(f"[numeric] verdict: {'ALIGNED ✓' if corr < 0 else 'MISALIGNED ✗ (density sits on obstacles!)'}")

    # (2) visual: overlay density on the real map at computed pixel positions
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe
        img, mres, morg = load_atc_map()
        s = SUBSETS["corridor"]
        origin, theta, (H, W) = np.asarray(s["origin"]), s["theta"], s["shape"]
        R = rotation_matrix(theta)

        # grid cell centres -> world -> map pixel
        ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        local = np.stack([ii.ravel() + 0.5, jj.ravel() + 0.5], axis=1)
        world = local @ R.T + origin
        rows, cols = world_to_map_pixel(world, img.shape, mres, morg)

        # crop to corridor vicinity
        pad = 200
        r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad, img.shape[0])
        c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad, img.shape[1])

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        # left: map + density scatter (only cells with people), colour = density
        ax = axes[0]
        ax.imshow(img[r0:r1, c0:c1], cmap="gray", vmin=0, vmax=255)
        dvals = density.ravel()
        m = dvals > dvals.max() * 0.02                     # cells with meaningful presence
        sc = ax.scatter(cols[m] - c0, rows[m] - r0, c=dvals[m], cmap="hot",
                        s=18, alpha=0.8)
        fig.colorbar(sc, ax=ax, fraction=0.03).set_label("accumulated density")

        # --- overlay the COORDINATE SYSTEM so the rotation is explicit ------------
        # grid origin (world corridor origin) -> pixel, in the cropped frame
        o_row, o_col = world_to_map_pixel(origin, img.shape, mres, morg)
        gx, gy = float(o_col - c0), float(o_row - r0)

        def wdir_to_pix(dx, dy):        # world direction (metres) -> pixel delta (dcol, drow)
            return dx / mres, -dy / mres    # col grows with +x; row grows DOWN, so +y -> -row

        c_, s_ = np.cos(theta), np.sin(theta)
        arrows = [                       # (world dir, length_m, colour, label)
            (wdir_to_pix(c_, s_),  14, "#00e5ff", f"grid  i  (H, 36 rows)"),   # grid rows axis
            (wdir_to_pix(-s_, c_),  9, "#76ff03", f"grid  j  (W, 12 cols)"),   # grid cols axis
            (wdir_to_pix(1, 0),     6, "#ffd600", "world +X"),                 # world x
            (wdir_to_pix(0, 1),     6, "#ff9100", "world +Y"),                 # world y
        ]
        for (dcol, drow), L, col, lab in arrows:
            ax.annotate("", xy=(gx + dcol * L, gy + drow * L), xytext=(gx, gy),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2))
            ax.text(gx + dcol * L * 1.08, gy + drow * L * 1.08, lab, color=col,
                    fontsize=8, fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")])
        ax.plot(gx, gy, "o", color="white", ms=7, mec="black", zorder=5)
        ax.text(gx + 6, gy + 6, f"grid origin\n(38.28, -15.81) m\nθ = {np.degrees(theta):.0f}°",
                color="white", fontsize=8,
                path_effects=[pe.withStroke(linewidth=2, foreground="black")])

        ax.set_title("(a) data density on real map + coordinate system\n"
                     "grid (i,j) axes are the world axes rotated by θ; hot spots fall in the FREE corridor",
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        # right: two grid-frame heatmaps side by side for shape comparison
        ax = axes[1]
        ax.imshow(density, cmap="hot", aspect="equal", origin="upper")
        ax.set_title("(b) accumulated density (grid frame, 36x12)\n"
                     "compare its shape with the corridor in (a)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        out = os.path.join(args.outdir, "map_orientation_check.png")
        fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
        print(f"[figure] {out}")
    except Exception as e:
        print(f"[figure] skipped ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
