"""
plot_obstacle_map.py  —  solid obstacle occupancy on the real ATC map
=====================================================================

Companion to plot_nav_mask.py. That figure shows the RESULT (walkable vs wall
cells); this one shows the INPUT the supervisor's 50% rule is measured against:
the solid obstacle mask (ROS OCCUPIED ∪ enclosed-UNKNOWN, i.e. pillar/stall
interiors filled) at the map's own 0.05 m/px resolution, drawn on the real map
in the SAME crop / projection as nav_mask_on_map.png so the two sit side by side.

    red overlay = obstacle pixels;  gray = the raw ATC laser map.

Run:
    module load scicomp-pytorch-env/2026.1
    python3 checks/plot_obstacle_map.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import navigation as nav
from navigation import load_atc_map, world_to_map_pixel
from data_pipeline.h5_to_grid import SUBSETS, rotation_matrix


def main():
    outdir = "check_outputs/navigation"
    os.makedirs(outdir, exist_ok=True)

    obst = nav._obstacle_pixels(solid=True)                # (Hpx,Wpx) bool solid obstacle mask

    img, mres, morg = load_atc_map()
    s = SUBSETS["corridor"]; O = np.asarray(s["origin"]); th = s["theta"]; H, W = s["shape"]
    R = rotation_matrix(th)
    ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    world = np.stack([ii.ravel() + 0.5, jj.ravel() + 0.5], 1) @ R.T + O
    rows, cols = world_to_map_pixel(world, img.shape, mres, morg)
    pad = 150                                              # SAME crop as plot_nav_mask.py
    r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad, img.shape[0])
    c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad, img.shape[1])

    crop_img = img[r0:r1, c0:c1]
    crop_obst = obst[r0:r1, c0:c1]
    overlay = np.zeros((*crop_obst.shape, 4))              # transparent RGBA, red where obstacle
    overlay[crop_obst] = [0.85, 0.10, 0.10, 0.55]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(crop_img, cmap="gray", vmin=0, vmax=255)
    ax.imshow(overlay)
    frac = crop_obst.mean()
    ax.set_title("solid obstacle occupancy (ROS: OCCUPIED ∪ enclosed-UNKNOWN)\n"
                 "the 50% rule is measured per 1 m cell against THIS mask "
                 f"({frac:.0%} of the crop is obstacle)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    out = os.path.join(outdir, "obstacle_map.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[figure] {out}  (obstacle pixels {int(obst.sum())})")


if __name__ == "__main__":
    main()
