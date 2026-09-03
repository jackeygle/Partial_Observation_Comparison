"""
plot_nav_mask.py  —  walkable mask (current rule) drawn on the real ATC map
===========================================================================

Regenerates check_outputs/navigation/nav_mask_on_map.png so it always reflects the
CURRENT rule in navigation.build_valid_mask_from_map (map kept at the same 1 m
resolution as the state; obstacles solidified; a cell is a wall iff >=50% of it is
obstacle pixels; largest connected component). Green = walkable cells (robots drive
here); red = wall cells (>=50% obstacle). Green should trace the free corridor,
red should sit on the mall walls/stalls.

Run:
    module load scicomp-pytorch-env/2026.1
    python3 checks/plot_nav_mask.py
"""

from __future__ import annotations

import os
import sys


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
import numpy as np

from crowdcore import navigation as nav
from crowdcore.navigation import load_atc_map, world_to_map_pixel
from crowdcore.data.h5_to_grid import SUBSETS, rotation_matrix


def main():
    outdir = "check_outputs/navigation"
    os.makedirs(outdir, exist_ok=True)

    walk = nav.build_valid_mask_from_config()              # (H,W) HYBRID rule (train-union data AND map footprint)
    wall = nav._map_wall("corridor", 1.0)                  # map wall (connected footprint; blocks driving AND sight)

    img, mres, morg = load_atc_map()
    s = SUBSETS["corridor"]; O = np.asarray(s["origin"]); th = s["theta"]; H, W = s["shape"]
    R = rotation_matrix(th)

    # crop: from the cell CENTRES (as before) so the extent matches obstacle_map.png
    ci, cj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    cworld = np.stack([ci.ravel() + 0.5, cj.ravel() + 0.5], 1) @ R.T + O
    crow, ccol = world_to_map_pixel(cworld, img.shape, mres, morg)
    pad = 150
    r0, r1 = max(crow.min() - pad, 0), min(crow.max() + pad, img.shape[0])
    c0, c1 = max(ccol.min() - pad, 0), min(ccol.max() + pad, img.shape[1])

    # build one filled quad per 1 m cell (rotated corners projected to map pixels)
    GREEN = (0.18, 0.62, 0.27, 0.55); RED = (0.85, 0.20, 0.20, 0.60); NONE = (0, 0, 0, 0)
    polys, faces = [], []
    for i in range(H):
        for j in range(W):
            corners = np.array([[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]], float)
            w = corners @ R.T + O
            rw, cw = world_to_map_pixel(w, img.shape, mres, morg)
            polys.append(np.stack([cw - c0, rw - r0], 1))    # (x=col, y=row) in cropped-image coords
            faces.append(RED if wall[i, j] else (GREEN if walk[i, j] else NONE))

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(img[r0:r1, c0:c1], cmap="gray", vmin=0, vmax=255)
    ax.add_collection(PolyCollection(polys, facecolors=faces, edgecolors=(1, 1, 1, 0.5), linewidths=0.4))
    ax.legend(handles=[Patch(facecolor=GREEN, edgecolor="w", label=f"walkable — {int(walk.sum())}"),
                       Patch(facecolor=RED, edgecolor="w", label=f"map wall — {int(wall.sum())}")],
              loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_title("walkable mask (filled 1 m cells) — HYBRID: pedestrian ever present AND ≥50% obstacle rule\n"
                 f"{int(walk.sum())}/{walk.size} walkable (1 m grid = state resolution, largest component)",
                 fontsize=10)
    ax.set_xlim(0, c1 - c0); ax.set_ylim(r1 - r0, 0)       # keep image orientation
    ax.set_xticks([]); ax.set_yticks([])
    out = os.path.join(outdir, "nav_mask_on_map.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[figure] {out}  (walkable {int(walk.sum())}, wall {int(wall.sum())})")


if __name__ == "__main__":
    main()
