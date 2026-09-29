"""Thesis figure: Senseiver-A (the paper's model) and Senseiver-G (ours).

    python3 -m thesis_figures.senseiver         # -> thesis_figures/out/senseiver.{pdf,png}

Shapes are those of the packaged checkpoints (supervisor_evaluation/models/senseiver_*.pt):
tokens of 4 observed values + a 64-d sin-cos position code (+ 8-d learned time embedding
and the scalar offset for G), projected to 32 channels; 3 encoder blocks, each a
cross-attention from the queries to the tokens followed by 3 self-attention layers
(block 1 has its own weights, blocks 2 and 3 share one set). A decodes with a
cross-attention from the 432 cells' queries [position code, learned vector q]; G has no decoder, a linear head
shared by all cells reads each cell's token. Parts that G changes are drawn in G's colour. Output thumbnails are each
model's own density reconstruction of the example frame.
"""
from __future__ import annotations

import os

import numpy as np
from matplotlib.patches import Rectangle

from compare import plotstyle as ps
from thesis_figures.common import (FS, FS_SMALL, GREY, LINE, arrow, block, canvas, field_rgb,
                                   grid_icon, method_color, save, text, thumbnail, token_icon)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CELL = 0.05


def main() -> None:
    d = np.load(os.path.join(HERE, "data", "example_frame.npz"))
    names = list(d["method_names"])
    walk = d["walkable"]
    vmax = float(np.percentile(d["truth"][0][walk], 99))
    G = method_color("Senseiver-G (ours)")

    fig, ax = canvas(16.0, 6.5)
    bw, bh = 2.35, 0.9                                 # block size
    rows = [("(a) Senseiver-A", "Senseiver-A", 4.55, False),
            ("(b) Senseiver-G (ours)", "Senseiver-G", 1.3, True)]
    for title, key, yc, is_g in rows:
        c_new = G if is_g else LINE                    # colour of what G changes
        text(ax, 0.15, yc + 1.62, title, FS, va="bottom")

        # input tokens
        tk = token_icon(ax, 0.35, yc - 0.33, 6, color=c_new)
        if is_g:
            text(ax, 0.6, yc - 0.55, r"$[\,y_i,\ \mathrm{PE}(\chi_i),\ \tau(\Delta_i)\,]$", FS_SMALL,
                 ha="center", color=c_new)
            text(ax, 0.6, yc - 0.9, r"observed cells of $t-15..t$", FS_SMALL - 0.5, color=GREY,
                 ha="center")
        else:
            text(ax, 0.6, yc - 0.55, r"$[\,y_i,\ \mathrm{PE}(\chi_i)\,]$", FS_SMALL, ha="center")
            text(ax, 0.6, yc - 0.9, r"observed cells of $t$", FS_SMALL - 0.5, color=GREY, ha="center")
        text(ax, 0.6, tk[3] + 0.08, "sensor\ntokens", FS_SMALL - 0.5, color=GREY, ha="center",
             va="bottom", linespacing=1.1)

        # encoder: 3 blocks, each cross-attention (queries from above) + 3 self-attention
        # layers; block 1 has its own weights, blocks 2 and 3 share one set
        ca = block(ax, 2.15, yc - bh / 2, bw, bh, "cross-attention", "keys, values: tokens")
        sa = block(ax, ca[2] + 0.45, yc - bh / 2, bw, bh, "self-attention", "3 layers")
        arrow(ax, (ca[2] + 0.05, yc), (sa[0] - 0.05, yc))
        enc = (ca[0] - 0.15, ca[1] - 0.15, sa[2] + 0.15, ca[3] + 0.15)
        ax.add_patch(Rectangle((enc[0], enc[1]), enc[2] - enc[0], enc[3] - enc[1], fill=False,
                               ec=GREY, lw=0.6, ls=(0, (3, 1.5)), zorder=1))
        text(ax, (enc[0] + enc[2]) / 2, enc[1] - 0.08,
             r"encoder block, $\times 3$ (blocks 2 and 3 share weights)", FS_SMALL - 0.5,
             color=GREY, ha="center", va="top")
        arrow(ax, (tk[2] + 0.1, yc), (enc[0] - 0.03, yc))
        if is_g:
            q = grid_icon(ax, ca[0] + 0.3, enc[3] + 0.42, 4, 12, 0.12, color=G)
            text(ax, q[2] + 0.12, (q[1] + q[3]) / 2, "432 cell tokens\n(one per grid cell)",
                 FS_SMALL - 0.5, color=G, va="center", linespacing=1.1)
        else:
            q = grid_icon(ax, ca[0] + 0.72, enc[3] + 0.42, 4, 4, 0.14)
            text(ax, q[2] + 0.12, (q[1] + q[3]) / 2, "64 learned\nlatents", FS_SMALL - 0.5,
                 color=GREY, va="center", linespacing=1.1)
        arrow(ax, ((ca[0] + ca[2]) / 2, q[1] - 0.04), ((ca[0] + ca[2]) / 2, ca[3] + 0.02))
        text(ax, (ca[0] + ca[2]) / 2 - 0.08, (q[1] + enc[3]) / 2, "queries", FS_SMALL - 1,
             color=GREY, ha="right", va="center")
        sa = (sa[0], sa[1], enc[2], sa[3])            # what follows starts after the encoder box

        if is_g:
            ro = block(ax, sa[2] + 0.55, yc - bh / 2, bw, bh, "linear read-out", "per cell token",
                       edge=G, lw=0.9)
            arrow(ax, (sa[2] + 0.05, yc), (ro[0] - 0.05, yc))
            last = ro
            text(ax, (ro[0] + ro[2]) / 2, ro[1] - 0.25, "no decoder: each cell token\nholds its own cell",
                 FS_SMALL - 0.5, color=G, ha="center", va="top", linespacing=1.1)
        else:
            z = grid_icon(ax, sa[2] + 0.35, yc - 0.28, 4, 4, 0.14)
            text(ax, (z[0] + z[2]) / 2, z[1] - 0.12, r"$z$", FS, ha="center", va="top")
            arrow(ax, (sa[2] + 0.05, yc), (z[0] - 0.05, yc))
            de = block(ax, z[2] + 0.35, yc - bh / 2, 2.2, bh, "cross-attention", "decoder")
            arrow(ax, (z[2] + 0.05, yc), (de[0] - 0.05, yc))
            text(ax, (de[0] + de[2]) / 2, de[1] - 0.45, r"queries: $[\mathrm{PE}(\chi),\ q]$ for all 432 cells",
                 FS_SMALL - 0.5, color=GREY, ha="center", va="center")
            arrow(ax, ((de[0] + de[2]) / 2, de[1] - 0.3), ((de[0] + de[2]) / 2, de[1] - 0.02))
            li = block(ax, de[2] + 0.35, yc - bh / 2, 0.95, bh, "linear")
            arrow(ax, (de[2] + 0.05, yc), (li[0] - 0.05, yc))
            last = li

        # output
        pred = d["predictions"][names.index(key)][0]
        xo = 13.15                                     # same place in both rows
        eo = thumbnail(ax, field_rgb(pred, walk, ps.CMAP["density"], vmax), xo, yc - 0.9, OUT_CELL)
        arrow(ax, (last[2] + 0.05, yc), (eo[0] - 0.08, yc))
        text(ax, (eo[0] + eo[1]) / 2, eo[2] - 0.08, r"$\hat{x}_t$", 9, ha="center")
        text(ax, eo[1] + 0.12, yc, "4 channels\non 432 cells", FS_SMALL - 0.5, color=GREY,
             va="center", linespacing=1.1)

    save(fig, os.path.join(HERE, "out", "senseiver"))


if __name__ == "__main__":
    main()
