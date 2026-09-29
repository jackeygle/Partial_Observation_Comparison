"""Thesis figure: the six reconstructions of one test frame, side by side.

    python3 -m thesis_figures.comparison        # -> thesis_figures/out/comparison_<day>_<frame>.{pdf,png}

Top row: the robots' observation, the truth and each method's density reconstruction, on
one colour scale. Bottom row: the absolute error of each reconstruction, on one colour
scale, with that frame's density RMSE on unobserved walkable cells under it. Frames come
from the evaluation's saved selection (supervisor_evaluation/.../selected_predictions.npz,
written by `evaluate.py images` with the final models).
"""
from __future__ import annotations

import os
import sys

import matplotlib as mpl
import numpy as np
from matplotlib.patches import Rectangle

from compare import plotstyle as ps
from thesis_figures.common import (FS, FS_SMALL, GREY, LINE, canvas, field_rgb, method_color,
                                   nice_top, save, sub_axes, text, thumbnail)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SEL = os.path.join(ROOT, "supervisor_evaluation/outputs/full/images/selected_predictions.npz")
ORDER = [("Senseiver-A", "Senseiver-A"), ("Senseiver-G", "Senseiver-G (ours)"),
         ("DINCAE", "DINCAE"), ("4DVarNet MSE", "4DVarNet"),
         ("4DVarNet aughead_obs (single)", "4DVarNet aug. var. (ours)"), ("EnKF", "EnKF")]
SEEN = "#a9c9ee"                                                 # as in the observation figure
FRAMES = [("atc-20130811", 32778), ("atc-20130811", 13551)]      # a busy and a quieter frame
CELL = 0.1


def draw(day: str, frame: int) -> None:
    z = np.load(SEL)
    i = int(np.flatnonzero((z["day"] == day) & (z["frame"] == frame))[0])
    names = list(z["method_names"])
    walk, seen = z["walkable"], z["observed"][i]
    truth = z["truth"][i, 0]
    preds = [z["predictions"][i, names.index(k), 0] for k, _ in ORDER]
    blind = walk & ~seen
    vmax = float(np.percentile(truth[walk], 99.5))
    errs = [np.abs(p - truth) for p in preds]
    emax = float(np.percentile(np.concatenate([e[walk] for e in errs]), 99.5))

    tw, gap = 12 * CELL, 0.52
    x0 = 0.95
    page = (16.0, 2 * 36 * CELL + 3.0)
    fig, ax = canvas(*page)
    ytop = page[1] - 0.8 - 36 * CELL                  # bottom 0.65 cm: the RMSE definition
    ybot = ytop - 0.55 - 36 * CELL

    def col_x(k):
        return x0 + k * (tw + gap)

    text(ax, 0.05, ytop + 18 * CELL, "density", FS, va="center", rotation=90)
    text(ax, 0.05, ybot + 18 * CELL, r"$|\hat{x} - x|$", FS, va="center", rotation=90)
    # observation and truth
    eo = thumbnail(ax, field_rgb(z["observation"][i, 0], walk, ps.CMAP["density"], vmax, hide=~seen),
                   col_x(0), ytop, CELL)
    text(ax, (eo[0] + eo[1]) / 2, eo[3] + 0.1, "observed", FS_SMALL + 0.5, ha="center", va="bottom")
    et = thumbnail(ax, field_rgb(truth, walk, ps.CMAP["density"], vmax), col_x(1), ytop, CELL)
    text(ax, (et[0] + et[1]) / 2, et[3] + 0.1, "truth", FS_SMALL + 0.5, ha="center", va="bottom")
    # the blind cells, as a reference for the error row
    rgb = np.ones(walk.shape + (3,))
    rgb[~walk] = [int(c, 16) / 255 for c in ("bd", "bd", "bd")]
    rgb[walk & seen] = [int(SEEN[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    eb = thumbnail(ax, rgb, col_x(1), ybot, CELL)
    text(ax, (eb[0] + eb[1]) / 2, eb[2] - 0.1, "blue: observed\nwhite: scored", FS_SMALL - 0.5,
         color=GREY, ha="center", va="top", linespacing=1.1)
    for k, ((key, label), p, e) in enumerate(zip(ORDER, preds, errs)):
        x = col_x(k + 2)
        er = thumbnail(ax, field_rgb(p, walk, ps.CMAP["density"], vmax), x, ytop, CELL)
        lab = label.replace(" aug. var.", "\naug. var.").replace(" (ours)", "\n(ours)")
        text(ax, (er[0] + er[1]) / 2, er[3] + 0.1, lab, FS_SMALL + 0.5, ha="center", va="bottom",
             linespacing=1.05, color=LINE)
        ax.add_patch(Rectangle((er[0], er[3] + 0.03), tw, 0.05, fc=method_color(label), ec="none"))
        ee = thumbnail(ax, field_rgb(e, walk, ps.CMAP["abs_error"], emax), x, ybot, CELL)
        rmse = float(np.sqrt(np.mean((p - truth)[blind] ** 2)))
        text(ax, (ee[0] + ee[1]) / 2, ee[2] - 0.1, f"frame RMSE\n{rmse:.3f}", FS_SMALL - 0.5, color=GREY,
             ha="center", va="top", linespacing=1.1)
    text(ax, 0.3, 0.02,
         r"frame RMSE $= \sqrt{\mathrm{mean}_{j \in B}\,(\hat{x}_j - x_j)^2}$ of the density (people/m$^2$), "
         r"this frame only, over its unobserved walkable cells $B$" "\n"
         "(white in the mask). Test-set results over all frames are in the accuracy table.",
         FS_SMALL - 0.5, color=GREY, va="bottom", linespacing=1.3)
    # colour bars at the right
    cb_x = col_x(8) - gap + 0.25
    for (y, cmap, vm, lab) in ((ytop, ps.CMAP["density"], vmax, r"people/m$^2$"),
                               (ybot, ps.CMAP["abs_error"], emax, r"people/m$^2$")):
        cax = sub_axes(fig, *page, cb_x, y, 0.12, 36 * CELL)
        cb = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(0, vm), cmap), cax=cax,
                          ticks=[0, nice_top(vm)])
        cb.outline.set_linewidth(0.4)
        cax.tick_params(labelsize=FS_SMALL - 0.5, length=1.5, width=0.4, pad=1)
        cax.set_ylabel(lab, fontsize=FS_SMALL - 0.5, labelpad=1)
    save(fig, os.path.join(HERE, "out", f"comparison_{day}_{frame}"))


def main() -> None:
    frames = FRAMES if len(sys.argv) < 3 else [(sys.argv[1], int(sys.argv[2]))]
    for day, frame in frames:
        draw(day, frame)


if __name__ == "__main__":
    main()
