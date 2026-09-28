"""Thesis figure: predicted uncertainty against the actual error, one test frame.

    python3 -m thesis_figures.uncertainty_maps  # -> thesis_figures/out/uncertainty_maps_<day>_<frame>.{pdf,png}

For each of the three probabilistic methods: the absolute density error |x_hat - x| and
the predicted density sigma-hat, next to each other. Error and sigma-hat are in the same
unit and share ONE colour scale across all six panels, so "sigma-hat looks like the error"
can be read directly. Under each pair: that frame's spread/RMSE on unobserved walkable
cells (spread = root-mean-square sigma-hat). Frames: the evaluation's saved selection.
"""
from __future__ import annotations

import os
import sys

import matplotlib as mpl
import numpy as np
from matplotlib.patches import Rectangle

from compare import plotstyle as ps
from thesis_figures.common import (FS, FS_SMALL, GREY, canvas, field_rgb, method_color, nice_top,
                                   save, sub_axes, text, thumbnail)
from thesis_figures.comparison import FRAMES, SEL

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = [("DINCAE", "DINCAE"), ("4DVarNet aughead_obs (single)", "4DVarNet (aug. head)"),
           ("EnKF", "EnKF")]
CELL = 0.1


def draw(day: str, frame: int) -> None:
    z = np.load(SEL)
    i = int(np.flatnonzero((z["day"] == day) & (z["frame"] == frame))[0])
    names, snames = list(z["method_names"]), list(z["spread_method_names"])
    walk, seen = z["walkable"], z["observed"][i]
    blind = walk & ~seen
    truth = z["truth"][i, 0]
    errs = [np.abs(z["predictions"][i, names.index(k), 0] - truth) for k, _ in METHODS]
    sigs = [z["density_spread"][i, snames.index(k)] for k, _ in METHODS]
    vmax = float(np.percentile(np.concatenate([a[walk] for a in errs + sigs]), 99))   # ONE scale

    gap, pair_gap = 0.25, 0.85
    page = (16.0, 36 * CELL + 1.9)
    fig, ax = canvas(*page)
    y0 = 0.75
    dmax = float(np.percentile(truth[walk], 99.5))
    et = thumbnail(ax, field_rgb(truth, walk, ps.CMAP["density"], dmax), 0.3, y0, CELL)
    text(ax, (et[0] + et[1]) / 2, et[3] + 0.1, "truth", FS_SMALL + 0.5, ha="center", va="bottom")
    x = et[1] + pair_gap
    for (key, label), e, s in zip(METHODS, errs, sigs):
        ee = thumbnail(ax, field_rgb(e, walk, ps.CMAP["spread"], vmax), x, y0, CELL)
        es = thumbnail(ax, field_rgb(s, walk, ps.CMAP["spread"], vmax), ee[1] + gap, y0, CELL)
        text(ax, (ee[0] + ee[1]) / 2, ee[3] + 0.1, r"$|\hat{x}-x|$", FS_SMALL + 0.5, ha="center",
             va="bottom")
        text(ax, (es[0] + es[1]) / 2, es[3] + 0.1, r"$\hat{\sigma}$", FS_SMALL + 0.5, ha="center",
             va="bottom")
        ax.add_patch(Rectangle((ee[0], ee[3] + 0.5), es[1] - ee[0], 0.05, fc=method_color(label),
                               ec="none"))
        text(ax, (ee[0] + es[1]) / 2, ee[3] + 0.6, label, FS, ha="center", va="bottom")
        rmse = float(np.sqrt(np.mean(((z["predictions"][i, names.index(key), 0] - truth)[blind]) ** 2)))
        spread = float(np.sqrt(np.mean(s[blind] ** 2)))
        text(ax, (ee[0] + es[1]) / 2, ee[2] - 0.1, f"spread / RMSE {spread / rmse:.2f}",
             FS_SMALL - 0.5, color=GREY, ha="center", va="top")
        x = es[1] + pair_gap
    cax = sub_axes(fig, *page, x - pair_gap + 0.35, y0, 0.12, 36 * CELL)
    cb = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(0, vmax), ps.CMAP["spread"]),
                      cax=cax, ticks=[0, nice_top(vmax)])
    cb.outline.set_linewidth(0.4)
    cax.tick_params(labelsize=FS_SMALL - 0.5, length=1.5, width=0.4, pad=1)
    cax.set_ylabel(r"people/m$^2$ (error and $\hat{\sigma}$)", fontsize=FS_SMALL - 0.5, labelpad=1)
    save(fig, os.path.join(HERE, "out", f"uncertainty_maps_{day}_{frame}"))


def main() -> None:
    frames = FRAMES if len(sys.argv) < 3 else [(sys.argv[1], int(sys.argv[2]))]
    for day, frame in frames:
        draw(day, frame)


if __name__ == "__main__":
    main()
