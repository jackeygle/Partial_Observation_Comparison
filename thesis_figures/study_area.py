"""Thesis figure: the study area and the crowd state.

    python3 -m thesis_figures.study_area        # -> thesis_figures/out/study_area.{pdf,png}

(a) The ATC localisation map (laser obstacle points) with the 36 x 12 m corridor box that
    is gridded. (b) Walkable (white) and obstacle (grey) cells of that grid. (c)-(f) The
    four state channels at one frame (the example frame of the pipeline figure).
The black dot marks grid cell (0, 0) on the map and on every grid, so orientation carries
over from (a) to (b)-(f).
"""
from __future__ import annotations

import os

import numpy as np

from compare import plotstyle as ps
from thesis_figures.common import (FS_SMALL, GREY, LINE, canvas, colorbar_under, field_rgb,
                                   save, sub_axes, text, thumbnail)

HERE = os.path.dirname(os.path.abspath(__file__))
CELL = 0.1                                   # cm per grid cell in the thumbnails
XLIM, YLIM = (-46.0, 62.0), (-37.5, 20.5)    # metres: the mapped part of the building
MAP_W = 7.2                                  # cm
MAP_H = MAP_W * (YLIM[1] - YLIM[0]) / (XLIM[1] - XLIM[0])
PAGE = (16.0, 1.05 + MAP_H + 0.55)


def signed_rgb(values, walkable, vlim):
    return field_rgb(values + vlim, walkable, ps.CMAP["signed"], 2 * vlim)


def main() -> None:
    from crowdcore import navigation as nav
    from crowdcore.data.h5_to_grid import SUBSETS, rotation_matrix

    d = np.load(os.path.join(HERE, "data", "example_frame.npz"))
    walk, x = d["walkable"], d["truth"]
    fig, ax = canvas(*PAGE)

    # ---- (a) the ATC map with the gridded corridor ------------------------------------
    img, res, org = nav.load_atc_map()
    xlim, ylim, mw, mh = XLIM, YLIM, MAP_W, MAP_H
    mx0, my0 = 0.05, 1.05
    max_ = sub_axes(fig, *PAGE, mx0, my0, mw, mh)
    # Laser obstacle points, max-pooled to 0.25 m so thin walls survive at print size.
    k = 5
    occ = (img == 0)[: img.shape[0] // k * k, : img.shape[1] // k * k]
    occ = occ.reshape(occ.shape[0] // k, k, occ.shape[1] // k, k).max(axis=(1, 3))
    h, w = occ.shape
    top = org[1] + img.shape[0] * res                       # ROS: image row 0 is the top edge
    max_.imshow(np.where(occ, 0.25, 1.0), cmap="gray", vmin=0, vmax=1, interpolation="nearest",
                extent=(org[0], org[0] + w * k * res, top - h * k * res, top), origin="upper")
    s = SUBSETS["corridor"]
    R, o = rotation_matrix(s["theta"]), np.asarray(s["origin"])
    corners = np.array([[0, 0], [36, 0], [36, 12], [0, 12], [0, 0]], float) @ R.T + o
    max_.fill(corners[:, 0], corners[:, 1], color=ps.METHOD_COLORS["DINCAE"], alpha=0.12, lw=0)
    max_.plot(corners[:, 0], corners[:, 1], color=ps.METHOD_COLORS["DINCAE"], lw=0.9)
    max_.plot(*o, "o", ms=2.6, color=LINE)
    max_.plot([xlim[0] + 3, xlim[0] + 13], [ylim[0] + 3] * 2, color=LINE, lw=1.0)   # scale bar
    max_.text(xlim[0] + 8, ylim[0] + 4, "10 m", fontsize=FS_SMALL, ha="center", va="bottom")
    max_.text(corners[1, 0] + 1.5, corners[1, 1] + 1.0, r"$36\times12$ m grid", fontsize=FS_SMALL,
              color=ps.METHOD_COLORS["DINCAE"], ha="left", va="bottom")
    max_.set_xlim(*xlim); max_.set_ylim(*ylim); max_.set_aspect("equal"); max_.axis("off")
    text(ax, mx0, my0 + mh + 0.15, "(a) ATC shopping centre", FS_SMALL + 0.5, va="bottom")

    # ---- (b)-(f) the grid and the four channels ------------------------------------------
    dens, vx, vy, var = x
    vlim = float(np.ceil(np.percentile(np.abs(np.r_[vx[walk], vy[walk]]), 99) * 2) / 2)
    dmax = float(np.percentile(dens[walk], 99))
    varmax = float(np.percentile(var[walk], 99))
    panels = [
        ("(b) walkable", np.where(walk, 1.0, 1.0), None),
        ("(c) density", field_rgb(dens, walk, ps.CMAP["density"], dmax),
         (ps.CMAP["density"], 0, dmax, [0, round(dmax, 1)], r"people / m$^2$")),
        (r"(d) $v_x$", signed_rgb(vx, walk, vlim), "shared"),
        (r"(e) $v_y$", signed_rgb(vy, walk, vlim), None),
        ("(f) vel. var.", field_rgb(var, walk, ps.CMAP["variance"], varmax),
         (ps.CMAP["variance"], 0, varmax, [0, float(f"{varmax:.2g}")], r"m$^2$/s$^2$")),
    ]
    tw, gap = 12 * CELL, 0.5
    x0 = PAGE[0] - 0.3 - 5 * tw - 4 * gap
    ty = my0 + mh - 36 * CELL                  # top-aligned with the map
    for k, (title, arr, cb) in enumerate(panels):
        px = x0 + k * (tw + gap)
        if k == 0:
            rgb = field_rgb(np.zeros_like(dens), walk, "Greys", 1.0)   # white walkable, grey walls
        else:
            rgb = arr
        ext = thumbnail(ax, rgb, px, ty, CELL)
        ax.plot(ext[0], ext[3], "o", ms=2.6, color=LINE, zorder=6, clip_on=False)
        text(ax, (ext[0] + ext[1]) / 2, ext[3] + 0.15, title, FS_SMALL + 0.5, ha="center", va="bottom")
        if cb == "shared":                     # v_x and v_y share one bar under both panels
            both = (ext[0], ext[1] + gap + tw, ext[2], ext[3])
            colorbar_under(fig, ax, PAGE, both, ps.CMAP["signed"], -vlim, vlim,
                           [-vlim, 0, vlim], "m/s")
        elif cb is not None:
            colorbar_under(fig, ax, PAGE, ext, *cb)
        elif k == 0:
            # thin 1 m grid lines on the walkable panel
            for i in range(1, 36):
                ax.plot([ext[0], ext[1]], [ext[3] - i * CELL] * 2, color="#e3e3e3", lw=0.2, zorder=2)
            for j in range(1, 12):
                ax.plot([ext[0] + j * CELL] * 2, [ext[2], ext[3]], color="#e3e3e3", lw=0.2, zorder=2)
            text(ax, (ext[0] + ext[1]) / 2, ext[2] - 0.12, f"{int(walk.sum())} of 432\nwalkable",
                 FS_SMALL - 0.5, color=GREY, ha="center")

    save(fig, os.path.join(HERE, "out", "study_area"))


if __name__ == "__main__":
    main()
