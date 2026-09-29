"""Thesis figure: overview of the study, from the gridded ATC data to the evaluation.

    python3 -m thesis_figures.prepare_frame     # once: caches the example frame
    python3 -m thesis_figures.pipeline          # -> thesis_figures/out/pipeline.{pdf,png}

Real data in every thumbnail (test day 2013-08-11, frame 32778, density channel):
ground truth, what the three robots see, and DINCAE's reconstruction and sigma-hat.
Details (grid size, sensing radius, split) belong in the caption, not on the figure.
"""
from __future__ import annotations

import os

import numpy as np
from matplotlib.patches import Circle, Rectangle

from compare import plotstyle as ps
from thesis_figures.common import (FS_SMALL, GREY, LINE, arrow, canvas, cell_centre,
                                   field_rgb, frame_box, method_color, save, text, thumbnail)

HERE = os.path.dirname(os.path.abspath(__file__))
CELL = 0.075                      # cm per grid cell in the thumbnails


def main() -> None:
    d = np.load(os.path.join(HERE, "data", "example_frame.npz"))
    walk, seen = d["walkable"], d["observed"]
    dens = d["truth"][0]
    vmax = float(np.percentile(dens[walk], 99))
    smax = float(np.percentile(d["dincae_density_sigma"][walk], 99))

    th = 36 * CELL                                # thumbnail height (2.7 cm)
    ty = 1.25                                     # thumbnail bottom
    head_y = ty + th + 0.2                        # stage titles sit just above
    fig, ax = canvas(16.0, head_y + 0.75)

    def stage(x, title):
        text(ax, x, head_y, title, ha="center", va="bottom", linespacing=1.15)

    # (a) ground truth ---------------------------------------------------------------
    xa = 0.35
    ea = thumbnail(ax, field_rgb(dens, walk, ps.CMAP["density"], vmax), xa, ty, CELL)
    stage((ea[0] + ea[1]) / 2, "Crowd\nstate")
    text(ax, (ea[0] + ea[1]) / 2, ty - 0.12, r"$x_t$", 9, ha="center")

    # (b) robot observations ---------------------------------------------------------
    xb = ea[1] + 1.15
    eb = thumbnail(ax, field_rgb(d["observation"][0], walk, ps.CMAP["density"], vmax, hide=~seen),
                   xb, ty, CELL)
    clip = Rectangle((eb[0], eb[2]), eb[1] - eb[0], eb[3] - eb[2], transform=ax.transData)
    for tr in np.transpose(d["robot_tracks"], (1, 0, 2)):
        pts = np.array([cell_centre(eb, CELL, r, c) for r, c in tr])
        ax.plot(pts[:, 0], pts[:, 1], color=LINE, lw=0.5, alpha=0.6, zorder=4, clip_path=clip)
    robots = [tuple(p) for p in d["robots"].tolist()]
    for r, c in set(robots):                           # two robots can share a cell
        x, y = cell_centre(eb, CELL, r, c)
        if robots.count((r, c)) > 1:
            text(ax, x + 0.1, y + 0.1, rf"$\times{robots.count((r, c))}$", FS_SMALL - 1, zorder=6,
                 bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.8))
        circ = Circle((x, y), int(d["sensing_range"]) * CELL, fill=False, lw=0.5, ls=(0, (2, 1.5)),
                      ec=LINE, zorder=4)
        ax.add_patch(circ); circ.set_clip_path(clip)
        ax.plot(x, y, marker="o", ms=3.2, mfc=LINE, mec="white", mew=0.5, zorder=5)
    stage((eb[0] + eb[1]) / 2, "Robot\nobservations")
    text(ax, (eb[0] + eb[1]) / 2, ty - 0.12, r"$y_t,\ \Omega_t$", 9, ha="center")
    arrow(ax, (ea[1] + 0.12, ty + th / 2), (eb[0] - 0.12, ty + th / 2))

    # (c) the six methods --------------------------------------------------------------
    mx0, mx1 = eb[1] + 0.7, eb[1] + 0.7 + 4.85
    my0, my1 = ty - 0.05, ty + th + 0.05
    frame_box(ax, mx0, my0, mx1, my1)
    stage((mx0 + mx1) / 2, "Reconstruction\nmethods")
    rows = [("Senseiver-A", r"$y_t$", False),
            ("Senseiver-G (ours)", r"$y_{t-15:t}$", False),
            ("DINCAE", r"$y_{t-1:t+1}$", True),
            ("4DVarNet", "200 s", False),
            ("4DVarNet aug. var. (ours)", "200 s", True),
            ("EnKF", r"$y_{1:t}$", True)]
    step = (my1 - my0 - 0.55) / (len(rows) - 1)
    for i, (name, window, sig) in enumerate(rows):
        y = my1 - 0.3 - i * step
        ax.add_patch(Rectangle((mx0 + 0.18, y - 0.08), 0.16, 0.16, fc=method_color(name), ec="none"))
        text(ax, mx0 + 0.45, y, name + (r"$^{\ast}$" if sig else ""), va="center")
        text(ax, mx1 - 0.15, y, window, FS_SMALL, color=GREY, ha="right", va="center")
    text(ax, mx0, my0 - 0.1, r"$^{\ast}$ also predicts its uncertainty $\hat{\sigma}_t$",
         FS_SMALL, color=GREY)
    arrow(ax, (eb[1] + 0.12, ty + th / 2), (mx0 - 0.1, ty + th / 2))

    # (d) output: reconstruction and uncertainty -------------------------------------
    xd = mx1 + 0.65
    ed = thumbnail(ax, field_rgb(d["dincae_mean"][0], walk, ps.CMAP["density"], vmax), xd, ty, CELL)
    ed2 = thumbnail(ax, field_rgb(d["dincae_density_sigma"], walk, ps.CMAP["spread"], smax),
                    ed[1] + 0.25, ty, CELL)
    stage((ed[0] + ed2[1]) / 2, "Output")
    text(ax, (ed[0] + ed[1]) / 2, ty - 0.12, r"$\hat{x}_t$", 9, ha="center")
    text(ax, (ed2[0] + ed2[1]) / 2, ty - 0.12, r"$\hat{\sigma}_t$", 9, ha="center")
    arrow(ax, (mx1 + 0.1, ty + th / 2), (ed[0] - 0.12, ty + th / 2))

    # (e) evaluation ------------------------------------------------------------------------
    ex0, ex1 = ed2[1] + 0.6, 15.95
    frame_box(ax, ex0, my0, ex1, my1)
    stage((ex0 + ex1) / 2, "Evaluation")
    lines = [("accuracy", "RMSE"), ("uncertainty", "CRPS"), ("", "spread/RMSE"),
             ("speed", "ms per frame")]
    for j, (k, v) in enumerate(lines):
        y = my1 - 0.35 - j * 0.42
        text(ax, ex0 + 0.12, y, k, FS_SMALL, color=GREY, va="center")
        text(ax, ex0 + 1.6, y, v, va="center")
    text(ax, ex0 + 0.15, my0 + 0.55, "on unobserved\nwalkable cells", FS_SMALL, color=GREY, va="center")
    arrow(ax, (ed2[1] + 0.12, ty + th / 2), (ex0 - 0.1, ty + th / 2))

    # ground truth to the evaluation -------------------------------------------------------
    gy = 0.25
    cx = (ea[0] + ea[1]) / 2
    ax.plot([cx, cx, (ex0 + ex1) / 2], [ty - 0.55, gy, gy], color=GREY, lw=0.7, ls=(0, (3, 2)))
    arrow(ax, ((ex0 + ex1) / 2, gy), ((ex0 + ex1) / 2, my0 - 0.02), color=GREY)
    text(ax, cx + 0.15, gy + 0.05, "ground truth (test days)", FS_SMALL, color=GREY, va="bottom")

    save(fig, os.path.join(HERE, "out", "pipeline"))


if __name__ == "__main__":
    main()
