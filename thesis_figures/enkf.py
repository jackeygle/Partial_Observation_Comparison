"""Thesis figure: the EnKF cycle, once per second.

    python3 -m thesis_figures.enkf              # -> thesis_figures/out/enkf.{pdf,png}

100 members. Forecast: each member runs PedPred3 on its last five analysed states and the
running bias estimate is subtracted. Process noise: a whole one-step forecast-error map
drawn from a bank of 8,192 built on the training days, AR(1) in time (rho = 0.5), scaled
by 1.5 and, on cells no robot sees now, by per-channel factors (1, 1.25, 1.4, 0.93), member
mean removed. Analysis: localised Kalman
update (radius 7 cells, inflation 1.02, cross-channel weights) with perturbed observations.
Output: ensemble mean and spread. Bias: 0.95 * bias + 0.05 * (forecast mean - y) at observed
cells. Thumbnails: the final EnKF on the example frame.
"""
from __future__ import annotations

import os

import numpy as np
from matplotlib.patches import Rectangle

from compare import plotstyle as ps
from thesis_figures.common import (FS_SMALL, GREY, LINE, arrow, block, canvas, field_rgb,
                                   method_color, save, text, thumbnail)

HERE = os.path.dirname(os.path.abspath(__file__))
C = method_color("EnKF")


def cards(ax, x, y, w, h, n=3, off=0.1, color=LINE):
    for k in range(n - 1, -1, -1):
        ax.add_patch(Rectangle((x + k * off, y + k * off), w, h, fc="white", ec=color, lw=0.5))
    return (x, y, x + w + (n - 1) * off, y + h + (n - 1) * off)


def main() -> None:
    d = np.load(os.path.join(HERE, "data", "example_frame.npz"))
    names = list(d["method_names"]); snames = list(d["spread_method_names"])
    walk, seen = d["walkable"], d["observed"]
    vmax = float(np.percentile(d["truth"][0][walk], 99))
    cell = 0.05
    fig, ax = canvas(16.0, 5.1)
    yc = 2.05
    bh = 1.25

    # ensemble of analysed states
    en = cards(ax, 0.3, yc - 0.4, 0.8, 0.6, n=4, off=0.08, color=C)
    text(ax, (en[0] + en[2]) / 2, en[3] + 0.1, "ensemble\n" r"$x_i^{a}$, $i = 1..100$", FS_SMALL,
         ha="center", va="bottom", linespacing=1.1)

    fc = block(ax, en[2] + 0.5, yc - bh / 2, 2.5, bh, "forecast",
               "PedPred3 on the last\n" r"5 analyses, $-$ bias")
    arrow(ax, (en[2] + 0.08, yc), (fc[0] - 0.05, yc))

    nz = block(ax, fc[2] + 0.45, yc - bh / 2, 3.05, bh, r"$+$ process noise $q_i$",
               "real error map, AR(1), " r"$\times1.5$" "\nchannel factors on blind cells", edge=C, lw=0.9)
    arrow(ax, (fc[2] + 0.05, yc), (nz[0] - 0.05, yc))
    bank = cards(ax, nz[0] + 0.35, nz[3] + 0.4, 0.9, 0.45, n=4, off=0.07)
    text(ax, bank[2] + 0.12, (bank[1] + bank[3]) / 2, "8,192 one-step\nforecast errors\n(training days)",
         FS_SMALL - 0.5, color=GREY, va="center", linespacing=1.1)
    arrow(ax, ((bank[0] + bank[2]) / 2, bank[1] - 0.03), ((bank[0] + bank[2]) / 2, nz[3] + 0.03))

    ku = block(ax, nz[2] + 0.45, yc - bh / 2, 2.75, bh, "Kalman update",
               "localised (7 cells),\ncross-channel weights")
    arrow(ax, (nz[2] + 0.05, yc), (ku[0] - 0.05, yc))
    oc = 0.035                                         # robot observations, from above
    ob = thumbnail(ax, field_rgb(d["observation"][0], walk, ps.CMAP["density"], vmax, hide=~seen),
                   (ku[0] + ku[2]) / 2 - 6 * oc, ku[3] + 0.4, oc)
    text(ax, ob[1] + 0.12, (ob[2] + ob[3]) / 2, r"robots' $y_t$" "\n(perturbed)", FS_SMALL,
         va="center", linespacing=1.1)
    arrow(ax, ((ku[0] + ku[2]) / 2, ob[2] - 0.03), ((ku[0] + ku[2]) / 2, ku[3] + 0.03))

    # outputs: ensemble mean and spread
    ox = ku[2] + 0.75
    arrow(ax, (ku[2] + 0.05, yc), (ox - 0.08, yc))
    pm = d["predictions"][names.index("EnKF")][0]
    sp = d["spreads"][snames.index("EnKF")]
    e1 = thumbnail(ax, field_rgb(pm, walk, ps.CMAP["density"], vmax), ox, yc - 0.9, cell)
    smax = float(np.percentile(sp[walk], 99))
    e2 = thumbnail(ax, field_rgb(sp, walk, ps.CMAP["spread"], smax), e1[1] + 0.2, yc - 0.9, cell)
    text(ax, (e1[0] + e1[1]) / 2, e1[3] + 0.08, "mean\n" r"$\hat{x}_t$", FS_SMALL, ha="center",
         va="bottom", linespacing=1.1)
    text(ax, (e2[0] + e2[1]) / 2, e2[3] + 0.08, "spread\n" r"$\hat{\sigma}_t$", FS_SMALL, ha="center",
         va="bottom", linespacing=1.1)

    # loop back: the analysis feeds the next second
    ly = fc[1] - 0.45
    xk = ku[2] - 0.4
    ax.plot([xk, xk], [ku[1], ly], color=LINE, lw=0.7)
    ax.plot([xk, (en[0] + en[2]) / 2], [ly, ly], color=LINE, lw=0.7)
    arrow(ax, ((en[0] + en[2]) / 2, ly), ((en[0] + en[2]) / 2, en[1] - 0.03))
    text(ax, (fc[0] + ku[2]) / 2, ly - 0.08, r"analysis ensemble $x_i^a$ $\rightarrow$ next second",
         FS_SMALL, color=GREY, ha="center", va="top")
    text(ax, 0.3, 0.0, r"bias $\leftarrow 0.95\,$bias$\,+\,0.05\,(\bar{x}^f - y_t)$ on observed cells,"
         " updated at every analysis", FS_SMALL, color=GREY, va="bottom")

    save(fig, os.path.join(HERE, "out", "enkf"))


if __name__ == "__main__":
    main()
