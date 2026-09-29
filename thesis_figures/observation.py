"""Thesis figure: the robot observation model.

    python3 -m thesis_figures.prepare_frame     # once
    python3 -m thesis_figures.prepare_day       # once
    python3 -m thesis_figures.observation       # -> thesis_figures/out/observation.{pdf,png}

(a) One frame: robots, their last 20 s of route, the 7-cell sensing radius, the walkable
    cells they see, and the walkable cells in range that obstacles hide from them.
(b) How often each cell is observed over the example test day.
(c) Fraction of walkable cells observed at each second of that day.
(d) For unobserved walkable cells: how long ago the cell was last observed (CDF).
"""
from __future__ import annotations

import os

import numpy as np
from matplotlib.patches import Circle, Rectangle

from compare import plotstyle as ps
from thesis_figures.common import (FS_SMALL, GREY, LINE, WALL, canvas, cell_centre,
                                   colorbar_under, field_rgb, save, sub_axes, text, thumbnail)

HERE = os.path.dirname(os.path.abspath(__file__))
CELL = 0.12
PAGE = (16.0, 6.3)
SEEN = "#a9c9ee"


def main() -> None:
    f = np.load(os.path.join(HERE, "data", "example_frame.npz"))
    g = np.load(os.path.join(HERE, "data", "example_day.npz"))
    walk, seen = f["walkable"], f["observed"]
    H, W = walk.shape
    fig, ax = canvas(*PAGE)
    ty = 1.25
    th = H * CELL

    # ---- (a) one frame ---------------------------------------------------------------------
    rr, cc = np.mgrid[0:H, 0:W]
    in_range = np.zeros_like(walk)
    for r, c in f["robots"]:
        in_range |= (rr - r) ** 2 + (cc - c) ** 2 <= int(f["sensing_range"]) ** 2
    rgb = np.ones((H, W, 3))
    rgb[~walk] = [int(WALL[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    rgb[walk & seen] = [int(SEEN[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    ea = thumbnail(ax, rgb, 0.35, ty, CELL)
    for r, c in zip(*np.nonzero(walk & in_range & ~seen)):          # hidden by obstacles
        ax.add_patch(Rectangle((ea[0] + c * CELL, ea[3] - (r + 1) * CELL), CELL, CELL,
                               fill=False, hatch="//////", lw=0, ec=GREY, zorder=2))
    clip = Rectangle((ea[0], ea[2]), ea[1] - ea[0], ea[3] - ea[2], transform=ax.transData)
    for tr in np.transpose(f["robot_tracks"], (1, 0, 2)):
        pts = np.array([cell_centre(ea, CELL, r, c) for r, c in tr])
        ax.plot(pts[:, 0], pts[:, 1], color=LINE, lw=0.6, zorder=4, clip_path=clip)
    robots = [tuple(p) for p in f["robots"].tolist()]
    for r, c in set(robots):                           # two robots can share a cell
        x, y = cell_centre(ea, CELL, r, c)
        if robots.count((r, c)) > 1:
            text(ax, x + 0.1, y + 0.1, rf"$\times{robots.count((r, c))}$", FS_SMALL - 1, zorder=6,
                 bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.8))
        circ = Circle((x, y), int(f["sensing_range"]) * CELL, fill=False, lw=0.6,
                      ls=(0, (2.5, 1.5)), ec=LINE, zorder=4)
        ax.add_patch(circ); circ.set_clip_path(clip)
        ax.plot(x, y, marker="o", ms=4, mfc=LINE, mec="white", mew=0.6, zorder=5)
    text(ax, ea[0], ea[3] + 0.15, "(a) one second", FS_SMALL + 0.5, va="bottom")
    ly = ty - 0.3                                                   # legend under (a)
    for k, (lab, kind) in enumerate([("seen", "seen"), ("occluded", "hidden"),
                                     ("obstacle", "wall")]):
        y = ly - k * 0.32
        if kind == "hidden":
            ax.add_patch(Rectangle((ea[0], y - 0.09), 0.18, 0.18, fill=False, hatch="//////",
                                   ec=GREY, lw=0.4))
        else:
            ax.add_patch(Rectangle((ea[0], y - 0.09), 0.18, 0.18,
                                   fc=SEEN if kind == "seen" else WALL, ec="none"))
        text(ax, ea[0] + 0.28, y, lab, FS_SMALL, va="center")

    # ---- (b) how often each cell is observed ------------------------------------------------
    freq = g["frequency"]
    eb = thumbnail(ax, field_rgb(freq, walk, ps.CMAP["coverage"], 1.0), ea[1] + 0.95, ty, CELL)
    text(ax, eb[0], eb[3] + 0.15, "(b) share of day seen", FS_SMALL + 0.5, va="bottom")
    colorbar_under(fig, ax, PAGE, eb, ps.CMAP["coverage"], 0, 1, [0, 0.5, 1], "fraction of time")

    # ---- (c) coverage through the day -------------------------------------------------------
    hours = ((g["time"] + 9 * 3600) % 86400) / 3600            # Japan Standard Time
    cov = g["coverage"]
    k = 300                                                    # 5-minute bins
    n = len(cov) // k
    binned = cov[: n * k].reshape(n, k)
    hb = hours[: n * k].reshape(n, k).mean(axis=1)
    cx0, cw = eb[1] + 1.4, 4.0
    axc = sub_axes(fig, *PAGE, cx0, ty, cw, th)
    axc.fill_between(hb, np.percentile(binned, 10, axis=1), np.percentile(binned, 90, axis=1),
                     color=ps.shade(ps.INK, 0.82), lw=0, label="10 to 90% of seconds")
    axc.plot(hb, binned.mean(axis=1), color=LINE, lw=0.9, label="5-min mean")
    axc.axhline(cov.mean(), color=GREY, lw=0.7, ls="--")
    axc.text(hb[0], cov.mean() + 0.2, f"day mean {cov.mean():.2f}", fontsize=FS_SMALL,
             color=GREY, ha="left", va="bottom")
    axc.legend(loc="lower right", fontsize=FS_SMALL - 0.5, handlelength=1.2, borderaxespad=0.2)
    axc.set_ylim(0, 1)
    axc.set_xlabel("time of day (h)", fontsize=FS_SMALL + 0.5)
    axc.set_ylabel("walkable cells observed", fontsize=FS_SMALL + 0.5)
    axc.tick_params(labelsize=FS_SMALL)
    axc.set_title("(c) coverage per second", loc="left", fontsize=FS_SMALL + 0.5)

    # ---- (d) how long unobserved cells have been unobserved -------------------------------
    hist = g["age_hist"].astype(float)
    cdf = np.cumsum(hist) / hist.sum()
    s = np.arange(len(cdf))
    dx0 = cx0 + cw + 1.3
    axd = sub_axes(fig, *PAGE, dx0, ty, PAGE[0] - 0.15 - dx0, th)
    axd.plot(s[1:], cdf[1:], color=LINE, lw=1.0)
    axd.set_xscale("log")
    axd.set_xlim(1, 1000)
    axd.set_ylim(0, 1.12)
    axd.set_yticks(np.linspace(0, 1, 6))
    for sec in (1, 15, 200):
        axd.plot([sec, sec], [0, cdf[sec]], color=GREY, lw=0.6, ls=":")
        axd.plot(sec, cdf[sec], "o", ms=2.5, color=LINE)
        right = sec < 100                           # the 200-s label goes above the curve
        axd.text(sec * (1.2 if right else 0.85), cdf[sec] - 0.04 if right else cdf[sec] + 0.03,
                 f"{cdf[sec]:.2f} within {sec} s", fontsize=FS_SMALL,
                 ha="left" if right else "right", va="top" if right else "bottom")
    axd.set_xlabel("seconds since last observed", fontsize=FS_SMALL + 0.5)
    axd.set_ylabel("share of blind cells", fontsize=FS_SMALL + 0.5)
    axd.tick_params(labelsize=FS_SMALL)
    axd.set_title("(d) age of blind cells", loc="left", fontsize=FS_SMALL + 0.5)

    save(fig, os.path.join(HERE, "out", "observation"))


if __name__ == "__main__":
    main()
