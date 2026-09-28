"""Shared helpers for the thesis schematics.

Academic style: Computer Modern (the LaTeX body font, bundled with matplotlib as cmr10),
thin black line work on white, colour only where it encodes something (method identity,
data values). Method colours and colormaps come from compare/plotstyle.py, so they match
the data figures. Coordinates are centimetres on the page.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.patches import FancyArrowPatch, Rectangle

from compare import plotstyle as ps

CM = 1 / 2.54
FS = 8.0            # body text: the size of a LaTeX caption at 11 pt
FS_SMALL = 7.0      # secondary annotations
LINE = "#1b1b1b"
GREY = "#6b6b6b"
WALL = "#bdbdbd"    # obstacle cells in every map thumbnail


def canvas(width_cm: float, height_cm: float):
    """A blank figure whose data coordinates are centimetres (1 unit = 1 cm on paper).

    Uses the data figures' style (compare/plotstyle.py) so any chart placed on the page
    with sub_axes looks like the other thesis figures."""
    ps.use(dpi=300)
    plt.rcParams.update({"font.size": FS, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(width_cm * CM, height_cm * CM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_cm)
    ax.set_ylim(0, height_cm)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def save(fig, stem: str) -> None:
    """Vector PDF for LaTeX, PNG for previewing. No tight bbox: the canvas is the page."""
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    fig.savefig(stem + ".pdf", bbox_inches=None, pad_inches=0)
    fig.savefig(stem + ".png", bbox_inches=None, pad_inches=0, dpi=300)
    plt.close(fig)
    print(f"[figure] {stem}.pdf", flush=True)


def text(ax, x, y, s, size=FS, color=LINE, ha="left", va="top", linespacing=1.35, **kw):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, linespacing=linespacing, **kw)


def frame_box(ax, x0, y0, x1, y1, lw=0.6, color=LINE, **kw):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=lw, ec=color, **kw))


def arrow(ax, p0, p1, color=LINE, lw=0.7, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>,head_length=3.2,head_width=1.6",
                                 color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0,
                                 mutation_scale=1))


def field_rgb(values, walkable, cmap, vmax, hide=None):
    """(H, W) field -> RGB image: colormap on walkable cells, grey walls, white where hidden."""
    rgb = colormaps[cmap](np.clip(values / vmax, 0, 1))[..., :3]
    rgb[~walkable] = [int(WALL[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    if hide is not None:
        rgb[hide & walkable] = 1.0
    return rgb


def thumbnail(ax, rgb, x0, y0, cell):
    """Draw an (H, W, 3) grid image with its top-left cell at the top; returns its extent."""
    h, w = rgb.shape[:2]
    ext = (x0, x0 + w * cell, y0, y0 + h * cell)
    ax.imshow(rgb, extent=ext, origin="upper", interpolation="nearest", zorder=1)
    frame_box(ax, ext[0], ext[2], ext[1], ext[3], lw=0.5, zorder=3)
    return ext


def cell_centre(ext, cell, row, col):
    return ext[0] + (col + 0.5) * cell, ext[3] - (row + 0.5) * cell


def block(ax, x0, y0, w, h, label, sub=None, edge=LINE, lw=0.6, face="white", size=FS):
    """A network module: thin box, name centred, optional grey shape/detail line under it."""
    ax.add_patch(Rectangle((x0, y0), w, h, fc=face, ec=edge, lw=lw, zorder=2))
    n = sub.count("\n") + 1 if sub else 0             # sub may span several lines
    cy = y0 + h / 2 + 0.13 * n
    text(ax, x0 + w / 2, cy, label, size, ha="center", va="center", zorder=3)
    if sub:
        text(ax, x0 + w / 2, cy - 0.17, sub, FS_SMALL - 0.5, color=GREY, ha="center", va="top",
             zorder=3, linespacing=1.15)
    return (x0, y0, x0 + w, y0 + h)


def token_icon(ax, x0, y0, n, w=0.5, h=0.09, gap=0.04, color=LINE):
    """A short stack of bars standing for a set of tokens."""
    for i in range(n):
        ax.add_patch(Rectangle((x0, y0 + i * (h + gap)), w, h, fc="white", ec=color, lw=0.5))
    return (x0, y0, x0 + w, y0 + n * (h + gap) - gap)


def grid_icon(ax, x0, y0, rows, cols, cell, color=LINE, face="white"):
    """A small rows x cols lattice of squares (latent array, cell tokens)."""
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(Rectangle((x0 + c * cell, y0 + r * cell), cell * 0.82, cell * 0.82,
                                   fc=face, ec=color, lw=0.35))
    return (x0, y0, x0 + cols * cell, y0 + rows * cell)


def sub_axes(fig, page_w, page_h, x0, y0, w, h):
    """An axes placed at (x0, y0, w, h) centimetres on a page_w x page_h cm canvas."""
    return fig.add_axes([x0 / page_w, y0 / page_h, w / page_w, h / page_h])


def colorbar_under(fig, ax, page, ext, cmap, vmin, vmax, ticks, label, gap=0.12, height=0.1):
    """Thin horizontal colour bar under a thumbnail, with its own ticks and a unit label."""
    import matplotlib as mpl
    cax = sub_axes(fig, *page, ext[0], ext[2] - gap - height, ext[1] - ext[0], height)
    cb = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(vmin, vmax), cmap),
                      cax=cax, orientation="horizontal", ticks=ticks)
    cb.outline.set_linewidth(0.4)
    cax.tick_params(labelsize=FS_SMALL - 0.5, length=1.5, width=0.4, pad=1)
    cax.set_xlabel(label, fontsize=FS_SMALL - 0.5, labelpad=1)
    return cax


def nice_top(v: float) -> float:
    """The largest 2-significant-digit number not above v: a colour-bar end tick that shows."""
    e = 10 ** (np.floor(np.log10(v)) - 1)
    return float(np.floor(v / e) * e)


METHOD_KEY = {"Senseiver-A": "Senseiver-A", "Senseiver-G (ours)": "Senseiver", "DINCAE": "DINCAE",
              "4DVarNet": "4DVarNet", "4DVarNet (aug. head)": "4DVarNet+aug", "EnKF": "EnKF"}


def method_color(name: str) -> str:
    return ps.METHOD_COLORS[METHOD_KEY[name]]
