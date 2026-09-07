"""
plotstyle.py — style shared by every figure. Fixes exactly three things, leaves
everything else alone.

Why this exists (all measured, not tidiness for its own sake):

  1. **Method -> colour mapping.** Before the refactor colours were set per script,
     so the same method had different colours in different figures. This is the
     highest-leverage fix: once the reader remembers "blue is DINCAE," every later
     figure saves them a trip to the legend.

  2. **Semantic -> colormap, one to one.** Before the refactor `Blues` appeared 4
     times, `gray` 3 times, `hot` 2 times, `Reds` 2 times -- and `Reds` was used for
     both "ensemble spread" and "error" at once, while density was `Blues` in some
     figures and `hot` in others. One colormap standing for two quantities means
     the reader cannot compare across figures.

  3. **Figure width fixed to the paper's text width.** Before the refactor font
     sizes were scattered across 23 different values (8.2/8.4/8.6/8.8/9.0/9.2/
     9.4/9.5/9.8/10/10.2/10.5/11/11.5/12/12.2/12.5/13/14/16...) -- the root cause
     wasn't a shaky hand, it was every figure being scaled to a different width and
     then hand-tuned to compensate. Fixing the width makes the problem disappear
     on its own.

Usage:

    from compare import plotstyle as ps
    ps.use()                                  # tunes rcParams, once per process
    fig, axes = ps.figure(ncols=4, rows_h=2.2)
    ax.bar(..., color=ps.METHOD_COLORS["DINCAE"])
    im = ax.imshow(density, cmap=ps.CMAP["density"], vmin=0, vmax=vmax)

**All text on a figure must be English.** Triton's matplotlib has only 24 fonts and
none of them are CJK; Chinese renders as tofu boxes and throws a stream of
"Glyph ... missing from font" warnings. Comments and docs are written in Chinese,
figure text in English -- the existing figures (plot_compare3 etc.) already follow
this convention.

Not this module's job: choosing a chart type for you, computing vmin/vmax for you,
drawing a legend. Those are specific to each figure; folding them in here would
just turn into a pile of switches.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# 1. Method -> colour
# --------------------------------------------------------------------------- #
#: Categorical palette. Taken from a dataviz spec's categorical slots, fixed
#: order, no cycling. Each of the five methods gets one slot, **consistent across
#: every figure and every sub-project**. Ablation rows (a4_k1, single-member) use
#: a paler shade of the same colour, see `shade()`.
METHOD_COLORS = {
    "DINCAE":    "#4269d0",   # blue
    "4DVarNet":  "#efb118",   # gold
    "Senseiver": "#ff725c",   # orange-red
    "EnKF":      "#6cc5b0",   # teal
    "4DVarNet+var": "#a463f2",  # purple -- the uncertainty head gets its own slot,
                                # it counts as the fifth method
}

#: Neutral colours: axes, text, gridlines. Three shades, do not add more.
INK = "#1b1b1b"          # body text and axis labels
INK_MUTED = "#6b6b6b"    # secondary annotations, footnotes
RULE = "#d9d9d9"         # gridlines and dividers


def method_color(name: str) -> str:
    """Pick a colour by row name; recognises compare5's row names
    ("4DVarNet NLL ens5", "EnKF k1", ...).

    Returns neutral grey rather than raising when nothing matches -- one grey line
    in a figure beats the whole script crashing, and grey itself is a signal that
    "this row has no registered colour."
    """
    n = name.strip()
    if n.startswith("4DVarNet"):
        return METHOD_COLORS["4DVarNet+var"] if ("NLL" in n or "nll" in n) \
            else METHOD_COLORS["4DVarNet"]
    for k, v in METHOD_COLORS.items():
        if n.startswith(k):
            return v
    return INK_MUTED


def shade(hex_color: str, amount: float = 0.55) -> str:
    """Mix a colour toward white, for a method's ablation/single-member row.
    amount=0 is the original colour, 1 is pure white."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    mix = lambda c: int(round(c + (255 - c) * amount))
    return "#%02x%02x%02x" % (mix(r), mix(g), mix(b))


# --------------------------------------------------------------------------- #
# 2. Semantic -> colormap
# --------------------------------------------------------------------------- #
#: One colormap represents exactly one quantity. Add a line here for a new
#: semantic, don't pick one ad hoc inside a plotting script.
CMAP = {
    "density":   "Blues",      # crowd density, sequential, zero is white
    "abs_error": "Purples",    # |error|, sequential -- deliberately not Reds,
                                # that one is reserved for spread
    "spread":    "Reds",       # ensemble spread / sigma-hat, sequential
    "signed":    "RdBu_r",     # signed quantities (vx, vy, residuals), zero-centred
    "mask":      "gray",       # binary figures like walkable / observation masks
}

#: Fill colour for undefined cells. 88.4% of blind velocity cells fall in this
#: category; plotting it is what makes "why the convention decides the winner"
#: visible.
UNDEFINED = "#ececec"


# --------------------------------------------------------------------------- #
# 3. Sizing and rcParams
# --------------------------------------------------------------------------- #
#: Paper text width (inches). A4 with 2.5cm margins is about 6.3in. Every figure
#: is produced at this width; **do not rescale it** when inserting into a
#: document -- rescaling is exactly where the 23 different font sizes came from.
TEXT_W = 6.3
#: Slide width (16:9 content area)
SLIDE_W = 9.5

#: Only three font-size tiers. That is enough; if it isn't, the fix is to split
#: the figure, not to add another size.
FS_TITLE, FS_LABEL, FS_TICK = 10.0, 9.0, 8.0


def use(dpi: int = 200):
    """Tune rcParams. Call once per process."""
    plt.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "savefig.bbox": "tight",
        "font.size": FS_LABEL,
        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_TICK,
        "axes.edgecolor": RULE,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,           # grid below the data, or bars get sliced by it
        "grid.color": RULE,
        "grid.linewidth": 0.6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def figure(ncols: int = 1, nrows: int = 1, width: float = TEXT_W,
           rows_h: float = 2.4, **kw):
    """Build a figure at a fixed width. Height scales with the row count; width
    is not left to the caller to choose -- that is the whole point."""
    return plt.subplots(nrows, ncols, figsize=(width, rows_h * nrows), **kw)


def save(fig, path: str):
    """Save the figure and print its path (every script used to print its own
    "[figure] ..." line; unified here)."""
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"[figure] {path}", flush=True)
