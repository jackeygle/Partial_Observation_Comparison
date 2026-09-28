"""Thesis figure: which observations each method uses to reconstruct frame t.

    python3 -m thesis_figures.time_windows      # -> thesis_figures/out/time_windows.{pdf,png}

One row per method, one square per second relative to the reconstructed frame t.
Filled = observations of that second are an input; the thick square is frame t.
4DVarNet solves fixed 200-s tiles of the day ([0, 200), [200, 400), ...), so its window
reaches past both ends of the axis and t can sit anywhere in it. The EnKF carries all
past observations forward in its ensemble; its forecast model reads the last five
analysed states, not raw observations.
"""
from __future__ import annotations

import os

from matplotlib.patches import FancyArrowPatch, Rectangle

from thesis_figures.common import FS_SMALL, GREY, LINE, canvas, method_color, save, text

HERE = os.path.dirname(os.path.abspath(__file__))
LO, HI = -20, 4                    # displayed offsets relative to t
SQ, PITCH = 0.2, 0.255             # square size and spacing (cm)


def main() -> None:
    rows = [  # name, set of used offsets, extends left, extends right, causal, sigma, note
        ("Senseiver-A", {0}, False, False, True, False, "current second only"),
        ("Senseiver-G (ours)", set(range(-15, 1)), False, False, True, False, "last 16 s"),
        ("DINCAE", {-1, 0, 1}, False, False, False, True, "one second either side"),
        ("4DVarNet", set(range(LO, HI + 1)), True, True, False, False,
         r"fixed 200-s tile containing $t$"),
        ("4DVarNet (aug. head)", set(range(LO, HI + 1)), True, True, False, True,
         r"fixed 200-s tile containing $t$"),
        ("EnKF", set(range(LO, 1)), True, False, True, True, "all past, carried by the ensemble"),
    ]
    rh = 0.52
    page_h = 1.15 + len(rows) * rh + 0.2
    fig, ax = canvas(16.0, page_h)
    x0 = 3.85                                        # left edge of the time axis
    xt = lambda off: x0 + (off - LO) * PITCH         # square's left edge for an offset
    top = page_h - 0.75

    # column heads
    text(ax, xt(0) + SQ / 2, top + 0.42, r"$t$", 9, ha="center", va="bottom")
    for off in (-20, -15, -10, -5, 5):
        if LO <= off <= HI:
            text(ax, xt(off) + SQ / 2, top + 0.42, rf"$t{off:+d}$", FS_SMALL, color=GREY,
                 ha="center", va="bottom")
    cx_causal, cx_sig, cx_note = xt(HI) + 0.8, xt(HI) + 1.55, xt(HI) + 2.1
    text(ax, cx_causal, top + 0.42, "causal", FS_SMALL, ha="center", va="bottom")
    text(ax, cx_sig, top + 0.42, r"$\hat{\sigma}$", 9, ha="center", va="bottom")

    for i, (name, used, left, right, causal, sig, note) in enumerate(rows):
        y = top - i * rh                              # row centre
        col = method_color(name)
        text(ax, 0.2, y, name, va="center")
        for off in range(LO, HI + 1):
            filled = off in used
            face = col if filled else "white"
            ax.add_patch(Rectangle((xt(off), y - SQ / 2), SQ, SQ, fc=face,
                                   ec=LINE if off == 0 else (col if filled else "#c8c8c8"),
                                   lw=1.1 if off == 0 else 0.4, zorder=3 if off == 0 else 2))
        for side, on in ((-1, left), (1, right)):
            if on:
                xs = xt(LO) - 0.08 if side < 0 else xt(HI) + SQ + 0.08
                ax.add_patch(FancyArrowPatch((xs, y), (xs + side * 0.35, y), color=col, lw=0.9,
                                             arrowstyle="-|>,head_length=2.5,head_width=1.3",
                                             mutation_scale=1, shrinkA=0, shrinkB=0))
        text(ax, cx_causal, y, "yes" if causal else "no", FS_SMALL,
             color=LINE if causal else GREY, ha="center", va="center")
        text(ax, cx_sig, y, r"$\checkmark$" if sig else "", 9, ha="center", va="center")
        text(ax, cx_note, y, note, FS_SMALL, color=GREY, va="center")

    # legend
    ly = 0.45
    ax.add_patch(Rectangle((x0, ly - SQ / 2), SQ, SQ, fc=GREY, ec="none"))
    text(ax, x0 + 0.32, ly, "observations used", FS_SMALL, va="center")
    ax.add_patch(Rectangle((x0 + 2.6, ly - SQ / 2), SQ, SQ, fc="white", ec=LINE, lw=1.1))
    text(ax, x0 + 2.92, ly, "reconstructed frame $t$", FS_SMALL, va="center")
    save(fig, os.path.join(HERE, "out", "time_windows"))


if __name__ == "__main__":
    main()
