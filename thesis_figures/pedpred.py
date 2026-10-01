"""Thesis figure: the PedPred3 forecast model of the EnKF.

    python3 -m thesis_figures.pedpred           # -> thesis_figures/out/pedpred.{pdf,png}

What methods/enkf/enkf_opt/pedpred/models.py builds (PedPred3, EncoderForcaster):

encoder, run once per input frame: conv 3x3 4->16 (36x12), conv 3x3 stride 2 16->32 (18x6),
conv 3x3 stride 2 32->64 (9x3), convolutional GRU with 64 hidden channels whose state is
carried from one input frame to the next.
forecaster, run once per output frame: a second convolutional GRU (64 channels, no input,
its state starts as the encoder's final state), transposed conv 4x4 stride 2 64->64 (18x6),
transposed conv 4x4 stride 2 64->64 (36x12), transposed conv 3x3 64->16, conv 3x3 16->16,
conv 1x1 16->4. Leaky ReLU (0.2) after every convolution but the last.
output: log density, mean velocity (2), log velocity variance. Trained 5 frames -> 5 frames;
the filter uses the first output frame (horizon 1).
"""
from __future__ import annotations

import os

from matplotlib.patches import Rectangle

from thesis_figures.common import (FS, FS_SMALL, GREY, LINE, arrow, block, canvas, method_color,
                                   save, text)

HERE = os.path.dirname(os.path.abspath(__file__))
C = method_color("EnKF")
BW, BH, GAP = 1.95, 1.3, 0.3


def cards(ax, x, y, w, h, n, off=0.09, color=LINE):
    for k in range(n - 1, -1, -1):
        ax.add_patch(Rectangle((x + k * off, y + k * off), w, h, fc="white", ec=color, lw=0.5))
    return (x, y, x + w + (n - 1) * off, y + h + (n - 1) * off)


def main() -> None:
    fig, ax = canvas(16.0, 7.5)
    y_e, y_f = 5.2, 1.85                               # encoder / forecaster rows
    xs = [2.6 + i * (BW + GAP) for i in range(6)]      # six columns of boxes

    # encoder (left to right), in the four right-hand columns --------------------------------
    text(ax, 0.15, y_e + 1.45, "encoder: run once for each of the five input frames", FS,
         va="bottom")
    inp = cards(ax, 0.85, y_e - 0.75, 0.5, 1.5, n=5)
    text(ax, (inp[0] + inp[2]) / 2, inp[1] - 0.1,
         "last 5 states\n" r"$x_{t-5}, \ldots, x_{t-1}$" "\n" r"4 ch., $36 \times 12$", FS_SMALL,
         ha="center", va="top", linespacing=1.15)
    enc = [("conv", "$3 \\times 3$\n$4 \\rightarrow 16$\n$36 \\times 12$"),
           ("conv", "$3 \\times 3$, stride 2\n$16 \\rightarrow 32$\n$18 \\times 6$"),
           ("conv", "$3 \\times 3$, stride 2\n$32 \\rightarrow 64$\n$9 \\times 3$")]
    prev = inp[2] + 0.1
    for i, (name, sub) in enumerate(enc):
        b = block(ax, xs[i + 2], y_e - BH / 2, BW, BH, name, sub, size=FS_SMALL)
        arrow(ax, (prev, y_e), (b[0] - 0.05, y_e))
        prev = b[2] + 0.05
    g1 = block(ax, xs[5], y_e - BH / 2, BW, BH, "conv. GRU", "64 hidden ch.\n$9 \\times 3$",
               edge=C, lw=0.9, size=FS_SMALL)
    arrow(ax, (prev, y_e), (g1[0] - 0.05, y_e))
    # recurrence: the state is carried to the next input frame
    xa, xb, yl = g1[0] + 0.5, g1[2] - 0.5, g1[3] + 0.35
    ax.plot([xb, xb, xa], [g1[3], yl, yl], color=C, lw=0.7)
    arrow(ax, (xa, yl), (xa, g1[3] + 0.02), color=C)
    text(ax, (xa + xb) / 2, yl + 0.06, "state $h$, carried\nto the next frame", FS_SMALL - 0.5,
         color=C, ha="center", va="bottom", linespacing=1.1)

    # hand-over ------------------------------------------------------------------------------
    xm = (g1[0] + g1[2]) / 2
    arrow(ax, (xm, g1[1] - 0.03), (xm, y_f + BH / 2 + 0.03), color=C)
    text(ax, xm - 0.12, (g1[1] + y_f + BH / 2) / 2, "$h$ after the\nfifth frame", FS_SMALL - 0.5,
         color=C, ha="right", va="center", linespacing=1.1)

    # forecaster (right to left) -------------------------------------------------------------
    text(ax, 0.15, y_f + 1.15, "forecaster: run once for each output frame", FS, va="bottom")
    dec = [("conv. GRU", "64 hidden ch.\nno input\n$9 \\times 3$"),
           ("transp. conv", "$4 \\times 4$, stride 2\n$64 \\rightarrow 64$\n$18 \\times 6$"),
           ("transp. conv", "$4 \\times 4$, stride 2\n$64 \\rightarrow 64$\n$36 \\times 12$"),
           ("transp. conv", "$3 \\times 3$\n$64 \\rightarrow 16$\n$36 \\times 12$"),
           ("conv", "$3 \\times 3$\n$16 \\rightarrow 16$\n$36 \\times 12$"),
           ("conv", "$1 \\times 1$\n$16 \\rightarrow 4$\n$36 \\times 12$")]
    prev = None
    for i, (name, sub) in enumerate(dec):
        col = 5 - i
        b = block(ax, xs[col], y_f - BH / 2, BW, BH, name, sub, size=FS_SMALL,
                  **(dict(edge=C, lw=0.9) if i == 0 else {}))
        if prev is not None:
            arrow(ax, (prev, y_f), (b[2] + 0.05, y_f))
        prev = b[0] - 0.05
    out = cards(ax, 0.85, y_f - 0.75, 0.5, 1.5, n=1)
    arrow(ax, (prev, y_f), (out[2] + 0.1, y_f))
    text(ax, (out[0] + out[2]) / 2, out[1] - 0.1,
         r"forecast of $x_t$:" "\n" r"$\log\rho,\ v_x,\ v_y,\ \log\nu$", FS_SMALL, ha="center",
         va="top", linespacing=1.15)
    text(ax, 15.85, 0.1, "leaky ReLU (slope 0.2) after every convolution except the last;"
         " GRU kernels: $3 \\times 3$ on the input, $5 \\times 5$ on the state",
         FS_SMALL - 0.5, color=GREY, ha="right", va="bottom")

    save(fig, os.path.join(HERE, "out", "pedpred"))


if __name__ == "__main__":
    main()
