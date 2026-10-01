"""Thesis figure: inside one Senseiver encoder block.

    python3 -m thesis_figures.attention_block   # -> thesis_figures/out/attention_block.{pdf,png}

What methods/senseiver/model.py builds (cross_attention_layer, self_attention_layer, mlp,
Residual, CrossAttention, SelfAttention), identical in Senseiver-A and Senseiver-G:

(a) cross-attention layer: the latents (queries) and the sensor tokens (keys, values) are
    each layer-normalised, multi-head attention (2 heads) follows, and its output is added
    to the un-normalised latents; then LayerNorm, Linear 32->32, GELU, Linear 32->32, added
    to its own input.
(b) self-attention layer, three in a row: the same, with queries = keys = values = the
    normalised latents.

One encoder block = (a) followed by 3 x (b); the encoder runs three blocks.
"""
from __future__ import annotations

import os

from matplotlib.patches import Circle, Rectangle

from thesis_figures.common import FS, FS_SMALL, GREY, LINE, arrow, block, canvas, save, text

HERE = os.path.dirname(os.path.abspath(__file__))
BH = 0.8
GAP = 0.28
R = 0.17


def plus(ax, x, y):
    ax.add_patch(Circle((x, y), R, fc="white", ec=LINE, lw=0.6, zorder=3))
    ax.plot([x - 0.09, x + 0.09], [y, y], color=LINE, lw=0.6, zorder=4)
    ax.plot([x, x], [y - 0.09, y + 0.09], color=LINE, lw=0.6, zorder=4)


def skip(ax, x_from, x_to, yc):
    """Residual connection: leaves the main line at x_from, runs above the boxes, ends in the adder."""
    y = yc + BH / 2 + 0.3
    ax.plot([x_from, x_from, x_to], [yc, y, y], color=GREY, lw=0.6)
    arrow(ax, (x_to, y), (x_to, yc + R + 0.02), color=GREY, lw=0.6)
    ax.add_patch(Circle((x_from, yc), 0.035, fc=LINE, ec=LINE, zorder=4))


def layer(ax, yc, cross: bool) -> None:
    x = 1.75
    text(ax, 0.2, yc, "latents\n" r"$N \times 32$", FS_SMALL, ha="left", va="center",
         linespacing=1.15)
    x_in = 1.05
    ln1 = block(ax, x, yc - BH / 2, 1.45, BH, "LayerNorm", size=FS_SMALL)
    ax.plot([x_in, ln1[0] - 0.05], [yc, yc], color=LINE, lw=0.7)
    arrow(ax, (ln1[0] - 0.3, yc), (ln1[0] - 0.05, yc))
    at = block(ax, ln1[2] + 0.95, yc - BH / 2, 2.6, BH, "multi-head attention", "2 heads",
               size=FS_SMALL)
    arrow(ax, (ln1[2] + 0.05, yc), (at[0] - 0.05, yc))
    text(ax, (ln1[2] + at[0]) / 2, yc + 0.08, "Q" if cross else "Q, K, V", FS_SMALL - 1, color=GREY,
         ha="center", va="bottom")
    p1 = at[2] + GAP + R
    arrow(ax, (at[2] + 0.05, yc), (p1 - R - 0.02, yc))
    plus(ax, p1, yc)
    skip(ax, x_in + 0.35, p1, yc)

    ln2 = block(ax, p1 + R + GAP + 0.3, yc - BH / 2, 1.45, BH, "LayerNorm", size=FS_SMALL)
    arrow(ax, (p1 + R + 0.02, yc), (ln2[0] - 0.05, yc))
    l1 = block(ax, ln2[2] + GAP, yc - BH / 2, 1.15, BH, "linear", r"$32 \rightarrow 32$", size=FS_SMALL)
    arrow(ax, (ln2[2] + 0.05, yc), (l1[0] - 0.05, yc))
    ge = block(ax, l1[2] + GAP, yc - BH / 2, 0.85, BH, "GELU", size=FS_SMALL)
    arrow(ax, (l1[2] + 0.05, yc), (ge[0] - 0.05, yc))
    l2 = block(ax, ge[2] + GAP, yc - BH / 2, 1.15, BH, "linear", r"$32 \rightarrow 32$", size=FS_SMALL)
    arrow(ax, (ge[2] + 0.05, yc), (l2[0] - 0.05, yc))
    p2 = l2[2] + GAP + R
    arrow(ax, (l2[2] + 0.05, yc), (p2 - R - 0.02, yc))
    plus(ax, p2, yc)
    skip(ax, p1 + R + 0.2, p2, yc)
    arrow(ax, (p2 + R + 0.02, yc), (p2 + R + 0.5, yc))
    text(ax, p2 + R + 0.58, yc, "latents\n" r"$N \times 32$", FS_SMALL, va="center",
         linespacing=1.15)

    # the two halves of the layer
    for x0, x1, name in ((ln1[0], at[2], "attention"), (ln2[0], l2[2], "perceptron")):
        ax.add_patch(Rectangle((x0 - 0.12, yc - BH / 2 - 0.12), x1 - x0 + 0.24, BH + 0.24,
                               fill=False, ec=GREY, lw=0.5, ls=(0, (3, 1.5)), zorder=1))
        if name == "perceptron":
            text(ax, (x0 + x1) / 2, yc - BH / 2 - 0.17, name, FS_SMALL - 0.5, color=GREY,
                 ha="center", va="top")
    text(ax, (p1 + p2) / 2 + 0.2, yc + BH / 2 + 0.34, "residual connection", FS_SMALL - 1,
         color=GREY, ha="center", va="bottom")

    if cross:                                          # sensor tokens enter from below
        xm = (at[0] + at[2]) / 2
        lnk = block(ax, xm - 0.725, yc - BH / 2 - 1.2, 1.45, 0.6, "LayerNorm", size=FS_SMALL)
        arrow(ax, (xm, lnk[3] + 0.03), (xm, at[1] - 0.03))
        text(ax, xm + 0.08, (lnk[3] + at[1]) / 2 - 0.06, "K, V", FS_SMALL - 1, color=GREY,
             va="center")
        arrow(ax, (lnk[0] - 0.55, (lnk[1] + lnk[3]) / 2), (lnk[0] - 0.05, (lnk[1] + lnk[3]) / 2))
        text(ax, lnk[0] - 0.63, (lnk[1] + lnk[3]) / 2, "sensor tokens\n" r"$M \times 32$", FS_SMALL,
             ha="right", va="center", linespacing=1.15)
        text(ax, lnk[2] + 0.25, (lnk[1] + lnk[3]) / 2,
             "attention: every latent collects\ninformation from all sensor tokens", FS_SMALL - 0.5,
             color=GREY, va="center", linespacing=1.15)
    else:
        text(ax, (ln1[0] + at[2]) / 2, yc - BH / 2 - 0.17,
             "attention: every latent collects\ninformation from all latents", FS_SMALL - 0.5,
             color=GREY, ha="center", va="top", linespacing=1.15)


def main() -> None:
    fig, ax = canvas(16.0, 6.9)
    y_a, y_b = 5.05, 1.35
    text(ax, 0.15, y_a + 1.2, "(a) cross-attention layer", FS, va="bottom")
    layer(ax, y_a, cross=True)
    text(ax, 0.15, y_b + 1.2, r"(b) self-attention layer, $\times 3$", FS, va="bottom")
    layer(ax, y_b, cross=False)
    text(ax, 15.85, 0.1, r"one encoder block $=$ (a) followed by $3 \times$ (b)", FS_SMALL,
         color=GREY, ha="right", va="bottom")
    save(fig, os.path.join(HERE, "out", "attention_block"))


if __name__ == "__main__":
    main()
