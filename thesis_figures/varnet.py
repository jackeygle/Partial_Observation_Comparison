"""Thesis figure: 4DVarNet and the aug. head.

    python3 -m thesis_figures.varnet            # -> thesis_figures/out/varnet.{pdf,png}

The variational cost J(x) = a_obs^2 ||(x - y) * Omega||^2 + a_reg^2 ||x - Phi(x)||^2 over a
200-s window, Phi the two-scale GENN prior, minimised by a learned gradient descent: the
gradient from automatic differentiation goes through a ConvLSTM that outputs the update,
for 20 iterations; Phi and the solver are trained end to end on the MSE to the truth.
The aug. head (ours) iterates the state [x, log sigma^2]: both terms become Gaussian NLLs
in that sigma^2, a variance head on [h, x] updates log sigma^2, and training uses the NLL.
"""
from __future__ import annotations

import os

import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from compare import plotstyle as ps
from thesis_figures.common import (FS, FS_SMALL, GREY, LINE, arrow, block, canvas, field_rgb,
                                   method_color, save, text, thumbnail)

HERE = os.path.dirname(os.path.abspath(__file__))
AUG = method_color("4DVarNet (aug. head)")


def main() -> None:
    d = np.load(os.path.join(HERE, "data", "example_frame.npz"))
    names = list(d["method_names"]); snames = list(d["spread_method_names"])
    walk, seen = d["walkable"], d["observed"]
    vmax = float(np.percentile(d["truth"][0][walk], 99))
    cell = 0.05
    fig, ax = canvas(16.0, 6.2)
    yc = 3.75

    # ---- input window -------------------------------------------------------------------
    for k in (2, 1):
        ax.add_patch(Rectangle((0.3 + k * 0.16, yc - 0.9 + k * 0.16), 12 * cell, 36 * cell,
                               fc="white", ec=LINE, lw=0.5))
    thumbnail(ax, field_rgb(d["observation"][0], walk, ps.CMAP["density"], vmax, hide=~seen),
              0.3, yc - 0.9, cell)
    text(ax, 0.8, yc + 1.25, r"$y,\ \Omega$ over a" "\n200-s window", FS_SMALL, ha="center",
         va="bottom", linespacing=1.1)
    text(ax, 0.8, yc - 1.05, r"$x^{(0)}$: observed" "\nvalues, gaps filled", FS_SMALL - 0.5,
         color=GREY, ha="center", va="top", linespacing=1.1)

    # ---- the cost, with the prior above it -----------------------------------------------
    cost = block(ax, 2.1, yc - 0.65, 3.9, 1.3, "", None)
    text(ax, 4.05, yc + 0.3, r"$J(x) = \alpha_{obs}^2\,\|(x-y)\odot\Omega\|^2$", FS, ha="center",
         va="center")
    text(ax, 4.25, yc - 0.2, r"$+\ \alpha_{reg}^2\,\|x-\Phi(x)\|^2$", FS, ha="center", va="center")
    text(ax, cost[0], cost[1] - 0.08, "variational cost", FS_SMALL - 0.5, color=GREY, va="top")
    phi = block(ax, 3.2, yc + 1.2, 2.3, 0.75, r"prior $\Phi$", "two-scale GENN")
    arrow(ax, (4.35, phi[1] - 0.03), (4.35, cost[3] + 0.03))
    arrow(ax, (1.75, yc), (cost[0] - 0.05, yc))

    # ---- learned gradient descent ----------------------------------------------------------
    gr = block(ax, cost[2] + 0.4, yc - 0.45, 1.55, 0.9, r"$\nabla_x J$", "autograd")
    arrow(ax, (cost[2] + 0.05, yc), (gr[0] - 0.05, yc))
    ls = block(ax, gr[2] + 0.4, yc - 0.45, 1.65, 0.9, "ConvLSTM", r"update $u_k$")
    arrow(ax, (gr[2] + 0.05, yc), (ls[0] - 0.05, yc))
    up = block(ax, ls[2] + 0.4, yc - 0.45, 1.5, 0.9, r"$x \leftarrow x - u_k$")
    arrow(ax, (ls[2] + 0.05, yc), (up[0] - 0.05, yc))
    # loop back to the cost
    ly = yc - 1.05
    ax.plot([(up[0] + up[2]) / 2, (up[0] + up[2]) / 2], [up[1], ly], color=LINE, lw=0.7)
    ax.plot([(up[0] + up[2]) / 2, (cost[0] + cost[2]) / 2], [ly, ly], color=LINE, lw=0.7)
    arrow(ax, ((cost[0] + cost[2]) / 2, ly), ((cost[0] + cost[2]) / 2, cost[1] - 0.03))
    text(ax, (cost[2] + up[0]) / 2 + 0.6, ly - 0.08, r"repeat, $k = 1,\dots,20$", FS_SMALL,
         color=GREY, ha="center", va="top")

    # ---- output -----------------------------------------------------------------------------------
    ox = up[2] + 0.65
    arrow(ax, (up[2] + 0.05, yc), (ox - 0.08, yc))
    pred = d["predictions"][names.index("4DVarNet MSE")][0]
    e1 = thumbnail(ax, field_rgb(pred, walk, ps.CMAP["density"], vmax), ox, yc - 0.9, cell)
    text(ax, (e1[0] + e1[1]) / 2, e1[2] - 0.08, r"$\hat{x}$", 9, ha="center")
    block(ax, e1[1] + 0.3, yc - 0.45, 1.45, 0.9, "MSE", "to the truth")
    arrow(ax, (e1[1] + 0.05, yc), (e1[1] + 0.26, yc), color=GREY)
    text(ax, e1[1] + 1.02, yc - 0.55, "training loss", FS_SMALL - 0.5, color=GREY, ha="center",
         va="top")

    # ---- the aug. head ------------------------------------------------------------------------------
    by0, by1 = 0.1, 1.85
    ax.add_patch(Rectangle((2.1, by0), up[2] - 2.1, by1 - by0, fill=False, ec=AUG, lw=0.8,
                           ls=(0, (3, 1.5))))
    text(ax, 2.25, by1 - 0.12, "4DVarNet (aug. head), ours", FS, color=AUG, va="top")
    lines = [r"state $[x,\ s]$ with $s = \log\sigma^2$",
             r"prior term: $(x-\Phi(x))^2 e^{-s} + s$;   observation term: $((x-y)^2e^{-s}+s)\odot\Omega$",
             r"a variance head on $[h,\ x]$ updates $s$ each iteration;  trained with the Gaussian NLL"]
    for j, s_ in enumerate(lines):
        text(ax, 2.25, by1 - 0.55 - j * 0.38, s_, FS_SMALL, va="top")
    sa = d["spreads"][snames.index("4DVarNet aughead_obs (single)")]
    smax = float(np.percentile(sa[walk], 99))
    pa = d["predictions"][names.index("4DVarNet aughead_obs (single)")][0]
    e2 = thumbnail(ax, field_rgb(pa, walk, ps.CMAP["density"], vmax), ox, by0 + 0.05, cell)
    e3 = thumbnail(ax, field_rgb(sa, walk, ps.CMAP["spread"], smax), e2[1] + 0.2, by0 + 0.05, cell)
    text(ax, (e2[0] + e2[1]) / 2, e2[3] + 0.08, r"$\hat{x}$", 9, ha="center", va="bottom")
    text(ax, (e3[0] + e3[1]) / 2, e3[3] + 0.08, r"$\hat{\sigma} = e^{s/2}$", FS_SMALL, ha="center",
         va="bottom")
    ax.add_patch(FancyArrowPatch((up[2], (by0 + by1) / 2), (ox - 0.08, (by0 + by1) / 2),
                                 arrowstyle="-|>,head_length=3.2,head_width=1.6", color=AUG,
                                 lw=0.7, mutation_scale=1))

    save(fig, os.path.join(HERE, "out", "varnet"))


if __name__ == "__main__":
    main()
