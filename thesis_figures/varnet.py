"""Thesis figure: 4DVarNet and the augmented-variance 4DVarNet (aug. var.).

    python3 -m thesis_figures.varnet            # -> thesis_figures/out/varnet.{pdf,png}

(a) 4DVarNet. The variational cost J(x) = a_obs^2 ||(x - y) * Omega||^2 + a_reg^2 ||x - Phi(x)||^2
over a 200-s window (Phi: two-scale GENN prior) is minimised by a learned gradient descent:
the gradient from automatic differentiation goes through a ConvLSTM (3x3, 64 channels) whose
hidden state h_k a 1x1-convolution head turns into the update u_k; K = 20 iterations. Trained
end to end on the MSE to the truth.
(b) The aug. var. model (ours), same skeleton. What changes is drawn in the method's colour:
the cost, whose two terms become Gaussian NLLs in s = log sigma^2; a second 1x1-convolution
head on [h_k, x] that updates s; the training loss (Gaussian NLL). The ConvLSTM now receives
the gradient with respect to both x and s. Shapes are those of the packaged checkpoints
(grad_net.out: 64 -> 800, grad_net.out_var: 864 -> 800; 800 = 4 channels x 200 frames).
"""
from __future__ import annotations

import os

import numpy as np
from matplotlib.patches import Rectangle

from compare import plotstyle as ps
from thesis_figures.common import (FS, FS_SMALL, GREY, LINE, arrow, block, canvas, field_rgb,
                                   method_color, save, text, thumbnail)

HERE = os.path.dirname(os.path.abspath(__file__))
AUG = method_color("4DVarNet aug. var. (ours)")
CELL = 0.05
# x layout shared by both rows (cm)
X_IN, X_COST, W_COST = 0.3, 1.55, 3.65
W_GR, W_LS, W_HD, W_UP, GAP = 1.15, 1.45, 1.25, 1.85, 0.3
BH = 0.9                                              # block height


def inputs(ax, d, walk, seen, vmax, yc, note, note_color=GREY):
    for k in (2, 1):
        ax.add_patch(Rectangle((X_IN + k * 0.13, yc - 0.9 + k * 0.13), 12 * CELL, 36 * CELL,
                               fc="white", ec=LINE, lw=0.5))
    thumbnail(ax, field_rgb(d["observation"][0], walk, ps.CMAP["density"], vmax, hide=~seen),
              X_IN, yc - 0.9, CELL)
    text(ax, X_IN + 0.43, yc + 1.2, r"$y,\ \Omega$" "\n200 s", FS_SMALL, ha="center", va="bottom",
         linespacing=1.1)
    text(ax, X_IN - 0.15, yc - 1.0, note, FS_SMALL - 0.5, color=note_color, va="top",
         linespacing=1.15)


def main() -> None:
    d = np.load(os.path.join(HERE, "data", "example_frame.npz"))
    names = list(d["method_names"]); snames = list(d["spread_method_names"])
    walk, seen = d["walkable"], d["observed"]
    vmax = float(np.percentile(d["truth"][0][walk], 99))
    fig, ax = canvas(16.0, 8.85)
    x_gr = X_COST + W_COST + GAP
    x_ls = x_gr + W_GR + GAP
    x_hd = x_ls + W_LS + 0.55
    x_up = x_hd + W_HD + 0.5
    x_out = x_up + W_UP + 0.4

    # ================= (a) 4DVarNet =====================================================
    ya = 6.3
    text(ax, 0.15, ya + 2.25, "(a) 4DVarNet", FS, va="bottom")
    inputs(ax, d, walk, seen, vmax, ya, r"$x^{(0)}$: observed" "\nvalues, gaps filled")
    cost = block(ax, X_COST, ya - 0.6, W_COST, 1.2, "", None)
    cx = (cost[0] + cost[2]) / 2
    text(ax, cx, ya + 0.27, r"$J(x) = \alpha_{obs}^2\,\|(x-y)\odot\Omega\|^2$", FS - 0.5, ha="center",
         va="center")
    text(ax, cx + 0.2, ya - 0.22, r"$+\ \alpha_{reg}^2\,\|x-\Phi(x)\|^2$", FS - 0.5, ha="center",
         va="center")
    phi = block(ax, cx - 1.1, ya + 1.1, 2.2, 0.75, r"prior $\Phi$", "two-scale GENN")
    arrow(ax, (cx, phi[1] - 0.03), (cx, cost[3] + 0.03))
    arrow(ax, (X_IN + 0.95, ya), (cost[0] - 0.05, ya))
    gr = block(ax, x_gr, ya - BH / 2, W_GR, BH, r"$\nabla_x J$", "autograd")
    ls = block(ax, x_ls, ya - BH / 2, W_LS, BH, "ConvLSTM", r"$3\times3$, 64 ch.", size=FS - 0.5)
    hd = block(ax, x_hd, ya - BH / 2, W_HD, BH, "head", r"$1\times1$ conv", size=FS - 0.5)
    up = block(ax, x_up, ya - BH / 2, W_UP, BH, r"$x \leftarrow x - u_k / K$", size=FS - 0.5)
    arrow(ax, (cost[2] + 0.05, ya), (gr[0] - 0.05, ya))
    arrow(ax, (gr[2] + 0.05, ya), (ls[0] - 0.05, ya))
    arrow(ax, (ls[2] + 0.05, ya), (hd[0] - 0.05, ya))
    arrow(ax, (hd[2] + 0.05, ya), (up[0] - 0.05, ya))
    text(ax, (ls[2] + hd[0]) / 2, ya + 0.1, r"$h_k$", FS_SMALL, ha="center", va="bottom")
    text(ax, (hd[2] + up[0]) / 2, ya + 0.1, r"$u_k$", FS_SMALL, ha="center", va="bottom")
    ly = ya - 1.05                                      # loop back to the cost
    ux = (up[0] + up[2]) / 2
    ax.plot([ux, ux], [up[1], ly], color=LINE, lw=0.7)
    ax.plot([ux, cx], [ly, ly], color=LINE, lw=0.7)
    arrow(ax, (cx, ly), (cx, cost[1] - 0.03))
    text(ax, (cost[2] + up[0]) / 2, ly - 0.08, r"repeat $K = 20$ times", FS_SMALL, color=GREY,
         ha="center", va="top")
    arrow(ax, (up[2] + 0.05, ya), (x_out - 0.08, ya))
    pred = d["predictions"][names.index("4DVarNet MSE")][0]
    e1 = thumbnail(ax, field_rgb(pred, walk, ps.CMAP["density"], vmax), x_out, ya - 0.9, CELL)
    text(ax, (e1[0] + e1[1]) / 2, e1[2] - 0.08, r"$\hat{x}$", 9, ha="center")
    lo = block(ax, e1[1] + 0.95, ya - BH / 2, 1.45, BH, "MSE", "to the truth")
    arrow(ax, (e1[1] + 0.06, ya), (lo[0] - 0.05, ya), color=GREY)
    text(ax, (lo[0] + lo[2]) / 2, lo[1] - 0.08, "training loss", FS_SMALL - 0.5, color=GREY,
         ha="center", va="top")

    # ================= (b) aug. var. ====================================================
    yb = 2.35
    text(ax, 0.15, yb + 1.85, "(b) 4DVarNet aug. var. (ours)", FS, va="bottom")
    text(ax, 5.0, yb + 1.85, r"state $[x,\ s]$, $s = \log\sigma^2$; changed parts in colour", FS_SMALL,
         color=AUG, va="bottom")
    inputs(ax, d, walk, seen, vmax, yb - 0.15, r"$x^{(0)}$ as in (a)" "\n" r"$s^{(0)}$: $\sigma = 0.22$",
           note_color=AUG)
    cost = block(ax, X_COST, yb - 0.75, W_COST, 1.35, "", None, edge=AUG, lw=0.9)
    text(ax, cx, yb + 0.22, r"$J(x,s) = \alpha_{obs}^2\, \Sigma_{\Omega}\, [(x-y)^2 e^{-s} + s]$",
         FS - 1, ha="center", va="center")
    text(ax, cx + 0.12, yb - 0.33, r"$+\ \alpha_{reg}^2\, \Sigma\, [(x-\Phi(x))^2 e^{-s} + s]$", FS - 1,
         ha="center", va="center")
    text(ax, cost[0], cost[3] + 0.06, r"same prior $\Phi$; both terms are Gaussian NLLs in $s$",
         FS_SMALL - 0.5, color=AUG, va="bottom")
    arrow(ax, (X_IN + 0.95, yb - 0.08), (cost[0] - 0.05, yb - 0.08))
    ym = yb - 0.08                                      # mid-height of the cost box
    gr = block(ax, x_gr, ym - BH / 2, W_GR, BH, r"$\nabla_{x,s} J$", "autograd", edge=AUG, lw=0.9)
    ls = block(ax, x_ls, ym - BH / 2, W_LS, BH, "ConvLSTM", r"$3\times3$, 64 ch.", size=FS - 0.5)
    arrow(ax, (cost[2] + 0.05, ym), (gr[0] - 0.05, ym))
    arrow(ax, (gr[2] + 0.05, ym), (ls[0] - 0.05, ym))
    # two heads: the state's (as in (a)) and the variance's (new)
    yh, yv = ym + 0.62, ym - 0.62
    hd = block(ax, x_hd, yh - 0.4, W_HD, 0.8, "head", r"$1\times1$ conv", size=FS - 0.5)
    vh = block(ax, x_hd, yv - 0.4, W_HD, 0.8, "var. head", r"$1\times1$ conv", edge=AUG, lw=0.9,
               size=FS - 0.5)
    upx = block(ax, x_up, yh - 0.4, W_UP, 0.8, r"$x \leftarrow x - u_k / K$", size=FS - 0.5)
    ups = block(ax, x_up, yv - 0.4, W_UP, 0.8, r"$s \leftarrow s - v_k / K$", edge=AUG, lw=0.9,
                size=FS - 0.5)
    jx = ls[2] + 0.36                                   # h_k splits to the two heads
    ax.plot([ls[2] + 0.03, jx], [ym, ym], color=LINE, lw=0.7)
    ax.plot([jx, jx], [yv, yh], color=LINE, lw=0.7)
    arrow(ax, (jx, yh), (hd[0] - 0.05, yh))
    arrow(ax, (jx, yv), (vh[0] - 0.05, yv), color=AUG)
    text(ax, ls[2] + 0.18, ym + 0.08, r"$h_k$", FS_SMALL, ha="center", va="bottom")
    arrow(ax, (hd[2] + 0.05, yh), (upx[0] - 0.05, yh))
    arrow(ax, (vh[2] + 0.05, yv), (ups[0] - 0.05, yv), color=AUG)
    text(ax, (hd[2] + upx[0]) / 2, yh + 0.08, r"$u_k$", FS_SMALL, ha="center", va="bottom")
    text(ax, (vh[2] + ups[0]) / 2, yv + 0.08, r"$v_k$", FS_SMALL, color=AUG, ha="center", va="bottom")
    text(ax, (vh[0] + vh[2]) / 2, vh[1] - 0.07, r"input $[h_k,\ x]$", FS_SMALL - 0.5, color=AUG,
         ha="center", va="top")
    ly = ups[1] - 0.55                                  # loop: the new [x, s] goes back to J
    ux = (ups[0] + ups[2]) / 2
    ax.add_patch(Rectangle((upx[0] - 0.1, ups[1] - 0.1), W_UP + 0.2, upx[3] - ups[1] + 0.2, fill=False,
                           ec=GREY, lw=0.5, ls=(0, (2, 1.5))))
    ax.plot([ux, ux], [ups[1] - 0.1, ly], color=LINE, lw=0.7)
    ax.plot([ux, cx], [ly, ly], color=LINE, lw=0.7)
    arrow(ax, (cx, ly), (cx, cost[1] - 0.03))
    text(ax, (cost[2] + ups[0]) / 2, ly - 0.08, r"repeat $K = 20$ times with the new $[x,\ s]$", FS_SMALL,
         color=GREY, ha="center", va="top")
    # outputs
    sa = d["spreads"][snames.index("4DVarNet aughead_obs (single)")]
    smax = float(np.percentile(sa[walk], 99))
    pa = d["predictions"][names.index("4DVarNet aughead_obs (single)")][0]
    e2 = thumbnail(ax, field_rgb(pa, walk, ps.CMAP["density"], vmax), x_out, ym - 0.9, CELL)
    e3 = thumbnail(ax, field_rgb(sa, walk, ps.CMAP["spread"], smax), e2[1] + 0.15, ym - 0.9, CELL)
    arrow(ax, (upx[2] + 0.12, ym), (x_out - 0.08, ym))
    text(ax, (e2[0] + e2[1]) / 2, e2[2] - 0.08, r"$\hat{x}$", 9, ha="center")
    text(ax, (e3[0] + e3[1]) / 2, e3[2] - 0.08, r"$\hat{\sigma}$", 9, color=AUG, ha="center")
    text(ax, (e3[0] + e3[1]) / 2, e3[2] - 0.5, r"$= e^{s/2}$", FS_SMALL, color=AUG, ha="center")
    lo = block(ax, e3[1] + 0.2, ym - BH / 2, 1.45, BH, "NLL", "of the truth", edge=AUG, lw=0.9)
    text(ax, (lo[0] + lo[2]) / 2, lo[1] - 0.08, "training loss", FS_SMALL - 0.5, color=GREY,
         ha="center", va="top")

    save(fig, os.path.join(HERE, "out", "varnet"))


if __name__ == "__main__":
    main()
