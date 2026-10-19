"""Thesis figure: the two-scale GENN prior Phi of 4DVarNet.

    python3 -m thesis_figures.genn              # -> thesis_figures/out/genn.{pdf,png}

(a) Phi(x) = Up(Phi_1(Dw(x))) + Phi_2(x) (Fablet et al., Eq. 10), as in
methods/varnet/prior_model.py with fold_axes="ct": each branch is a 3-D convolution
5x3x3 over (time, row, column) with its centre tap fixed to zero, 4 -> 96, ReLU, then two
1x1x1 convolutions 96 -> 96 (ReLU) and 96 -> 4. The coarse branch runs on the state after
2x2 mean pooling in space (per frame) and is brought back by a 2x2, stride-2 transposed
convolution 4 -> 4 (per frame), then added to the fine branch.
(b) One input/output channel pair of the 5x3x3 kernel: the centre tap is zero, so the output
at a space-time point never reads the input at that point.
"""
from __future__ import annotations

import os

from matplotlib.patches import Circle, Rectangle

from thesis_figures.common import (FIVE_METHODS, SIX_ALLWALK, FS, FS_SMALL, GREY, LINE, arrow, block, canvas,
                                   method_color, save, text)

# temporal kernel of psi: 3 in the five-method thesis's 4DVarNet, 5 otherwise
KT = 3 if (FIVE_METHODS or SIX_ALLWALK) else 5
SLICES = ["t-1", "t", "t+1"] if KT == 3 else ["t-2", "t-1", "t", "t+1", "t+2"]

HERE = os.path.dirname(os.path.abspath(__file__))
COL = method_color("4DVarNet")
BH = 0.95                                   # block height (cm)


def volume(ax, x0, yc, label):
    """A small stack of frames standing for a (4, 200, 36, 12) state window."""
    w, h = 0.42, 1.1
    for k in (3, 2, 1, 0):
        ax.add_patch(Rectangle((x0 + k * 0.1, yc - h / 2 + k * 0.1), w, h, fc="white", ec=LINE,
                               lw=0.5, zorder=2))
    text(ax, x0 + 0.36, yc - h / 2 - 0.1, label, FS_SMALL, ha="center", va="top",
         linespacing=1.15)
    return x0 + w + 0.3                     # right edge


def kernel(ax, x0, y0, cell=0.22, gap=0.3):
    """KT 3x3 time slices of one kernel; the centre tap of the middle slice is zero."""
    for k, lab in enumerate(SLICES):
        xs = x0 + k * (3 * cell + gap)
        for r in range(3):
            for c in range(3):
                zero = k == KT // 2 and r == 1 and c == 1
                ax.add_patch(Rectangle((xs + c * cell, y0 + r * cell), cell, cell,
                                       fc="white" if zero else "#f6e2b3", ec=LINE,
                                       lw=0.9 if zero else 0.4, zorder=3 if zero else 2))
                if zero:
                    text(ax, xs + 1.5 * cell, y0 + 1.5 * cell, "0", FS_SMALL, ha="center",
                         va="center", zorder=4)
        text(ax, xs + 1.5 * cell, y0 - 0.08, rf"${lab}$" if lab != "t" else r"$t$", FS_SMALL,
             ha="center", va="top")
    return x0 + KT * 3 * cell + (KT - 1) * gap


def main() -> None:
    fig, ax = canvas(16.0, 7.0)

    # ================= (a) the two branches =============================================
    yf, yc = 5.35, 3.15                      # row centres: fine, coarse
    text(ax, 0.15, 6.85, r"(a) prior $\Phi(x) = \mathrm{Up}(\Phi_1(\mathrm{Dw}(x)))"
         r" + \Phi_2(x)$", FS, va="top")
    ym = (yf + yc) / 2
    xr = volume(ax, 0.3, ym, r"$x$" "\n" r"$4\times200$" "\n" r"$\times36\times12$")
    xs = xr + 0.25                           # split point
    ax.plot([xr, xs], [ym, ym], color=LINE, lw=0.7)
    ax.plot([xs, xs], [yc, yf], color=LINE, lw=0.7)

    x_pool, w_pool = xs + 0.35, 1.75
    x_psi, w_psi = x_pool + w_pool + 0.45, 2.6
    x_p1, w_p1 = x_psi + w_psi + 0.4, 2.0
    x_p2, w_p2 = x_p1 + w_p1 + 0.4, 2.0
    x_up, w_up = x_p2 + w_p2 + 0.4, 2.0
    x_sum = x_up + w_up + 0.55

    for y, tag in ((yf, "fine"), (yc, "coarse")):
        psi = block(ax, x_psi, y - BH / 2, w_psi, BH, r"$\psi$: 3-D conv.",
                    rf"${KT}\times3\times3$, centre tap 0" "\n" r"$4 \to 96$, ReLU", edge=COL, lw=0.9,
                    size=FS - 0.5)
        p1 = block(ax, x_p1, y - BH / 2, w_p1, BH, r"$1\times1\times1$ conv.",
                   r"$96 \to 96$, ReLU",
                   size=FS - 0.5)
        p2 = block(ax, x_p2, y - BH / 2, w_p2, BH, r"$1\times1\times1$ conv.", r"$96 \to 4$",
                   size=FS - 0.5)
        arrow(ax, (psi[2] + 0.04, y), (p1[0] - 0.04, y))
        arrow(ax, (p1[2] + 0.04, y), (p2[0] - 0.04, y))
        if tag == "fine":
            arrow(ax, (xs, y), (psi[0] - 0.04, y))
            ax.plot([p2[2] + 0.02, x_sum], [y, y], color=LINE, lw=0.7)
            arrow(ax, (x_sum, y), (x_sum, ym + 0.2))
            text(ax, x_psi, y + BH / 2 + 0.12, r"fine scale $\Phi_2$: full $36\times12$ grid",
                 FS_SMALL, color=GREY, va="bottom")
        else:
            pool = block(ax, x_pool, y - BH / 2, w_pool, BH, "mean pool",
                         r"$2\times2$ in space," "\n" "per frame")
            up = block(ax, x_up, y - BH / 2, w_up, BH, "transp. conv.",
                       r"$2\times2$, stride 2," "\n" r"$4\to4$, per frame")
            arrow(ax, (xs, y), (pool[0] - 0.04, y))
            arrow(ax, (pool[2] + 0.04, y), (psi[0] - 0.04, y))
            arrow(ax, (p2[2] + 0.04, y), (up[0] - 0.04, y))
            ax.plot([up[2] + 0.02, x_sum], [y, y], color=LINE, lw=0.7)
            arrow(ax, (x_sum, y), (x_sum, ym - 0.2))
            text(ax, x_psi, y + BH / 2 + 0.12,
                 r"coarse scale $\Phi_1$: $18\times6$ grid, own weights", FS_SMALL, color=GREY,
                 va="bottom")
            text(ax, (pool[0] + pool[2]) / 2, y - BH / 2 - 0.1, r"Dw", FS_SMALL, color=GREY,
                 ha="center", va="top")
            text(ax, (up[0] + up[2]) / 2, y - BH / 2 - 0.1, r"Up", FS_SMALL, color=GREY,
                 ha="center", va="top")
    ax.add_patch(Circle((x_sum, ym), 0.2, fc="white", ec=LINE, lw=0.6, zorder=3))
    text(ax, x_sum, ym + 0.01, "+", FS, ha="center", va="center", zorder=4)
    arrow(ax, (x_sum + 0.22, ym), (x_sum + 0.55, ym))
    volume(ax, x_sum + 0.62, ym, r"$\Phi(x)$" "\n" "same shape")

    # ================= (b) the zero centre tap ==========================================
    text(ax, 0.15, 1.95, r"(b) the kernel of $\psi$ (one input/output channel pair)", FS,
         va="top")
    xk = kernel(ax, 0.6, 0.45)
    text(ax, xk + 0.5, 1.45,
         r"The tap at the output's own point $(t, i, j)$ is fixed to zero, and the layers after"
         "\n"
         r"$\psi$ act point by point, so a branch's output at $s$ never reads $x(s)$. Pooling and"
         "\n"
         r"the stride-2 transposed convolution act per frame, so each fine cell takes one coarse"
         "\n"
         r"output, which excludes its own pooled cell. Hence $\Phi(x)(s)$ does not depend on"
         "\n"
         r"$x(s)$ and $\Phi$ cannot learn the identity, which would make $\|x-\Phi(x)\|^2$ vanish.",
         FS_SMALL, va="top", linespacing=1.3)

    save(fig, os.path.join(HERE, "out", "genn"))


if __name__ == "__main__":
    main()
