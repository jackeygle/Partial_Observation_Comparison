"""
plot_de_application.py  —  where the deep-ensembles machinery attaches to our model

One diagram, whose whole job is to show how little has to move. The argument the slide makes is
that adopting the method touches the TRAINING LOSS and adds one head, and leaves the variational
cost, the prior and the solver exactly as 4DVarNet defines them — which matters because those
are the parts the reproduction was spent verifying.

The solver/cost loop is drawn vertically so the two arrows between them stay visible: the solver
hands the cost a candidate x, the cost hands back grad_x J, twenty times.

    de_application.png
"""
from __future__ import annotations
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = os.path.join(ROOT, "check_outputs", "eval")
INK, MUTED = "#20334d", "#5b6a7d"
KEEP, NEW = "#5b6a7d", "#0e6b8a"

fig, ax = plt.subplots(figsize=(12.6, 5.0))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
fig.subplots_adjust(0, 0, 1, 1)


def box(x, y, w, h, text, col, fs=10.5, alpha=0.10, lw=1.5):
    for kw in (dict(fc=col, ec=col, alpha=alpha, zorder=1),
               dict(fc="none", ec=col, zorder=3)):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012", lw=lw, **kw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            zorder=4, linespacing=1.55)


def arrow(x1, y1, x2, y2, col=MUTED, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 lw=1.4, color=col, zorder=5, linestyle=ls))


# ── everything inside this frame is untouched ─────────────────────────────────
ax.add_patch(Rectangle((0.025, 0.12), 0.545, 0.74, fc=KEEP, ec=KEEP, alpha=0.05,
                       lw=1.2, ls="--", zorder=0))
ax.text(0.2975, 0.895, "unchanged — exactly as 4DVarNet defines it",
        ha="center", fontsize=11.5, color=MUTED, style="italic")

box(0.045, 0.60, 0.115, 0.18, "$x_0$, $y$, $\\Omega$", KEEP, 11.5)
box(0.205, 0.56, 0.150, 0.26, "solver\n20 iterations", KEEP)
box(0.205, 0.17, 0.150, 0.21, "$J(x)$\nvariational cost", KEEP, 10)
box(0.410, 0.60, 0.130, 0.18, "$\\hat{x}$\nreconstruction", KEEP, 10)

arrow(0.163, 0.69, 0.203, 0.69)
arrow(0.358, 0.69, 0.408, 0.69)
# the twenty-step loop, drawn vertically so both arrows stay legible
arrow(0.243, 0.555, 0.243, 0.385)
arrow(0.317, 0.385, 0.317, 0.555)
ax.text(0.222, 0.470, "$x$", fontsize=11, color=MUTED, ha="right")
ax.text(0.338, 0.470, "$\\nabla_{x} J$", fontsize=11, color=MUTED, ha="left")

# ── the addition ──────────────────────────────────────────────────────────────
box(0.620, 0.575, 0.175, 0.23, "variance head\n(pointwise)", NEW, 10.5, alpha=0.14, lw=2.0)
arrow(0.543, 0.69, 0.617, 0.69, col=NEW)
arrow(0.798, 0.69, 0.845, 0.69, col=NEW)
ax.text(0.862, 0.69, "$\\hat{\\sigma}^{2}$", fontsize=17, color=NEW, va="center")
ax.text(0.7075, 0.855, "NEW", ha="center", fontsize=12, color=NEW, fontweight="bold")

# ── the loss ──────────────────────────────────────────────────────────────────
box(0.455, 0.135, 0.500, 0.155,
    "training loss:   Eq. 14  (MSE)   $\\rightarrow$   Gaussian NLL", NEW, 12,
    alpha=0.14, lw=2.0)
arrow(0.478, 0.595, 0.560, 0.295, col=NEW, ls=":")
arrow(0.706, 0.570, 0.760, 0.295, col=NEW, ls=":")
ax.text(0.705, 0.325, "the only other change", ha="center", fontsize=10.5, color=NEW)

fig.text(0.5, -0.015,
         "The gradient the solver descends is still the gradient of the variational cost, "
         "computed from observations alone — 4DVarNet Sec. 3.4 is preserved. "
         "$\\hat{\\sigma}^{2}$ is produced after the solve and never enters $J(x)$.",
         ha="center", fontsize=10, color=MUTED)

p = os.path.join(EV, "de_application.png")
fig.savefig(p, dpi=190, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"[figure] {p}")
