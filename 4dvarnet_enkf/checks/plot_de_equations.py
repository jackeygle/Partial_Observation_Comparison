"""
plot_de_equations.py  —  typeset the equations of Lakshminarayanan et al. 2017 for the slides

The deck walks the algorithm equation by equation, so each equation is rendered as its own
image rather than typed as slide text: pptx text boxes cannot set mathematics, and a screenshot
of the PDF would not scale. Every formula below is transcribed from the paper; the equation
numbers are theirs.

  de_eq_score.png    Sec. 2.2   the proper-scoring-rule condition and the resulting loss
  de_eq_nll.png      Eq. 1      the regression criterion, annotated term by term
  de_eq_adv.png      Sec. 2.3   the adversarial example, and the augmented objective
  de_alg1.png        Alg. 1     the training procedure, as the paper writes it
  de_eq_combine.png  Sec. 2.4   the mixture, and its Gaussian moment match
"""
from __future__ import annotations
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = os.path.join(ROOT, "check_outputs", "eval")
INK, MUTED, ACC = "#20334d", "#5b6a7d", "#0e6b8a"


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


def save(fig, name):
    p = os.path.join(EV, name)
    fig.savefig(p, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[figure] {p}")


def eqbox(ax, x, y, w, h, col=ACC, alpha=0.07):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.014",
                                fc=col, ec=col, alpha=alpha, lw=0, zorder=0))


# ─────────────────────────────────────────── Sec. 2.2  proper scoring rule
fig, ax = canvas(12.2, 3.3)
ax.text(0.02, 0.88, "A scoring rule grades a predicted distribution against what happened:",
        fontsize=12, color=INK)
eqbox(ax, 0.05, 0.44, 0.90, 0.30)
ax.text(0.5, 0.59, r"$S(p_\theta,\,q)\;=\;\int q(y,x)\,S\!\left(p_\theta,(y,x)\right)dy\,dx$"
                   r"$\qquad\qquad$"
                   r"$S(p_\theta,\,q)\;\leq\;S(q,\,q)$",
        ha="center", va="center", fontsize=17, color=INK)
ax.text(0.30, 0.30, "expected score", ha="center", fontsize=10.5, color=MUTED)
ax.text(0.755, 0.30, "PROPER, with equality iff $p_\\theta = q$", ha="center",
        fontsize=10.5, color=ACC)
ax.text(0.02, 0.09, "so the network is trained by minimising", fontsize=12, color=INK)
ax.text(0.44, 0.09, r"$\mathcal{L}(\theta)\;=\;-\,S(p_\theta,\,q)$",
        fontsize=15, color=INK, va="center")
save(fig, "de_eq_score.png")


# ─────────────────────────────────────────── Eq. 1  regression criterion
fig, ax = canvas(12.2, 3.5)
eqbox(ax, 0.06, 0.50, 0.88, 0.36)
ax.text(0.5, 0.68,
        r"$-\log p_\theta(y_n\,|\,x_n)\;=\;"
        r"\frac{\log \sigma_\theta^{2}(x)}{2}\;+\;"
        r"\frac{\left(y-\mu_\theta(x)\right)^{2}}{2\,\sigma_\theta^{2}(x)}\;+\;"
        r"\mathrm{constant}$",
        ha="center", va="center", fontsize=19, color=INK)
ax.plot([0.365, 0.475], [0.455, 0.455], lw=1.3, color="#b5651d")
ax.plot([0.515, 0.700], [0.455, 0.455], lw=1.3, color="#4a7a4a")
ax.text(0.42, 0.36, "grows with $\\sigma$:\npunishes claiming ignorance", ha="center",
        fontsize=10, color="#b5651d", va="top", linespacing=1.5)
ax.text(0.607, 0.36, "shrinks with $\\sigma$:\npunishes confident errors", ha="center",
        fontsize=10, color="#4a7a4a", va="top", linespacing=1.5)
ax.text(0.5, 0.10,
        r"minimised at $\sigma_\theta^{2}(x) = (y-\mu_\theta(x))^{2}$ "
        "— the predicted variance is fitted to the actual squared error",
        ha="center", fontsize=11.5, color=ACC)
save(fig, "de_eq_nll.png")


# ─────────────────────────────────────────── Sec. 2.3  adversarial training
fig, ax = canvas(12.2, 3.0)
ax.text(0.02, 0.86, "Optional step 2 — perturb each input along the direction that "
                    "most increases the loss:", fontsize=12, color=INK)
eqbox(ax, 0.10, 0.44, 0.80, 0.28)
ax.text(0.5, 0.58,
        r"$x' \;=\; x \;+\; \epsilon\,\mathrm{sign}\!\left(\nabla_{x}\,"
        r"\ell(\theta,\,x,\,y)\right)$",
        ha="center", va="center", fontsize=18, color=INK)
ax.text(0.02, 0.27, "and train on both, so the objective for one network becomes",
        fontsize=12, color=INK)
ax.text(0.5, 0.10, r"$\ell(\theta,\,x,\,y)\;+\;\ell(\theta,\,x',\,y)$",
        ha="center", fontsize=16, color=INK)
ax.text(0.965, 0.10, r"$\epsilon = 1\%$ of the input range", ha="right", fontsize=10.5,
        color=MUTED)
save(fig, "de_eq_adv.png")


# ─────────────────────────────────────────── Algorithm 1
fig, ax = canvas(12.2, 4.4)
eqbox(ax, 0.02, 0.03, 0.96, 0.91, col=MUTED, alpha=0.05)
ax.text(0.04, 0.885, "Algorithm 1", fontsize=13, color=INK, fontweight="bold")
ax.text(0.175, 0.888, "(their notation; $M$ networks, trained independently)",
        fontsize=10.5, color=MUTED)
rows = [
    ("1:", r"let each network parametrise $p_\theta(y|x)$; use a proper scoring rule as "
           r"$\ell(\theta,x,y)$", "defaults $M=5$, $\\epsilon = 1\\%$ of the input range"),
    ("2:", r"initialise $\theta_1,\theta_2,\dots,\theta_M$ randomly", "the only source of diversity"),
    ("3:", r"for $m = 1 : M$  do", "independent — trivially parallel"),
    ("4:", r"$\quad$ sample a data point $n_m$ for each net", "minibatch in practice"),
    ("5:", r"$\quad$ $x'_{n_m} = x_{n_m} + \epsilon\,\mathrm{sign}"
           r"(\nabla_{x_{n_m}}\ell(\theta_m, x_{n_m}, y_{n_m}))$", "optional"),
    ("6:", r"$\quad$ minimise $\ell(\theta_m, x_{n_m}, y_{n_m}) + "
           r"\ell(\theta_m, x'_{n_m}, y_{n_m})$  w.r.t. $\theta_m$", "optional second term"),
]
y = 0.795
for tag, body, note in rows:
    ax.text(0.045, y, tag, fontsize=11.5, color=MUTED, family="monospace")
    ax.text(0.085, y, body, fontsize=12.5, color=INK, va="center")
    ax.text(0.965, y, note, fontsize=9.5, color=MUTED, ha="right", va="center", style="italic")
    y -= 0.125
ax.text(0.5, 0.045,
        "No shared state between members: no bagging, no boosting, no averaged weights — "
        "each network sees the whole dataset.",
        ha="center", fontsize=10.5, color=ACC)
save(fig, "de_alg1.png")


# ─────────────────────────────────────────── Sec. 2.4  prediction
fig, ax = canvas(12.2, 3.7)
ax.text(0.02, 0.90, "At test time the ensemble is a uniformly-weighted mixture:",
        fontsize=12, color=INK)
ax.text(0.5, 0.755, r"$p(y|x)\;=\;M^{-1}\sum_{m=1}^{M} p_{\theta_m}(y\,|\,x,\theta_m)$",
        ha="center", fontsize=16, color=INK)
ax.text(0.02, 0.585, "approximated by a single Gaussian with the mixture's first two moments:",
        fontsize=12, color=INK)
eqbox(ax, 0.05, 0.20, 0.90, 0.33)
ax.text(0.5, 0.365,
        r"$\mu_*(x)=M^{-1}\!\sum_m \mu_{\theta_m}(x)$"
        r"$\qquad$"
        r"$\sigma_*^{2}(x)=M^{-1}\!\sum_m\left(\sigma^{2}_{\theta_m}(x)+"
        r"\mu^{2}_{\theta_m}(x)\right)-\mu_*^{2}(x)$",
        ha="center", va="center", fontsize=16, color=INK)
# mathtext has no \underbrace; label the two halves with drawn rules instead
ax.text(0.5, 0.115,
        r"equivalently   $\sigma_*^{2}\;=\;M^{-1}\sum_m \sigma^{2}_{\theta_m}"
        r"\;+\;\mathrm{var}_m\!\left(\mu_{\theta_m}\right)$",
        ha="center", fontsize=15, color=INK)
ax.plot([0.398, 0.545], [0.075, 0.075], lw=1.3, color="#4a7a4a")
ax.plot([0.575, 0.712], [0.075, 0.075], lw=1.3, color="#b5651d")
ax.text(0.4715, 0.035, "aleatoric", ha="center", fontsize=10.5, color="#4a7a4a")
ax.text(0.6435, 0.035, "epistemic", ha="center", fontsize=10.5, color="#b5651d")
save(fig, "de_eq_combine.png")
