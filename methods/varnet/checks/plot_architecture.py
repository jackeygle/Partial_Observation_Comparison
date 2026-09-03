"""
plot_architecture.py  —  architecture diagrams for the model presentation

Draws three figures from the LIVE checkpoint, so a shape or a parameter count on a slide
can never drift from the model that actually produced the results:

  arch_overview.png   how the two components sit in one solver iteration
  arch_genn.png       inside the GENN prior: zero-centre psi, pointwise phi, two-scale Eq.10
  arch_solver.png     inside the solver: variational cost -> autograd -> ConvLSTM -> update

Everything is boxes-and-arrows on a plain axis rather than a graph library: the shapes are
the point, and they need to be annotated with the real tensor dimensions.
"""
from __future__ import annotations
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import torch                                                        # noqa: E402
from methods.varnet.checks.model_io import load_solver                                    # noqa: E402

OUT = os.path.join(ROOT, "check_outputs", "eval")
CK = os.path.join(ROOT, "runs", "varnet_a2_k1", "varnet_best.pt")
solver, A, _ = load_solver(CK, "cpu")
NP = lambda m: sum(p.numel() for p in m.parameters())
C, T, H, W = 4, A["dT"], 36, 12
HID, LH = A["hidden"], A["lstm_hidden"]

INK, MUTED = "#20334d", "#5b6a7d"
C_PRIOR, C_SOLVER, C_DATA, C_COST = "#0e6b8a", "#b5651d", "#4a7a4a", "#7a4a7a"


def box(ax, x, y, w, h, text, fc, fs=9, ec=None, lw=1.4, alpha=0.13):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec or fc, lw=lw, alpha=alpha, zorder=1))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc="none", ec=ec or fc, lw=lw, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, zorder=4, linespacing=1.45)


def arrow(ax, x1, y1, x2, y2, text="", fs=7.5, col=None, rad=0.0, dy=0.012):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 lw=1.3, color=col or MUTED,
                                 connectionstyle=f"arc3,rad={rad}", zorder=2))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, text, ha="center", va="bottom",
                fontsize=fs, color=col or MUTED, zorder=4)


def canvas(w=13, h=6.2):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return fig, ax


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[figure] {p}")


# ─────────────────────────────────────────────────── 1. overview
fig, ax = canvas(13, 6.0)
ax.text(0.5, 0.965, "One solver iteration — the two learned components",
        ha="center", fontsize=14, color=INK, weight="bold")
ax.text(0.5, 0.915, f"repeated {solver.n_iter} times; both components are trained jointly, "
        f"end to end (paper Eq. 14)", ha="center", fontsize=9.5, color=MUTED)

box(ax, 0.02, 0.60, 0.15, 0.19,
    f"state  $x^{{(k)}}$\n({C}, {T}, {H}, {W})\ndensity, vx, vy, var", C_DATA, 9)
box(ax, 0.02, 0.20, 0.15, 0.24,
    f"observations\n$y$, mask $\\Omega$\nsame shape\n\nrobots see ~47%\nof cells per frame", C_DATA, 8.5)

box(ax, 0.235, 0.44, 0.17, 0.22,
    f"prior  $\\Phi$  (GENN)\n$\\|x-\\Phi(x)\\|^2$\n\n{NP(solver.phi):,} params\n(0.7%)", C_PRIOR, 9)
box(ax, 0.235, 0.13, 0.17, 0.16, "observation term\n$\\|(x-y)\\odot\\Omega\\|^2$", C_DATA, 9)

box(ax, 0.455, 0.28, 0.15, 0.30,
    "variational cost\n$J(x)$\n\n"
    r"$\alpha_{obs}^2\,\|\cdot\|^2_{W_{obs}}$" "\n+\n" r"$\alpha_{reg}^2\,\|\cdot\|^2_{W_{reg}}$"
    f"\n\n{NP(solver.var_cost)} learned\nweights", C_COST, 8.5)

box(ax, 0.655, 0.34, 0.145, 0.20,
    "autograd\n$\\nabla_x J$\n\nno adjoint\nderived by hand", "#666", 9)
box(ax, 0.845, 0.30, 0.135, 0.28,
    f"ConvLSTM\nsolver\n\n$u=\\mathcal{{L}}(\\mathrm{{LSTM}}[\\nabla_x J])$\n\n"
    f"{NP(solver.grad_net):,}\nparams (99.3%)", C_SOLVER, 9)

arrow(ax, 0.17, 0.695, 0.235, 0.58, "")
arrow(ax, 0.17, 0.32, 0.235, 0.23, "")
arrow(ax, 0.405, 0.545, 0.455, 0.50, "")
arrow(ax, 0.405, 0.21, 0.455, 0.33, "")
arrow(ax, 0.605, 0.43, 0.655, 0.43, "")
arrow(ax, 0.80, 0.44, 0.845, 0.44, "")
# feedback: routed BELOW everything so it does not cross the boxes it connects
ax.add_patch(FancyArrowPatch((0.913, 0.30), (0.913, 0.10), arrowstyle="-", mutation_scale=1,
                             lw=1.6, color=C_SOLVER, zorder=2))
ax.add_patch(FancyArrowPatch((0.913, 0.10), (0.095, 0.10), arrowstyle="-", mutation_scale=1,
                             lw=1.6, color=C_SOLVER, zorder=2))
ax.add_patch(FancyArrowPatch((0.095, 0.10), (0.095, 0.60), arrowstyle="-|>", mutation_scale=14,
                             lw=1.6, color=C_SOLVER, zorder=2))
ax.text(0.505, 0.125, r"$x^{(k+1)} = x^{(k)} - u\,/\,n_{iter}$", ha="center", fontsize=11,
        color=C_SOLVER, weight="bold")

ax.text(0.5, 0.855, "the gradient fed to the LSTM is the gradient of the COST, not of the "
        "training loss — it never sees the ground truth (paper §3.4)",
        ha="center", fontsize=8.5, color=MUTED, style="italic")
save(fig, "arch_overview.png")


# ─────────────────────────────────────────────────── 2. GENN
# Taller canvas and fixed row bands: the two diagrams plus their captions need four
# non-overlapping horizontal bands, so the y positions are laid out once here rather than
# nudged per element.
fig, ax = canvas(13, 7.6)
ax.text(0.5, 0.972, f"Inside the prior  $\\Phi$  —  GENN, {NP(solver.phi):,} parameters",
        ha="center", fontsize=14, color=INK, weight="bold")
ax.text(0.5, 0.932, r"paper Eq. 10:   $\Phi(x) = Up\,(\,\Phi_1(Dw(x))\,) + \Phi_2(x)$",
        ha="center", fontsize=11, color=C_PRIOR)

# band 1 (y 0.72-0.90): one branch expanded
ax.text(0.03, 0.885, "One branch:   $\\Phi_b(x)=\\varphi(\\psi(x))$", fontsize=10.5,
        color=C_PRIOR, weight="bold")
box(ax, 0.03, 0.725, 0.115, 0.125, f"$x$\n({C},{T},{H},{W})", C_DATA, 9)
box(ax, 0.185, 0.715, 0.17, 0.145,
    f"$\\psi$ : Conv3d {A.get('kt',3)}x{A.get('kh',3)}x{A.get('kw',3)}\nCENTRE TAP = 0\n"
    f"$\\to$ ({HID},{T},{H},{W})", C_PRIOR, 8.6)
box(ax, 0.395, 0.715, 0.115, 0.145, "ReLU", "#888", 9)
box(ax, 0.55, 0.705, 0.20, 0.165,
    f"$\\varphi$ : pointwise 1x1x1\nConv3d + ReLU + Conv3d\n({HID}) $\\to$ ({HID}) $\\to$ ({C})\n"
    f"$\\to$ ({C},{T},{H},{W})", C_PRIOR, 8.4)
box(ax, 0.79, 0.725, 0.115, 0.125, f"$\\Phi_b(x)$", C_PRIOR, 9.5)
for x1, x2 in ((0.145, 0.185), (0.355, 0.395), (0.51, 0.55), (0.75, 0.79)):
    arrow(ax, x1, 0.788, x2, 0.788)

# band 2 (y 0.55-0.70): caption for band 1
ax.text(0.03, 0.685,
        "$\\psi$'s zero centre tap is the whole point: the output at a space-time point never reads the input at that same\n"
        "point, so $\\|x-\\Phi(x)\\|^2$ cannot be driven to zero by learning $\\Phi=$ identity — a prior that says nothing.\n"
        "$\\varphi$ then stays pointwise (every kernel is 1, as Sec. 3.2 requires) so it cannot smuggle that dependence\n"
        "back in. All of the prior's spatial and temporal context therefore comes from the single $\\psi$ layer.",
        fontsize=8.6, color=MUTED, va="top", linespacing=1.75)

# band 3 (y 0.14-0.46): two scales
ax.text(0.03, 0.455, "Two scales (Eq. 10)", fontsize=10.5, color=C_PRIOR, weight="bold")
box(ax, 0.03, 0.245, 0.10, 0.115, "$x$", C_DATA, 9.5)
box(ax, 0.185, 0.315, 0.155, 0.095, "$Dw$ : avg-pool /2", C_PRIOR, 8.6)
box(ax, 0.395, 0.290, 0.165, 0.145,
    f"$\\Phi_1$ on {H//2}x{W//2}\n3x3 here spans\n6x6 fine cells", C_PRIOR, 8.6)
box(ax, 0.615, 0.315, 0.155, 0.095, "$Up$ : ConvTranspose\n(learned)", C_PRIOR, 8.2)
box(ax, 0.395, 0.140, 0.165, 0.105, f"$\\Phi_2$ on {H}x{W}\nsees $x$ itself", C_PRIOR, 8.6)
box(ax, 0.825, 0.235, 0.115, 0.130, "$+$\n$\\Phi(x)$", C_PRIOR, 10)
arrow(ax, 0.13, 0.330, 0.185, 0.360)
arrow(ax, 0.13, 0.275, 0.395, 0.195, rad=-0.05)
arrow(ax, 0.34, 0.362, 0.395, 0.362)
arrow(ax, 0.56, 0.362, 0.615, 0.362)
arrow(ax, 0.77, 0.355, 0.825, 0.315, rad=0.06)
arrow(ax, 0.56, 0.190, 0.825, 0.270, rad=-0.05)

# band 4 (y < 0.11): caption for band 3
ax.text(0.03, 0.095,
        f"$\\Phi_1$ runs AT the coarse resolution — that is what makes the two branches see different scales; on the\n"
        f"half-size grid the same 3x3 kernel covers 6x6 fine cells. $\\Phi_2$ sees $x$ itself, not a high-pass residual.\n"
        f"Each branch is {NP(solver.phi.branch_fine):,} parameters, $Up$ adds {NP(solver.phi.up)}.",
        fontsize=8.6, color=MUTED, va="top", linespacing=1.75)
save(fig, "arch_genn.png")


# ─────────────────────────────────────────────────── 3. solver
fig, ax = canvas(13, 6.4)
ax.text(0.5, 0.965, f"Inside the solver  —  ConvLSTM, {NP(solver.grad_net):,} parameters",
        ha="center", fontsize=14, color=INK, weight="bold")
ax.text(0.5, 0.918, r"paper Eq. 11:   $g^{(k+1)}=\mathrm{LSTM}[\alpha\nabla_x U_\Phi,\,h,\,c]$,"
        r"   $x^{(k+1)}=x^{(k)}-\mathcal{L}(g^{(k+1)})$",
        ha="center", fontsize=10.5, color=C_SOLVER)

box(ax, 0.03, 0.60, 0.145, 0.17, f"$\\nabla_x J$\n({C},{T},{H},{W})", C_COST, 9)
box(ax, 0.215, 0.585, 0.165, 0.20,
    f"fold time into\nchannels\n({C}x{T}={C*T}, {H}, {W})", "#888", 8.5)
box(ax, 0.425, 0.555, 0.20, 0.26,
    f"ConvLSTM cell\nConv2d({C*T}+{LH} $\\to$ 4x{LH}, 3x3)\ngates i, f, o, g\n\n"
    f"{NP(solver.grad_net.lstm.gates):,} params", C_SOLVER, 8.5)
box(ax, 0.675, 0.585, 0.145, 0.20, f"1x1 conv\n{LH} $\\to$ {C*T}\n\n{NP(solver.grad_net.out):,}",
    C_SOLVER, 8.5)
box(ax, 0.865, 0.60, 0.115, 0.17, f"update $u$\n({C},{T},{H},{W})", C_SOLVER, 9)
for x1, x2 in ((0.175, 0.215), (0.38, 0.425), (0.625, 0.675), (0.82, 0.865)):
    arrow(ax, x1, 0.685, x2, 0.685)
ax.add_patch(FancyArrowPatch((0.525, 0.545), (0.525, 0.505), arrowstyle="-|>", mutation_scale=12,
                             lw=1.3, color=C_SOLVER, zorder=2))
ax.text(0.535, 0.512, "hidden state $(h,c)$ carried across all "
        f"{solver.n_iter} iterations", fontsize=8.4, color=C_SOLVER, va="center")

ax.text(0.03, 0.435, "Why this component holds 99.3% of the parameters", fontsize=10.5,
        color=C_SOLVER, weight="bold")
ax.text(0.03, 0.385,
        f"The state is a whole {T}-frame window, and time is folded into the convolution's channel axis, so the\n"
        f"cell's input is C x dT = {C} x {T} = {C*T} channels over a {H}x{W} grid:\n\n"
        f"      gates = 4 x (({C*T}+{LH}) x {LH} x 3 x 3 + {LH}) = {NP(solver.grad_net.lstm.gates):,}\n\n"
        f"The paper's Lorenz-96 solver is ~1,000 parameters because its state is 40 numbers, not "
        f"{C}x{T}x{H}x{W} = {C*T*H*W:,}.\nA literal match is not meaningful; what matters is whether the size is "
        f"justified. Measured: doubling it made\nresults worse, and cutting it to 24% (with a 3.5x larger prior) "
        f"made blind-zone error 45% worse.",
        fontsize=8.6, color=MUTED, va="top", linespacing=1.6)

ax.text(0.03, 0.055,
        "Two implementation details that are NOT in the paper, inherited from the authors' own SSH code: the "
        "gradient is\nrescaled by its RMS from the first iteration (raw gradients span orders of magnitude across "
        "iterations and\ntraining diverges without it), and the cost carries per-channel weights $W_{obs}, W_{reg}$ "
        "where Eq. 4 has two scalars.",
        fontsize=8.2, color=MUTED, va="top", linespacing=1.6, style="italic")
save(fig, "arch_solver.png")

print(f"\n  params: GENN {NP(solver.phi):,} ({NP(solver.phi)/NP(solver)*100:.1f}%)  "
      f"ConvLSTM {NP(solver.grad_net):,} ({NP(solver.grad_net)/NP(solver)*100:.1f}%)  "
      f"cost {NP(solver.var_cost)}  total {NP(solver):,}")
