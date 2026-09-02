"""
plot_frameworks.py  —  one diagram per method, for the three-way framework comparison

  fw_enkf.png    the filter cycle, every stage opened up: forecast, noise, bias, clip,
                 gain from the ensemble covariance, localization, per-member perturbed
                 observations, inflation
  fw_4dvar.png   the variational solver with ONE descent step expanded into its six
                 operations, and the 20-step loop drawn around it
  fw_prior.png   inside the prior Phi: Eq.10's two branches, and inside a branch the
                 zero-centre psi followed by the pointwise phi
  fw_unc.png     the variance head at TRAINING time: its 12 input channels, its two 1x1
                 convs, the sigma range it produces, the beta-NLL and why beta=0 failed
  fw_unc_ens.png the same method at INFERENCE time: the five members, what differs between
                 them, and which half of sigma^2 each term of the moment matching supplies

The four are drawn with the same box style and the same colour per method, so they can be
put on consecutive slides and read as one story: fw_enkf and fw_4dvar are the two methods
at the same level of detail, fw_prior opens the one box in fw_4dvar that does the physics,
and fw_unc is the delta of the third method against the second.

Every parameter count and every shape is read from the checkpoints and from the EnKF
baseline's own configuration, so a diagram cannot drift from the model it describes.
"""
from __future__ import annotations
import os
import sys

import json as _json_

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = os.path.join(ROOT, "check_outputs", "eval")
INK, MUTED = "#20334d", "#5b6a7d"
C_EK, C_VN, C_OBS, C_UN = "#b5651d", "#0e6b8a", "#4a7a4a", "#7a4a7a"

sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "checks"))
from model_io import load_solver  # noqa: E402

# ─────────────────────────── sizes, read from the artefacts ───────────────────────────
sys.path.insert(0, os.path.join(ROOT, "enkf_lab"))
from pedpred.utils import load_model  # noqa: E402
N_SURR = sum(p.numel() for p in
             load_model(os.path.join(ROOT, "enkf_lab", "apt-ibex_train_model_28D.pth"),
                        torch.device("cpu")).parameters())
_S, _A, _ = load_solver(os.path.join(ROOT, "runs/varnet_b0_k1/varnet_best.pt"), "cpu")
N_PHI = sum(p.numel() for p in _S.phi.parameters())
N_SOLV = sum(p.numel() for p in _S.grad_net.parameters())
N_TOT = sum(p.numel() for p in _S.parameters())
N_IT, DT, HID = _S.n_iter, _A["dT"], _S.grad_net.lstm.hidden_ch
# the prior, branch by branch, so fw_prior's labels add up to N_PHI on the slide
N_BR = sum(p.numel() for p in _S.phi.branch_fine.parameters())
N_PSI = sum(p.numel() for p in _S.phi.branch_fine.psi.parameters())
N_UP = sum(p.numel() for p in _S.phi.up.parameters())
SCALE = _S.phi.scale
# grid, from the EnKF baseline's own constants
H, W, C = 36, 12, 4
NENS, RAD, INFL = 100, 7, 1.02
SDIM = C * H * W

# the variance head, instantiated exactly as train_varnet.py builds it
sys.path.insert(0, ROOT)
# the variance read-out's size, from the solver itself: a second 1x1 conv on the hidden state
N_VAR = (_S.grad_net.lstm.hidden_ch + 1) * (C * DT)
# channel names, and the sigma the head actually produces across the field (from
# checks/diag_varhead_trace.py). There is no longer a per-channel floor: sigma^2 = softplus(head)
import config  # noqa: E402
CHAN = config.CFG["grid"]["channels"]
_VHT = os.path.join(EV, "varhead_trace.json")
SIG_RANGE = (_json_.load(open(_VHT))["sigma_range"]
             if os.path.exists(_VHT) else None)
_SP = float(torch.nn.functional.softplus(torch.tensor(-3.0)))


# ───────────────────────────────── drawing helpers ─────────────────────────────────
def canvas(w=14.2, h=5.9):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


def frame(ax, x, y, w, h, col, alpha=0.09, lw=1.5, ls="-", z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010",
                                fc=col, ec=col, alpha=alpha, lw=lw, zorder=z))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010",
                                fc="none", ec=col, lw=lw, ls=ls, zorder=z + 2))


def stage(ax, x, y, w, h, title, lines, col, tfs=10.2, lfs=8.4, alpha=0.09):
    """A box with a bold heading and a left-aligned list of internal operations."""
    frame(ax, x, y, w, h, col, alpha)
    ax.text(x + w / 2, y + h - 0.052, title, ha="center", va="center", fontsize=tfs,
            color=col, fontweight="bold", zorder=6)
    ax.plot([x + 0.012, x + w - 0.012], [y + h - 0.098] * 2, color=col, lw=0.8,
            alpha=0.45, zorder=6)
    # Fit the lines to the box rather than to a fixed step: with a fixed step a long list
    # silently runs out through the bottom edge and lands on whatever is drawn below.
    top = y + h - 0.118
    step = min(0.082, (top - y - 0.020) / max(len(lines), 1))
    ty = top - step / 2
    for ln in lines:
        ax.text(x + 0.016, ty, ln, ha="left", va="center", fontsize=lfs, color=INK,
                zorder=6)
        ty -= step


def plain(ax, x, y, w, h, text, col, fs=9.6, alpha=0.13):
    frame(ax, x, y, w, h, col, alpha)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, zorder=6, linespacing=1.55)


def arrow(ax, x1, y1, x2, y2, col=MUTED, ls="-", rad=0.0, lw=1.4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 lw=lw, color=col, zorder=8, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def save(fig, name):
    p = os.path.join(EV, name)
    fig.savefig(p, dpi=185, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[figure] {p}")


# ═══════════════════════════════════════════════════════ 1. the filter, opened up
fig, ax = canvas()
ax.text(0.5, 0.972, "EnKF  —  one frame at a time, and the estimate for frame $t$ uses "
                    "observations up to $t$ only",
        ha="center", fontsize=12.5, color=C_EK, style="italic")

# the ensemble as a stack, so "100 copies of the whole state" is visible
for i in range(5):
    o = i * 0.0085
    frame(ax, 0.020 + o, 0.470 - o, 0.130, 0.235, C_EK, 0.08 if i < 4 else 0.16)
ax.text(0.020 + 0.130 / 2 + 0.034, 0.470 - 0.034 + 0.235 / 2,
        f"{NENS} members\n\n$X_t \\in \\mathbb{{R}}^{{{NENS} \\times {SDIM}}}$\n"
        f"{C} ch $\\times$ {H}$\\times${W}",
        ha="center", va="center", fontsize=9.4, color=INK, zorder=7, linespacing=1.6)

stage(ax, 0.215, 0.395, 0.245, 0.480, "FORECAST", [
    f"$X_f = f(X_t)$   —   {NENS} forward passes",
    f"$f$ = PedPred3 surrogate, {N_SURR:,} params",
    "$+\\;\\mathcal{N}(0,\\;(0.01\\,\\sigma_{proc})^2)$",
    "$-\\;$ bias estimate  (EMA over past frames)",
    "clip to physical bounds",
], C_EK)

stage(ax, 0.500, 0.300, 0.300, 0.575, "ANALYSIS", [
    f"$Y_f = C X_f$   (3 robots, radius 2 cells)",
    "$P_{xy} = X'^{T}Y'/(N\\!-\\!1)$",
    "$P_{yy} = Y'^{T}Y'/(N\\!-\\!1) + R$",
    "$K = (P_{xy}P_{yy}^{+}) \\odot \\mathrm{Loc}^{T}$",
    f"$\\mathrm{{Loc}} = e^{{-d^2/2r^2}}$,  cut at $d \\leq {RAD}$",
    "per member $i$:",
    "   $X_a^i = X_f^i + K(y_t + \\varepsilon_i - Y_f^i)$",
    f"inflate {INFL} about the mean, clip",
], C_EK, lfs=8.2)

plain(ax, 0.500, 0.130, 0.300, 0.115,
      f"observations $y_t$, $\\Omega_t$   —   {C} channels, only the visited cells", C_OBS, 8.8)

plain(ax, 0.845, 0.615, 0.140, 0.115, "mean\n$\\rightarrow$ estimate $\\hat{x}_t$", C_EK, 9.4)
plain(ax, 0.845, 0.445, 0.140, 0.115, "std\n$\\rightarrow$ uncertainty $\\sigma_t$", C_EK, 9.4)

arrow(ax, 0.186, 0.605, 0.212, 0.605)
arrow(ax, 0.462, 0.605, 0.497, 0.605)
arrow(ax, 0.650, 0.248, 0.650, 0.297, col=C_OBS)
arrow(ax, 0.802, 0.660, 0.842, 0.672)
arrow(ax, 0.802, 0.600, 0.842, 0.502)
ax.text(0.478, 0.632, "$X_f$", fontsize=9.4, color=MUTED, ha="center")

# the recursion, routed below everything so it crosses nothing
_yr = 0.048
ax.plot([0.915, 0.915], [0.440, _yr], color=C_EK, lw=1.4, ls=":", zorder=8)
ax.plot([0.915, 0.087], [_yr, _yr], color=C_EK, lw=1.4, ls=":", zorder=8)
arrow(ax, 0.087, _yr, 0.087, 0.432, col=C_EK, ls=":")
ax.text(0.47, _yr, "the analysed ensemble becomes the next frame's prior   "
                   "$t \\rightarrow t+1$",
        ha="center", va="center", fontsize=9.6, color=C_EK, zorder=9,
        bbox=dict(fc="white", ec="none", pad=3))

ax.text(0.5, 0.918, "nothing here is fitted to our data: the surrogate is given pre-trained, "
                    "and the filter itself has no free parameters",
        ha="center", fontsize=9.8, color=MUTED)
save(fig, "fw_enkf.png")


# ═══════════════════════════════════════════════════════ 2. the solver, one step opened up
fig, ax = canvas(14.2, 6.5)
ax.text(0.5, 0.975, f"4DVarNet end to end  —  a whole {DT}-frame window at once, and where "
                    f"the uncertainty branch attaches",
        ha="center", fontsize=12.5, color=C_VN, style="italic")

plain(ax, 0.018, 0.371, 0.150, 0.190,
      f"$x_0$, $y$, $\\Omega$\n\n$(B,{C},{DT},{H},{W})$\nall {DT} frames", C_OBS, 9.4)

# the loop, drawn as a container so "20 times" applies to the whole body
frame(ax, 0.285, 0.058, 0.450, 0.832, C_VN, 0.035, lw=1.6, ls="--", z=1)
ax.text(0.510, 0.856, f"repeat {N_IT} times   —   the LSTM state $(h,c)$ is carried across "
                      f"steps",
        ha="center", fontsize=10.2, color=C_VN, fontweight="bold", zorder=6)

# (operation, right-aligned side note). The notes live INSIDE the step boxes: the container
# is flanked by the input and output boxes, so anything placed beside it collides.
STEPS = [
    ("$dy = (x-y)\\odot\\Omega$        $dx = x - \\Phi(x)$",
     "the observation term and the prior term, per point"),
    ("$J = \\alpha_{obs}^2\\|dy\\|^2_{W_{obs}} + \\alpha_{reg}^2\\|dx\\|^2_{W_{reg}}$",
     f"$\\alpha_{{obs}},\\alpha_{{reg}}$ scalars,  "
     f"$W_{{obs}},W_{{reg}}\\in\\mathbb{{R}}^{{{C}}}$ — all learnable"),
    ("$g = \\partial J/\\partial x$   by autograd",
     "the gradient of the COST — no truth enters here,\nand no adjoint of $\\Phi$ is derived by hand"),
    (f"$\\hat{{g}} = g\\,/\\,\\mathrm{{RMS}}(g)$,  reshape to $(B,{C * DT},{H},{W})$",
     f"{C}$\\times${DT} folded onto the channel axis"),
    (f"ConvLSTM $({C * DT}\\rightarrow{HID}$, $3\\times3) \\rightarrow 1\\times1$ conv "
     f"$\\rightarrow u$", f"{N_SOLV:,} params — 97% of the whole model"),
    (f"$x \\leftarrow x - u\\,/\\,{N_IT}$", "one learned descent step"),
]
_y = 0.717
for i, (txt, note) in enumerate(STEPS):
    frame(ax, 0.303, _y, 0.414, 0.098, C_VN, 0.10)
    ax.text(0.319, _y + 0.049, f"{i + 1}", fontsize=9, color=C_VN, fontweight="bold",
            va="center", ha="center", zorder=6)
    ax.text(0.340, _y + 0.065, txt, fontsize=9.4, color=INK, va="center", ha="left",
            zorder=6)
    if note:
        ax.text(0.340, _y + 0.028, note, fontsize=7.6, color=MUTED, va="center", ha="left",
                zorder=6, linespacing=1.45)
    if i:
        arrow(ax, 0.510, _y + 0.120, 0.510, _y + 0.102, lw=1.1)
    _y -= 0.120

# Phi hangs off the left of step 1, which is the only step that calls it
plain(ax, 0.100, 0.711, 0.155, 0.110, f"prior $\\Phi$  —  GENN\n{N_PHI:,} params", C_VN, 9.2)
arrow(ax, 0.257, 0.766, 0.300, 0.766, col=C_VN)
ax.text(0.1775, 0.687, "the next figure opens this box", fontsize=8.4, color=MUTED,
        ha="center", va="top", style="italic")

plain(ax, 0.775, 0.560, 0.150, 0.170,
      f"$\\hat{{x}}$\n\n$(B,{C},{DT},{H},{W})$", C_VN, 9.4)
arrow(ax, 0.170, 0.466, 0.283, 0.466)
arrow(ax, 0.737, 0.560, 0.773, 0.620)

# Method 3's whole branch, hung off x_hat so the end-to-end path is on one figure. Drawn in
# its own colour inside a dashed container: everything above is method 2 and is untouched.
frame(ax, 0.762, 0.058, 0.222, 0.462, C_UN, 0.05, lw=1.4, ls="--", z=1)
ax.text(0.900, 0.492, "method 3 adds only this", fontsize=9.2, color=C_UN,
        fontweight="bold", ha="center", zorder=6)
for _t, _y in ((f"variance head\n{N_VAR} params", 0.362),
               (f"$\\sigma^2$   $(B,{C},{DT},{H},{W})$", 0.244),
               (f"$\\times5$ members\n$\\sigma^2_* = \\overline{{\\sigma^2_m}} + "
                f"\\mathrm{{Var}}_m(\\hat{{x}}_m)$", 0.096)):
    plain(ax, 0.775, _y, 0.196, 0.106 if _y != 0.244 else 0.078, _t, C_UN, 8.8)
arrow(ax, 0.795, 0.556, 0.795, 0.472, col=C_UN)
arrow(ax, 0.873, 0.358, 0.873, 0.326, col=C_UN)
arrow(ax, 0.873, 0.240, 0.873, 0.206, col=C_UN)
ax.text(0.873, 0.040, "trained on the NLL instead of Eq. 14 —\n"
                      "the cost, the prior and the solver do not change",
        fontsize=8.2, color=MUTED, ha="center", va="top", linespacing=1.7)

ax.text(0.5, 0.928, f"the prior and the solver are trained together, end to end, on 32 days "
                    f"—  {N_TOT:,} parameters in total",
        ha="center", fontsize=9.8, color=MUTED)
save(fig, "fw_4dvar.png")


# ═══════════════════════════════════════════════════════ 3. inside the prior
fig, ax = canvas(14.2, 5.6)
ax.text(0.5, 0.968, f"Inside the prior $\\Phi$  —  {N_PHI:,} parameters, 0.5% of the model, "
                    f"and the only part that encodes the dynamics",
        ha="center", fontsize=12.5, color=C_VN, style="italic")

ax.text(0.5, 0.888, "$\\Phi(x) = \\mathrm{Up}(\\Phi_1(\\mathrm{Dw}(x))) + \\Phi_2(x)$"
                    "        (paper Eq. 10)",
        ha="center", fontsize=13, color=INK)

plain(ax, 0.020, 0.430, 0.115, 0.150, f"$x$\n$(B,{C},{DT},{H},{W})$", C_VN, 9.2)

# coarse branch
plain(ax, 0.185, 0.610, 0.135, 0.135,
      f"Dw\navg-pool ${SCALE}\\times{SCALE}$\n${H}\\times{W} \\rightarrow "
      f"{-(-H // SCALE)}\\times{-(-W // SCALE)}$", C_VN, 8.8)
plain(ax, 0.360, 0.610, 0.150, 0.135,
      f"$\\Phi_1$  coarse branch\n{N_BR:,} params\n$3\\times3$ now spans "
      f"${SCALE * 3}\\times{SCALE * 3}$ cells", C_VN, 8.8)
plain(ax, 0.550, 0.610, 0.145, 0.135,
      f"Up\nConvTranspose2d\nlearned, {N_UP} params", C_VN, 8.8)
# fine branch
plain(ax, 0.360, 0.265, 0.150, 0.135,
      f"$\\Phi_2$  fine branch\n{N_BR:,} params\nsees $x$ at full resolution", C_VN, 8.8)

plain(ax, 0.745, 0.430, 0.105, 0.150, "$+$", C_VN, 15)
plain(ax, 0.890, 0.430, 0.100, 0.150, f"$\\Phi(x)$\nsame shape", C_VN, 9.2)

arrow(ax, 0.137, 0.540, 0.183, 0.660)
arrow(ax, 0.137, 0.470, 0.358, 0.335)
arrow(ax, 0.322, 0.677, 0.358, 0.677)
arrow(ax, 0.512, 0.677, 0.548, 0.677)
arrow(ax, 0.697, 0.660, 0.743, 0.545)
arrow(ax, 0.512, 0.335, 0.743, 0.465)
arrow(ax, 0.852, 0.505, 0.888, 0.505)

# what is inside one branch
frame(ax, 0.185, 0.055, 0.510, 0.160, C_VN, 0.055, lw=1.3, ls="--")
ax.text(0.200, 0.176, "inside either branch", fontsize=9.4, color=C_VN,
        fontweight="bold", ha="left", zorder=6)
_bx = [
    (f"$\\psi$: Conv3d $({C}\\rightarrow32)$\n$3\\times3\\times3$, centre VOXEL zeroed\n"
     f"{N_PSI:,} params", 0.205, 0.140),
    ("ReLU", 0.380, 0.055),
    ("$\\varphi$: Conv3d $32\\rightarrow32$, $1\\times1\\times1$\nReLU\n"
     f"Conv3d $32\\rightarrow{C}$, $1\\times1\\times1$", 0.465, 0.185),
]
for txt, x, w in _bx:
    plain(ax, x, 0.070, w, 0.095, txt, C_VN, 7.9, alpha=0.10)
arrow(ax, 0.352, 0.118, 0.377, 0.118, lw=1.1)
arrow(ax, 0.437, 0.118, 0.462, 0.118, lw=1.1)

ax.text(0.715, 0.108, "the zeroed centre tap is the point — and it zeroes exactly\n"
                      "ONE voxel: $\\Phi(x)(t,h,w)$ never reads $x(t,h,w)$ itself, but it\n"
                      "does read its space-time neighbours, the same frame's\n"
                      "$x(t,h\\pm1,w\\pm1)$ included — 26 of the 27 taps are live.\n"
                      "That is enough: $\\Phi$ cannot be the identity, so $x - \\Phi(x)$\n"
                      "is a genuine prediction error (paper Sec. 3.2: $\\psi(x)(s)\\perp x(s)$,\n"
                      "where $s$ is a space-time point, not a frame)\n"
                      "\n"
                      "$\\varphi$'s kernel is 1 — POINTWISE: it mixes the 32 feature\n"
                      "channels at each $(t,h,w)$ separately, adding depth\n"
                      "without touching time or space a second time",
        fontsize=8.6, color=MUTED, va="center", ha="left", linespacing=1.7)
ax.text(0.7975, 0.395, f"{N_BR:,} $+$ {N_BR:,} $+$ {N_UP} $=$ {N_PHI:,}",
        fontsize=8.6, color=MUTED, ha="center", va="top")
save(fig, "fw_prior.png")


# ═══════════════════════════════════════════════════════ 4a. the head, at training time
# Three blocks. Dropped: the per-channel sigma percentile table, duplicated verbatim on the
# appendix page about what the head produces; the beta=0 failure box, which defends a choice
# rather than describing the method; and the head's initialisation rationale. All three moved
# to the slide notes.
fig, ax = canvas(14.2, 5.4)
ax.text(0.5, 0.958, f"Method 3, training time  —  the cost $J$, the prior and the solver are "
                    f"untouched; the head's {N_VAR} parameters and the loss are the whole "
                    f"change",
        ha="center", fontsize=12.2, color=C_UN, style="italic")
ax.text(0.5, 0.902, f"$\sigma^2$ stays OUTSIDE $J(x)$ on purpose: it is not part of the "
                    f"physical state being assimilated, so the gradient handed to the solver "
                    f"is still the gradient of the cost (4DVarNet Sec. 3.4)",
        ha="center", fontsize=9.2, color=INK)

# ── what does NOT change
frame(ax, 0.020, 0.706, 0.960, 0.150, C_VN, 0.055, lw=1.3, ls="--")
ax.text(0.034, 0.826, "unchanged from method 2", fontsize=8.8, color=C_VN,
        fontweight="bold", ha="left", zorder=6)
for _t, _x, _w in ((r"$x_0$, $y$, $\Omega$", 0.040, 0.115),
                   (f"solver, {N_IT} steps on $J(x)$" + "\n" + f"{N_TOT:,} params", 0.200,
                    0.190),
                   (r"$\hat{x}$   " + f"$(B,{C},{DT},{H},{W})$", 0.435, 0.155)):
    plain(ax, _x, 0.724, _w, 0.078, _t, C_VN, 8.6, alpha=0.12)
arrow(ax, 0.157, 0.763, 0.198, 0.763, lw=1.2)
arrow(ax, 0.392, 0.763, 0.433, 0.763, lw=1.2)
ax.text(0.618, 0.763, "not one line of $J$, $\Phi$ or the ConvLSTM changes — so the\n"
                      "reconstruction stays comparable with method 2",
        fontsize=8.6, color=MUTED, va="center", ha="left", linespacing=1.6)

# ── the head, in one block
stage(ax, 0.020, 0.322, 0.420, 0.362, f"the head  —  {N_VAR} parameters", [
    r"input   $\hat{x}$ · $\Omega$ · $|\hat{x}-\Phi(\hat{x})|$,  "
    f"{3 * C} channels, all POINTWISE",
    "",
    f"        Conv3d $({3 * C}" + r"\rightarrow" + "32)$, $1\\times1\\times1$"
    f"          {3 * C * 32 + 32}",
    "        ReLU",
    f"        Conv3d $(32" + r"\rightarrow" + f"{C})$,  $1\\times1\\times1$"
    f"            {32 * C + C}",
    "",
    r"output  $\sigma^2 = \mathrm{softplus}(\cdot)$,  one per cell per channel",
], C_UN, tfs=10, lfs=9.0)
ax.text(0.230, 0.300, "the mask lets it treat blind cells differently; the residual says where "
                      "the\nreconstruction breaks the learnt dynamics, and costs one extra "
                      "$\Phi$ pass ($\\approx$0.5%)",
        ha="center", va="top", fontsize=8.4, color=MUTED, linespacing=1.7)

# ── the loss. The paper's criterion is the headline, in the paper's own notation, because
# that is what a reader comparing against Sec. 2.2.1 needs to see. Our one deviation is a
# single small line under it -- shrinking it to a footnote is fine, dropping it would make the
# slide claim something we do not do.
frame(ax, 0.470, 0.322, 0.510, 0.362, C_UN, 0.09)
ax.text(0.725, 0.646, "the training loss:   MSE (Eq. 14)   $\\rightarrow$   the paper's "
                      "negative log-likelihood",
        ha="center", va="center", fontsize=10, color=C_UN, fontweight="bold", zorder=6)
ax.text(0.725, 0.556, r"$-\log p_\theta(y_n|x_n) \;=\; "
                      r"\frac{\log \sigma^2_\theta(x)}{2} \;+\; "
                      r"\frac{(y - \mu_\theta(x))^2}{2\sigma^2_\theta(x)} \;+\; "
                      r"\mathrm{const}$",
        ha="center", va="center", fontsize=14, color=INK, zorder=6)
ax.text(0.630, 0.470, "punishes claiming", fontsize=7.8, color=MUTED, ha="center",
        va="center", zorder=6)
ax.text(0.630, 0.448, "TOO MUCH uncertainty", fontsize=7.8, color=MUTED, ha="center",
        va="center", zorder=6)
ax.text(0.845, 0.470, "punishes claiming", fontsize=7.8, color=MUTED, ha="center",
        va="center", zorder=6)
ax.text(0.845, 0.448, "TOO LITTLE", fontsize=7.8, color=MUTED, ha="center", va="center",
        zorder=6)
ax.text(0.725, 0.404, r"$\mu_\theta$ is the reconstruction $\hat{x}$, $y$ is the truth, and "
                      r"$\sigma^2_\theta$ is the head's output",
        fontsize=8.4, color=MUTED, ha="center", va="center", zorder=6)
ax.plot([0.488, 0.962], [0.388] * 2, color=C_UN, lw=0.7, alpha=0.4, zorder=6)
ax.text(0.725, 0.354, r"one deviation: each point is multiplied by its own $\sigma^2$ "
                      r"(detached), $\beta\!=\!1$." + "\n"
                      r"Plain Eq. 1 drove $\sigma^2$ to zero here and cost 52% RMSE.",
        fontsize=8.2, color=C_UN, ha="center", va="center", zorder=6, linespacing=1.7)

ax.text(0.5, 0.180, f"Two changes, and nothing else: {N_VAR} parameters after the solve, and a "
                    f"different training loss.\nThe reconstruction the solver produces is the "
                    f"same object it was in method 2.",
        ha="center", va="center", fontsize=9.8, color=INK, linespacing=1.9,
        bbox=dict(fc=C_UN, ec=C_UN, alpha=0.07, pad=6))

save(fig, "fw_unc.png")


# ═══════════════════════════════════════════════════════ 4b. the ensemble, at inference time
fig, ax = canvas(14.2, 5.7)
ax.text(0.5, 0.968, "Method 3, inference time  —  five members, and where each half of "
                    "$\\sigma^2_*$ comes from",
        ha="center", fontsize=12.2, color=C_UN, style="italic")

ax.text(0.5, 0.906, "the five differ in ONE thing: the weight initialisation. Same 32 days, "
                    "same robot routes, same noise realisation, same window order "
                    "(paper Sec. 2.4 — bagging is reported to hurt)",
        ha="center", fontsize=9.2, color=MUTED)

# the five members
_MY = [0.742, 0.610, 0.478, 0.346, 0.214]
for _m, _y in enumerate(_MY):
    plain(ax, 0.030, _y, 0.250, 0.095,
          f"member {_m},  init seed {_m}      $\\rightarrow$      "
          f"$\\hat{{x}}_{_m}$,  $\\sigma^2_{_m}$", C_UN, 8.8, alpha=0.10)
    arrow(ax, 0.284, _y + 0.047, 0.332, 0.480, col=C_UN, lw=1.1)
ax.text(0.155, 0.178, f"each member is the whole pipeline — {N_TOT:,} $+$ {N_VAR} params,\n"
                      f"trained for 150 epochs on its own",
        fontsize=8.4, color=MUTED, ha="center", va="top", style="italic", linespacing=1.6)

# the combination
stage(ax, 0.340, 0.330, 0.300, 0.300, "moment matching  (paper Sec. 2.4)", [
    "$\\hat{x}_* = \\frac{1}{M}\\sum_m \\hat{x}_m$",
    "",
    "$\\sigma^2_* = \\overline{\\sigma^2_m} \; + \; "
    "\\mathrm{Var}_m(\\hat{x}_m)$",
], C_UN, tfs=9.6, lfs=10.5)
ax.text(0.418, 0.312, "ALEATORIC\nwhat each net says it\ncannot account for;\n"
                      "does not shrink with $M$",
        fontsize=8.0, color=MUTED, ha="center", va="top", linespacing=1.6)
ax.text(0.575, 0.312, "EPISTEMIC\nhow far the nets are\nfrom each other;\n"
                      "shrinks as they agree",
        fontsize=8.0, color=MUTED, ha="center", va="top", linespacing=1.6)

arrow(ax, 0.644, 0.480, 0.688, 0.480, col=C_UN)

# what comes out, and what it is scored with
stage(ax, 0.692, 0.330, 0.288, 0.300, "what comes out, per cell per channel", [
    f"$\\hat{{x}}_*$, $\\sigma^2_*$   $(B,{C},{DT},{H},{W})$",
    "",
    "scored with:  CRPS  ·  NLL",
    "spread/skill ratio  ·  interval coverage",
], C_UN, tfs=9.6, lfs=8.8)
ax.text(0.836, 0.312, "the same four scores are computed for the EnKF from its ensemble std,\n"
                      "so the calibration comparison is like for like",
        fontsize=8.2, color=MUTED, ha="center", va="top", linespacing=1.6)

ax.text(0.5, 0.075, "This half of the comparison does NOT inherit the smoother-versus-filter "
                    "caveat: calibration asks whether a method's stated confidence matches ITS "
                    "OWN error,\nso a method with twice our error can still be perfectly "
                    "calibrated — and one with a tenth of it can be badly overconfident.",
        ha="center", va="center", fontsize=9.4, color=INK, linespacing=1.8,
        bbox=dict(fc=C_UN, ec=C_UN, alpha=0.07, pad=6))
save(fig, "fw_unc_ens.png")


print(f"\n  surrogate {N_SURR:,}   prior {N_PHI:,} (branch {N_BR:,}, psi {N_PSI:,}, up {N_UP})"
      f"   solver {N_SOLV:,}   total {N_TOT:,}   var head {N_VAR}"
      f"\n  n_iter {N_IT}   dT {DT}   lstm hidden {HID}   state {SDIM}   ensemble {NENS}")


# ═══════════════════════════════════════════════════════ 5. what the head changes
# Two columns, not three: the filter is no longer described in this deck. With only our own two
# configurations left, five of the original eight rows were identical between them, so the
# shared part is stated once and the table keeps only what actually differs.
import json as _json
_J = lambda n: _json.load(open(os.path.join(EV, n)))
_SPD = _J("bench_speed_b0.json")
_ms = lambda k: _SPD["runs"][k]["per_frame_s"] * 1000

ROWS = [
    ("trainable parameters", f"{N_TOT:,}", f"{N_TOT:,} $+$ {N_VAR}"),
    ("uncertainty from",     "none", "learnt $\\sigma^2$ $+$ member spread"),
    ("inference",            f"{_ms('varnet_k1'):.1f} ms/frame",
                             f"~{_ms('varnet_k1') * 5:.0f} ms/frame  (5 members)"),
]

fig, ax = canvas(13.0, 4.4)
ax.text(0.5, 0.955, "what the uncertainty head changes, and what it leaves alone",
        ha="center", fontsize=12.5, color=C_VN, style="italic")

# the shared half, said once instead of as five duplicated rows
frame(ax, 0.020, 0.660, 0.960, 0.190, C_VN, 0.07)
ax.text(0.5, 0.800, "identical in both", ha="center", va="center", fontsize=10,
        color=C_VN, fontweight="bold", zorder=6)
ax.text(0.5, 0.722, f"a whole {DT}-frame window at once   ·   the same learnt prior $\\Phi$ "
                    f"({N_PHI:,} params)   ·   observations enter through the observation term "
                    f"of $J$   ·   trained end to end on 32 days",
        ha="center", va="center", fontsize=9.4, color=INK, zorder=6)

COLX = [0.030, 0.420, 0.720]
for x, h, c in zip(COLX, ["", "4DVarNet", "4DVarNet $+$ uncertainty"], [MUTED, C_VN, C_UN]):
    ax.text(x + (0.13 if x > 0.05 else 0), 0.575, h, fontsize=12.5, color=c,
            fontweight="bold", ha="center" if x > 0.05 else "left")
ax.plot([0.020, 0.980], [0.530] * 2, color=INK, lw=1.2)

y = 0.440
for i, (lab, a2, b2) in enumerate(ROWS):
    if i % 2:
        ax.add_patch(FancyBboxPatch((0.020, y - 0.055), 0.960, 0.100,
                                    boxstyle="square,pad=0", fc=MUTED, ec="none",
                                    alpha=0.045, zorder=0))
    ax.text(COLX[0], y, lab, fontsize=11, color=INK, va="center")
    for x, txt, col in zip(COLX[1:], (a2, b2), [C_VN, C_UN]):
        ax.text(x + 0.13, y, txt, fontsize=10.5, color=INK, va="center", ha="center")
    y -= 0.125
ax.plot([0.020, 0.980], [y + 0.058] * 2, color=INK, lw=1.0, alpha=0.4)
ax.text(0.5, y - 0.010, f"the cost $J$, the prior and the solver are untouched — only the "
                        f"{N_VAR}-parameter head and the training loss are new",
        ha="center", fontsize=10, color=C_UN, style="italic")
save(fig, "fw_table.png")
