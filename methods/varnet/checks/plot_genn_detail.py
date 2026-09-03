"""
plot_genn_detail.py  —  the prior Φ, one piece per figure

  genn_role.png       where Φ sits: the second term of the variational cost
  genn_psi.png        ψ, the zero-centre convolution — drawn as an actual kernel grid
  genn_phi.png        φ, the pointwise stack
  genn_twoscale.png   Eq. 10, the two branches and their resolutions

Split out of the single overview figure so each piece gets a slide of its own. These are
DIAGRAMS ONLY — the reasoning lives in the slide bullets, where it stays editable. The one
exception is the kernel grid in genn_psi: "the central value is set to zero" is the
constraint the whole architecture rests on, and it is far easier to believe when the zeroed
tap is visible than when it is a sentence.

Shapes and parameter counts are read from the live checkpoint.
"""
from __future__ import annotations
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "checks"))
from model_io import load_solver  # noqa: E402

OUT = os.path.join(ROOT, "check_outputs", "eval")

# The architecture the deck presents. B0 is hidden 32 / kt 3 — i.e. exactly the
# config.yaml defaults, so psi's kernel is 3x3x3 and there is no system-specific block to
# explain away. A2 (hidden 64, kt 5) scores better but is a capacity experiment; presenting the
# baseline configuration keeps the story about the GENN definition rather than about our tuning.
# Flip this one path to re-target every figure.
CKPT = os.path.join(ROOT, "runs/varnet_b0_k1/varnet_best.pt")
S, A, _ = load_solver(CKPT, "cpu")
NP = lambda m: sum(p.numel() for p in m.parameters())
C, T, H, W = 4, A["dT"], 36, 12
HID = A["hidden"]
KT, KH, KW = A.get("kt", 3), A.get("kh", 3), A.get("kw", 3)
BR = S.phi.branch_fine

INK, MUTED = "#20334d", "#5b6a7d"
C_PRIOR, C_DATA, C_RED = "#0e6b8a", "#4a7a4a", "#c02020"


def box(ax, x, y, w, h, text, fc, fs=10):
    for kw in (dict(fc=fc, ec=fc, alpha=0.13, zorder=1), dict(fc="none", ec=fc, zorder=3)):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012", lw=1.5, **kw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, zorder=4, linespacing=1.5)


def arrow(ax, x1, y1, x2, y2, col=None, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 lw=1.4, color=col or MUTED,
                                 connectionstyle=f"arc3,rad={rad}", zorder=2))


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"[figure] {p}")


# ───────────────────────────────────────────────────────── 1. where Φ sits
fig, ax = canvas(12.5, 3.6)
ax.text(0.5, 0.90,
        r"$J(x)\;=\;\alpha_{obs}^2\,\|(x-y)\odot\Omega\|^2\;\;+\;\;\alpha_{reg}^2\,\|\,x-\Phi(x)\,\|^2$",
        ha="center", fontsize=17, color=INK)
ax.text(0.283, 0.755, "fit what the robots measured", ha="center", fontsize=10.5, color=C_DATA)
ax.text(0.672, 0.755, "stay a plausible crowd field", ha="center", fontsize=10.5, color=C_PRIOR)
ax.plot([0.155, 0.41], [0.815, 0.815], lw=1.2, color=C_DATA)
ax.plot([0.545, 0.80], [0.815, 0.815], lw=1.2, color=C_PRIOR)

box(ax, 0.05, 0.22, 0.21, 0.32, f"candidate state  $x$\n({C}, {T}, {H}, {W})", C_DATA, 11)
box(ax, 0.395, 0.19, 0.21, 0.38, f"$\\Phi$\nGENN\n{NP(S.phi):,} parameters", C_PRIOR, 11)
box(ax, 0.74, 0.22, 0.21, 0.32, f"$\\Phi(x)$\n({C}, {T}, {H}, {W})", C_PRIOR, 11)
arrow(ax, 0.26, 0.38, 0.395, 0.38); arrow(ax, 0.605, 0.38, 0.74, 0.38)
ax.text(0.5, 0.09, "same shape in, same shape out — Φ never changes the state's dimensions",
        ha="center", fontsize=9.5, color=MUTED)
save(fig, "genn_role.png")


# ───────────────────────────────────────────────────────── 2. psi
FW, FH = 12.5, 3.7
fig, ax = canvas(FW, FH)
ax.text(0.5, 1.02, f"ψ :  Conv3d({C} → {HID}, kernel {KT}×{KH}×{KW}),  {NP(BR.psi):,} parameters",
        ha="center", fontsize=13, color=C_PRIOR)

# Square cells: the axes span 0..1 on both axes over a non-square figure, so a cell that is
# CW wide has to be CW * (FW/FH) tall to come out square on the page.
CW = 0.062
CH = CW * FW / FH

# left: the kernel, cell by cell — centre time slice
gx, gy = 0.055, 0.235
for i in range(KH):
    for j in range(KW):
        mid = (i == KH // 2 and j == KW // 2)
        ax.add_patch(Rectangle((gx + j * CW, gy + (KH - 1 - i) * CH), CW, CH,
                               fc=C_RED if mid else C_PRIOR, alpha=0.9 if mid else 0.22,
                               ec="white", lw=2.0, zorder=2))
        ax.text(gx + (j + 0.5) * CW, gy + (KH - 0.5 - i) * CH, "0" if mid else "w",
                ha="center", va="center", fontsize=13, zorder=3,
                color="white" if mid else INK, weight="bold" if mid else "normal")
ax.text(gx + KW * CW / 2, gy + KH * CH + 0.045,
        f"centre time slice of the {KT}×{KH}×{KW} kernel", ha="center", fontsize=10.5, color=MUTED)
ax.text(gx + KW * CW / 2, gy - 0.05, "the centre tap is masked to 0\non every forward pass",
        ha="center", fontsize=11, color=C_RED, va="top", linespacing=1.5)

# right: what that means — every neighbour of s is read, s itself is not
cx = 0.385
for i in range(KH):
    for j in range(KW):
        mid = (i == KH // 2 and j == KW // 2)
        ax.add_patch(Rectangle((cx + j * CW, gy + (KH - 1 - i) * CH), CW, CH,
                               fc="white" if mid else C_DATA, alpha=1.0 if mid else 0.22,
                               ec=C_RED if mid else "white", lw=2.0,
                               ls="--" if mid else "-", zorder=2))
ax.text(cx + 1.5 * CW, gy + 1.5 * CH, "$s$", ha="center", va="center", fontsize=14,
        color=C_RED, zorder=3, weight="bold")
ax.text(cx + KW * CW / 2, gy + KH * CH + 0.045, "input read at each neighbour of $s$",
        ha="center", fontsize=10.5, color=MUTED)
ax.text(cx + KW * CW / 2, gy - 0.05, "$x(s)$ itself is never read", ha="center", fontsize=11,
        color=C_RED, va="top")
arrow(ax, cx + KW * CW + 0.03, gy + 1.5 * CH, cx + KW * CW + 0.115, gy + 1.5 * CH)
box(ax, cx + KW * CW + 0.135, gy + CH, 0.17, CH, "$\\psi(x)(s)$", C_PRIOR, 12)
save(fig, "genn_psi.png")


# ───────────────────────────────────────────────────────── 3. phi
fig, ax = canvas(12.5, 3.4)
ax.text(0.5, 0.90, f"φ :  {A['n_phi_layers']} pointwise Conv3d layers, every kernel 1×1×1,  "
                   f"{NP(BR.phi):,} parameters", ha="center", fontsize=13, color=C_PRIOR)

box(ax, 0.010, 0.36, 0.145, 0.30, f"ψ output\n({HID}, {T}, {H}, {W})", C_PRIOR, 10)
box(ax, 0.195, 0.40, 0.085, 0.22, "ReLU", "#888", 10)
box(ax, 0.320, 0.36, 0.155, 0.30, f"Conv3d 1×1×1\n{HID} → {HID}", C_PRIOR, 10)
box(ax, 0.515, 0.40, 0.085, 0.22, "ReLU", "#888", 10)
box(ax, 0.640, 0.36, 0.155, 0.30, f"Conv3d 1×1×1\n{HID} → {C}", C_PRIOR, 10)
box(ax, 0.835, 0.36, 0.145, 0.30, f"branch output\n({C}, {T}, {H}, {W})", C_PRIOR, 10)
for x1, x2 in ((0.155, 0.195), (0.280, 0.320), (0.475, 0.515), (0.600, 0.640), (0.795, 0.835)):
    arrow(ax, x1, 0.51, x2, 0.51)
ax.text(0.5, 0.20,
        f"kernel 1 everywhere → each of the {T}×{H}×{W} = {T * H * W:,} space-time positions is "
        f"transformed on its own,\nusing only its own {HID} feature values. No neighbouring cell, "
        f"no neighbouring frame.",
        ha="center", fontsize=10.5, color=MUTED, va="top", linespacing=1.7)
save(fig, "genn_phi.png")


# ───────────────────────────────────────────────────────── 4. two scales
fig, ax = canvas(12.5, 4.0)
ax.text(0.5, 0.92, r"$\Phi(x) \;=\; Up\left(\Phi_1(Dw(x))\right) \;+\; \Phi_2(x)$",
        ha="center", fontsize=16, color=C_PRIOR)

box(ax, 0.02, 0.40, 0.095, 0.22, f"$x$\n{H}×{W}", C_DATA, 10.5)
box(ax, 0.185, 0.545, 0.145, 0.16, "$Dw$\navg-pool /2", C_PRIOR, 10)
box(ax, 0.40, 0.505, 0.16, 0.235, f"$\\Phi_1$\nψ + φ  on {H // 2}×{W // 2}\n{NP(BR):,} params",
    C_PRIOR, 10)
box(ax, 0.635, 0.545, 0.15, 0.16, f"$Up$\nConvTranspose", C_PRIOR, 10)
box(ax, 0.40, 0.245, 0.16, 0.20, f"$\\Phi_2$\nψ + φ  on {H}×{W}\n{NP(BR):,} params", C_PRIOR, 10)
box(ax, 0.865, 0.40, 0.11, 0.22, f"$+$\n$\\Phi(x)$", C_PRIOR, 11)
arrow(ax, 0.115, 0.545, 0.185, 0.615, rad=0.08)
arrow(ax, 0.115, 0.475, 0.40, 0.345, rad=-0.06)
arrow(ax, 0.33, 0.625, 0.40, 0.625)
arrow(ax, 0.56, 0.625, 0.635, 0.625)
arrow(ax, 0.785, 0.620, 0.865, 0.545, rad=0.08)
arrow(ax, 0.56, 0.345, 0.865, 0.470, rad=-0.06)

save(fig, "genn_twoscale.png")

print(f"\n  prior {NP(S.phi):,}  =  branch {NP(BR):,} × 2  +  up {NP(S.phi.up)}"
      f"   (psi {NP(BR.psi):,} + phi {NP(BR.phi):,} per branch)")
