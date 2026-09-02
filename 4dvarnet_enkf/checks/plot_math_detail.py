"""
plot_math_detail.py  —  the four "open the box" figures for the appendix of the three-way deck

  math_j.png         the variational cost with the channel sum written out as 4 explicit
                     terms, plus what every symbol in it is
  math_conv.png      one output value of psi, written as the full quadruple sum, plus what the
                     zeroed centre voxel does and does not exclude
  math_solver_step.png   one solver step: the four gates, what each decides, and the whole
                         update worked out at one voxel
(the per-step trajectory was measured too — see checks/diag_solver_trace.py's own printout —
 but it is not drawn here.)

These sit behind the framework diagrams: fw_*.png says what happens, these say how it is
computed. Nothing here is a new experiment — every number is read from a checkpoint, from the
exported observation masks, or from a module's own shapes.

    python3 checks/plot_math_detail.py
"""
from __future__ import annotations
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = os.path.join(ROOT, "check_outputs", "eval")
INK, MUTED = "#20334d", "#5b6a7d"
C_EK, C_VN, C_OBS, C_UN = "#b5651d", "#0e6b8a", "#4a7a4a", "#7a4a7a"

sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "checks"))
from model_io import load_solver  # noqa: E402

# ───────────────────────── everything read from the artefacts ─────────────────────────
_S, _A, _CK = load_solver(os.path.join(ROOT, "runs/varnet_b0_k1/varnet_best.pt"), "cpu")
DT, N_IT = _A["dT"], _S.n_iter
C, H, W = 4, 36, 12
CHAN = ["density", "vx", "vy", "var"]
N_PTS = DT * H * W                                        # points per channel, B = 1

# the trained cost weights — these are what the slide quotes
A_OBS = float(_S.var_cost.alpha_obs.detach())
A_REG = float(_S.var_cost.alpha_reg.detach())
W_OBS = _S.var_cost.w_obs.detach().tolist()
W_REG = _S.var_cost.w_reg.detach().tolist()

# psi, from the module itself
_PSI = _S.phi.branch_fine.psi
KT, KH, KW = _PSI.conv.kernel_size
N_TAP = KT * KH * KW
IN_CH, OUT_CH = _PSI.conv.in_channels, _PSI.conv.out_channels
N_PSI_W = OUT_CH * IN_CH * N_TAP
N_PSI_MASKED = OUT_CH * IN_CH                             # one voxel per (out, in) pair
N_PSI = sum(p.numel() for p in _PSI.parameters())
PAD = _PSI.conv.padding

# the solver's two convolutions, by their real weight shapes
_G = _S.grad_net
GW = tuple(_G.lstm.gates.weight.shape)                    # (4*hidden, in+hidden, k, k)
OW = tuple(_G.out.weight.shape)                           # (C*T, hidden, 1, 1)
HID = _G.lstm.hidden_ch
N_GATE = _G.lstm.gates.weight.numel() + _G.lstm.gates.bias.numel()
N_OUT = _G.out.weight.numel()
N_SOLV = sum(p.numel() for p in _G.parameters())
assert N_GATE + N_OUT == N_SOLV, "the solver is more than its two convolutions"

# observation density, from the exported masks the EnKF and the solver both read.
# NOTE the directory names: enkf_k1_full is every-frame observation (the model of record),
# enkf_k4_full is every-4th-frame. check_outputs/enkf_full is the k=4 export, not k=1 —
# reading it here would put a k=4 observation pattern next to k=1 results.
def _omega(sub):
    fs = sorted(glob.glob(os.path.join(ROOT, "check_outputs", sub, "obs_*.npz")))
    if not fs:
        raise SystemExit(f"no exported observation file under check_outputs/{sub}/")
    O = np.load(fs[0])["Omega"]                           # (frames, H, W) bool
    day = os.path.basename(fs[0]).replace("obs_", "").replace(".npz", "")
    return O.reshape(O.shape[0], -1).sum(1).astype(int), day


PF1, _DAY = _omega("enkf_k1_full")
PF4, _DAY4 = _omega("enkf_k4_full")
assert _DAY == _DAY4, "the two strips must be drawn from the same day"
WIN1, WIN4 = PF1[:DT], PF4[:DT]                           # the window the strips show
N_CELL = H * W
SDIM = C * N_CELL                                         # the EnKF's flat state
NENS, RAD, INFL = 100, 7, 1.02


def canvas(w=14.2, h=6.0):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


def frame(ax, x, y, w, h, col, alpha=0.09, lw=1.5, ls="-", z=2):
    for kw in (dict(fc=col, ec=col, alpha=alpha, zorder=z),
               dict(fc="none", ec=col, zorder=z + 2)):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010", lw=lw, ls=ls,
                                    **kw))


def plain(ax, x, y, w, h, text, col, fs=9.6, alpha=0.12):
    frame(ax, x, y, w, h, col, alpha)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            zorder=6, linespacing=1.6)


def lines_box(ax, x, y, w, h, title, lines, col, tfs=9.8, lfs=8.4, alpha=0.09):
    frame(ax, x, y, w, h, col, alpha)
    ax.text(x + w / 2, y + h - 0.048, title, ha="center", va="center", fontsize=tfs,
            color=col, fontweight="bold", zorder=6)
    ax.plot([x + 0.012, x + w - 0.012], [y + h - 0.092] * 2, color=col, lw=0.8, alpha=0.45,
            zorder=6)
    top = y + h - 0.112
    step = min(0.078, (top - y - 0.018) / max(len(lines), 1))
    # a row needs roughly its font height in axes units; below that the rows overlap. Shrink
    # rather than overlap, and say so, so a too-long list is a visible cost not a silent bug.
    _fh = lfs / (ax.figure.get_size_inches()[1] * 72)
    if step < _fh:
        _new = max(6.0, lfs * step / _fh)
        print(f"[fit] '{title[:34]}': {len(lines)} lines in {h:.3f} -> font {lfs:.1f}"
              f"->{_new:.1f}")
        lfs = _new
    ty = top - step / 2
    for ln in lines:
        ax.text(x + 0.016, ty, ln, ha="left", va="center", fontsize=lfs, color=INK, zorder=6)
        ty -= step


def arrow(ax, x1, y1, x2, y2, col=MUTED, ls="-", rad=0.0, lw=1.4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                 lw=lw, color=col, zorder=8, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def save(fig, name):
    p = os.path.join(EV, name)
    fig.savefig(p, dpi=185, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[figure] {p}")


# ══════════════════════════════════════════ A. J(x), with the channel sum written out
fig, ax = canvas(14.2, 6.0)
ax.text(0.5, 0.965, f"The variational cost, with the channel sum written out",
        ha="center", fontsize=12.5, color=C_VN, style="italic")

ax.text(0.030, 0.812, "$J(x) =$", fontsize=16, color=INK, ha="left", va="center")
ax.text(0.128, 0.856,
        "$\\alpha_{obs}^2 \\cdot \\frac{1}{N}\\,($"
        "$w_{obs,0}^2\\,\\Sigma(dy_0)^2 + w_{obs,1}^2\\,\\Sigma(dy_1)^2 + "
        "w_{obs,2}^2\\,\\Sigma(dy_2)^2 + w_{obs,3}^2\\,\\Sigma(dy_3)^2$"
        "$)$",
        fontsize=14, color=C_OBS, ha="left", va="center")
ax.text(0.128, 0.762,
        "$+\\;\\alpha_{reg}^2 \\cdot \\frac{1}{N}\\,($"
        "$w_{reg,0}^2\\,\\Sigma(dx_0)^2 + w_{reg,1}^2\\,\\Sigma(dx_1)^2 + "
        "w_{reg,2}^2\\,\\Sigma(dx_2)^2 + w_{reg,3}^2\\,\\Sigma(dx_3)^2$"
        "$)$",
        fontsize=14, color=C_VN, ha="left", va="center")
ax.text(0.128, 0.700, "fit the observations", fontsize=9.4, color=C_OBS, ha="left",
        style="italic")
ax.text(0.360, 0.700, "obey the learnt dynamics", fontsize=9.4, color=C_VN, ha="left",
        style="italic")

# ── what each symbol is
lines_box(ax, 0.020, 0.310, 0.470, 0.340, "the two residuals", [
    f"$dy_c = (x_c - y_c)\\odot\\Omega_c$      observation misfit",
    f"      zero wherever a cell was not observed, so a",
    f"      blind cell reaches $J$ only through $dx$",
    "",
    f"$dx_c = x_c - \\Phi(x)_c$          prior misfit",
    f"      how far the state is from what the learnt",
    f"      dynamics say it should be",
], C_VN, tfs=10, lfs=9.0)

lines_box(ax, 0.510, 0.310, 0.470, 0.340, f"the sums, and $N$", [
    f"$\\Sigma$ runs over every point of the window:",
    f"      ${DT}$ frames $\\times\\;{H}\\times{W}$ cells $= {N_PTS:,}$ per channel",
    f"      the {DT} frames are simply ADDED IN — the time axis",
    f"      is treated exactly like the two space axes",
    "",
    f"$N = T\\!\\cdot\\!H\\!\\cdot\\!W = {N_PTS:,}$   (the same number)",
    f"      so each $\\Sigma/N$ is a mean square, and $J$ does not",
    f"      grow with the window length",
], C_VN, tfs=10, lfs=9.0)

# ── the learnable weights, with their trained values
_TY = 0.235
ax.text(0.020, _TY + 0.030, f"$c = 0\\ldots3$ are the {C} physical channels. "
                            f"$\\alpha_{{obs}}$ and $\\alpha_{{reg}}$ are scalars, "
                            f"$W_{{obs}}$ and $W_{{reg}}$ are one weight per channel — all four "
                            f"are LEARNED, not set by hand.",
        fontsize=9.6, color=INK, ha="left")
_CX = [("$c$", 0.055), ("channel", 0.135), ("$w_{obs,c}$", 0.240), ("$w_{reg,c}$", 0.330)]
for _t, _x in _CX:
    ax.text(_x, _TY - 0.030, _t, fontsize=9.0, color=MUTED, ha="center")
ax.plot([0.030, 0.375], [_TY - 0.050] * 2, color=MUTED, lw=0.8, alpha=0.6)
_ry = _TY - 0.078
for _i, _c in enumerate(CHAN):
    for _t, _x in ((str(_i), 0.055), (_c, 0.135), (f"{W_OBS[_i]:.3f}", 0.240),
                   (f"{W_REG[_i]:.3f}", 0.330)):
        ax.text(_x, _ry, _t, fontsize=9.0, color=INK, ha="center")
    _ry -= 0.036
ax.text(0.055, _ry + 0.004, f"$\\alpha_{{obs}} = {A_OBS:.3f}$        "
                            f"$\\alpha_{{reg}} = {A_REG:.3f}$",
        fontsize=9.6, color=INK, ha="left", va="top")

ax.text(0.700, 0.130, "The weights exist because the 4 channels are on\n"
                      "different scales — velocity swings by a couple of m/s\n"
                      "where density is a small count. A single unweighted\n"
                      "sum of squares would be a statement about vx alone.",
        fontsize=9.4, color=INK, ha="center", va="center", linespacing=1.9,
        bbox=dict(fc=C_VN, ec=C_VN, alpha=0.07, pad=6))
save(fig, "math_j.png")


# ══════════════════════════════════════════ E/F: the 4 channels, and one worked convolution
# Both read check_outputs/eval/channel_budget.json, produced by checks/diag_channel_budget.py
# on a GPU. Every number below is measured on one real 200-frame window — none is illustrative.
import json as _json

_BUD = os.path.join(EV, "channel_budget.json")
if not os.path.exists(_BUD):
    print(f"[skip] {_BUD} missing — run sbatch/submit_channel_budget.sbatch first")
    raise SystemExit(0)
BD = _json.load(open(_BUD))
CH = BD["channels"]
SC, CO, CV, MIX = BD["scale"], BD["cost"], BD["conv"], BD["mixing"]
X0D, XHD = CO["at_x0"], CO["at_xhat"]


# ══════════════════════════════════════════ E. how the 4 channels are handled
fig, ax = canvas(14.2, 6.6)
ax.text(0.5, 0.978, f"The {len(CH)} channels  —  where they are kept apart, where they are "
                    f"mixed, and which one the cost is actually about",
        ha="center", fontsize=12.5, color=C_VN, style="italic")

# ── the state itself
ax.text(0.5, 0.930, f"one frame is {len(CH)} separate ${H}\\times{W}$ fields on the same grid:  "
                    + "     ".join(f"$c\\!=\\!{i}$  {c}" for i, c in enumerate(CH)),
        ha="center", fontsize=10.2, color=INK)

# ── stage by stage: apart or mixed
ax.text(0.020, 0.882, "stage", fontsize=9.4, color=MUTED, ha="left")
ax.text(0.290, 0.882, "channels", fontsize=9.4, color=MUTED, ha="left")
ax.text(0.435, 0.882, "what happens", fontsize=9.4, color=MUTED, ha="left")
ax.plot([0.015, 0.985], [0.862] * 2, color=INK, lw=1.1)
_y = 0.826
for _i, (_st, _kind, _what) in enumerate(MIX):
    _mixed = _kind.startswith("MIXED")
    if _i % 2:
        ax.add_patch(Rectangle((0.015, _y - 0.021), 0.970, 0.042, fc=MUTED, ec="none",
                               alpha=0.045, zorder=0))
    ax.text(0.020, _y, _st, fontsize=9.2, color=INK, va="center", ha="left")
    ax.text(0.290, _y, _kind, fontsize=9.0, va="center", ha="left",
            color=C_EK if _mixed else C_OBS, fontweight="bold" if _mixed else "normal")
    ax.text(0.435, _y, _what, fontsize=8.4, color=MUTED, va="center", ha="left")
    _y -= 0.0445
ax.plot([0.015, 0.985], [_y + 0.024] * 2, color=INK, lw=1.0, alpha=0.4)

# ── the scales, which is why W exists at all
_TY = 0.400
ax.text(0.020, _TY + 0.052, "the scales are why $W_{obs}$, $W_{reg}$ and the per-channel "
                            "$\\sigma^2$ floor all exist",
        fontsize=10, color=C_VN, fontweight="bold", ha="left")
_COLS = [("channel", 0.045), ("std of truth", 0.175), ("range", 0.300),
         ("obs_std", 0.415), ("$W_{obs}$", 0.505), ("$W_{reg}$", 0.585)]
for _h, _x in _COLS:
    ax.text(_x, _TY, _h, fontsize=8.6, color=MUTED, ha="center")
ax.plot([0.020, 0.625], [_TY - 0.020] * 2, color=MUTED, lw=0.8, alpha=0.6)
_ry = _TY - 0.048
for _i, _c in enumerate(CH):
    _s = SC[_c]
    for _v, _x in ((_c, 0.045), (f"{_s['std']:.4f}", 0.175),
                   (f"[{_s['vmin']:.2f}, {_s['vmax']:.2f}]", 0.300),
                   (f"{_s['obs_std']:.5f}", 0.415),
                   (f"{CO['w_obs'][_i]:.3f}", 0.505), (f"{CO['w_reg'][_i]:.3f}", 0.585)):
        ax.text(_x, _ry, _v, fontsize=8.6, color=INK, ha="center")
    _ry -= 0.040
_rng = max(SC[c]["std"] for c in CH) / min(SC[c]["std"] for c in CH)
_orng = max(SC[c]["obs_std"] for c in CH) / min(SC[c]["obs_std"] for c in CH)
ax.text(0.020, _ry + 0.008, f"the widest channel is {_rng:.1f}$\\times$ the narrowest, and "
                            f"their sensor noises differ by {_orng:.0f}$\\times$ — a single "
                            f"unweighted MSE\nwould be a statement about vx and nothing else",
        fontsize=8.6, color=MUTED, ha="left", va="top", linespacing=1.7)

# ── the cost, decomposed, on one real window
_BX, _BY = 0.655, 0.055
frame(ax, _BX, _BY, 0.330, 0.400, C_VN, 0.08)
ax.text(_BX + 0.165, _BY + 0.372, f"$J$ on one real window, per channel", ha="center",
        va="center", fontsize=9.8, color=C_VN, fontweight="bold", zorder=6)
ax.plot([_BX + 0.012, _BX + 0.318], [_BY + 0.348] * 2, color=C_VN, lw=0.8, alpha=0.45,
        zorder=6)
_hx = [(_BX + 0.055, "channel"), (_BX + 0.165, "at $x_0$"), (_BX + 0.265, "at $\\hat{x}$")]
for _x, _t in _hx:
    ax.text(_x, _BY + 0.318, _t, fontsize=8.4, color=MUTED, ha="center", zorder=6)
_cy = _BY + 0.278
for _c in CH:
    ax.text(_BX + 0.055, _cy, _c, fontsize=8.6, color=INK, ha="center", zorder=6)
    for _x, _d in ((_BX + 0.165, X0D), (_BX + 0.265, XHD)):
        _sh = _d["per_channel"][_c]["share"]
        ax.text(_x, _cy, f"{_sh * 100:.1f}%", fontsize=8.6, zorder=6, ha="center",
                color=C_EK if _sh > 0.4 else INK,
                fontweight="bold" if _sh > 0.4 else "normal")
    _cy -= 0.040
ax.plot([_BX + 0.012, _BX + 0.318], [_cy + 0.020] * 2, color=MUTED, lw=0.8, alpha=0.5,
        zorder=6)
ax.text(_BX + 0.165, _cy - 0.016, f"$J$          {X0D['J']:.4f}      {XHD['J']:.4f}",
        fontsize=8.8, color=INK, ha="center", zorder=6)
ax.text(_BX + 0.165, _cy - 0.056,
        f"obs term   {X0D['obs_total']:.4f}      {XHD['obs_total']:.4f}\n"
        f"prior term {X0D['reg_total']:.4f}      {XHD['reg_total']:.4f}",
        fontsize=8.6, color=MUTED, ha="center", va="center", zorder=6, linespacing=1.7)
ax.text(_BX + 0.165, _cy - 0.115,
        f"{BD['day'].split('_')[0]}, window {BD['window']}",
        fontsize=7.8, color=MUTED, ha="center", style="italic", zorder=6)

ax.text(0.335, 0.030, f"Two things to read off that table.  The observation term starts at "
                      f"EXACTLY {X0D['obs_total']:.4f}: $x_0$ fills the observed cells in with "
                      f"$y$ verbatim, so\n$(x_0-y)\\odot\\Omega \\equiv 0$ and the first descent "
                      f"step is driven entirely by the prior.  And vx is "
                      f"{XHD['per_channel']['vx']['share'] * 100:.0f}% of $J$ at $\\hat{{x}}$ — "
                      f"the cost is mostly about velocity.",
        ha="center", va="bottom", fontsize=9.2, color=INK, linespacing=1.85,
        bbox=dict(fc=C_VN, ec=C_VN, alpha=0.07, pad=6))
save(fig, "math_channels.png")


# ══════════════════════════════════════════ F. psi: the sum, and the same sum on real numbers
# B (the formula and the stencil) and F (the worked 3x3) used to be two slides. They both drew
# the masked centre tap and both explained it, so they are one slide now. What used to be F's
# steps 3 and 4 -- through phi, and the residual that enters J -- is on the prior-chain page,
# which does it properly for all three layers.
_P = np.array(CV["patch"])
_W = np.array(CV["weight"])
_PT = np.array(CV["per_tau"])
_PC = np.array(CV["partial_per_channel"])
_t, _h, _w, _o = CV["t"], CV["h"], CV["w"], CV["out_feature"]

fig, ax = canvas(14.2, 6.4)
ax.text(0.5, 0.972, f"How $\\psi$ computes ONE value  —  the sum, then the same sum on real "
                    f"numbers",
        ha="center", fontsize=12.5, color=C_VN, style="italic")
ax.text(0.5, 0.888, "$\\psi_o(t,h,w) = \\sum_{c=1}^{" + str(IN_CH) + "}"
                    "\\sum_{\\tau=-1}^{1}\\sum_{i=-1}^{1}\\sum_{j=-1}^{1}"
                    "M[\\tau,i,j]\\;W[o,c,\\tau,i,j]\\;x[c,\\,t\\!+\\!\\tau,\\,"
                    "h\\!+\\!i,\\,w\\!+\\!j]\\;+\\;b_o$",
        ha="center", fontsize=13, color=INK)
ax.text(0.5, 0.828, f"$M$ is the mask: 1 everywhere except $M[0,0,0]=0$.   Below: feature "
                    f"${_o}$ at voxel $({_t},{_h},{_w})$ of {BD['day'].split('_')[0]}, checked "
                    f"against $\\mathtt{{conv3d}}$ in checks/diag\\_channel\\_budget.py.",
        ha="center", fontsize=8.8, color=MUTED)

# ── step 1: the density channel's same-frame slice, in full
ax.text(0.020, 0.762, f"the {CH[0]} channel at $\\tau=0$ (the same frame), written out",
        fontsize=10, color=C_VN, fontweight="bold", ha="left")
_GX, _GY, _G = 0.020, 0.560, 0.040
for _lbl, _x, _v, _f in (("input $x$", _GX, _P[0, 1], "{:.3f}"),
                         ("weight $W$", _GX + 0.150, _W[0, 1], "{:+.4f}"),
                         ("product", _GX + 0.300, _P[0, 1] * _W[0, 1], "{:+.4f}")):
    ax.text(_x + 1.5 * _G, _GY + 3 * _G + 0.020, _lbl, ha="center", fontsize=9.2, color=INK)
    for _i in range(3):
        for _j in range(3):
            _dead = _i == 1 and _j == 1
            ax.add_patch(Rectangle((_x + _j * _G, _GY + (2 - _i) * _G), _G, _G,
                                   fc="white" if _dead else C_VN, ec=C_EK if _dead else C_VN,
                                   lw=1.6 if _dead else 0.8, alpha=1.0 if _dead else 0.14,
                                   zorder=3))
            ax.text(_x + (_j + 0.5) * _G, _GY + (2.5 - _i) * _G,
                    "—" if _dead else _f.format(_v[_i, _j]), ha="center", va="center",
                    fontsize=7.2, color=C_EK if _dead else INK, zorder=6)
ax.text(_GX + 3 * _G + 0.015, _GY + 1.5 * _G, "$\\times$", fontsize=14, color=MUTED,
        ha="center", va="center")
ax.text(_GX + 0.150 + 3 * _G + 0.015, _GY + 1.5 * _G, "$=$", fontsize=14, color=MUTED,
        ha="center", va="center")
ax.text(_GX + 0.300 + 3 * _G + 0.020, _GY + 1.5 * _G, f"sum\n$= {_PT[0, 1]:+.4f}$",
        fontsize=10, color=INK, ha="left", va="center", linespacing=1.6)
ax.text(0.020, 0.528, f"the centre is blanked by the mask. The value there is "
                      f"{CV['centre_values'][0]:.4f}, and the unmasked\nweight would have "
                      f"contributed ${CV['dropped_if_unmasked'][0]:+.4f}$ — one term out of "
                      f"{IN_CH * (N_TAP - 1)}.",
        fontsize=8.6, color=C_EK, ha="left", va="top", linespacing=1.7)

# ── step 2: all 12 partial sums
ax.text(0.020, 0.442, f"the same thing for every channel and every frame offset  —  "
                      f"{IN_CH}$\\times$3 partial sums",
        fontsize=10, color=C_VN, fontweight="bold", ha="left")
_TX, _TY2 = 0.030, 0.398
for _j, _lab in enumerate(("$\\tau=-1$", "$\\tau=0$", "$\\tau=+1$", "per channel")):
    ax.text(_TX + 0.110 + _j * 0.098, _TY2, _lab, fontsize=8.8, color=MUTED, ha="center")
ax.plot([_TX, _TX + 0.465], [_TY2 - 0.018] * 2, color=MUTED, lw=0.8, alpha=0.6)
_ry = _TY2 - 0.048
for _i, _c in enumerate(CH):
    ax.text(_TX + 0.006, _ry, _c, fontsize=8.8, color=INK, ha="left")
    for _j in range(3):
        ax.text(_TX + 0.110 + _j * 0.098, _ry, f"{_PT[_i, _j]:+.4f}", fontsize=8.8, color=INK,
                ha="center")
    ax.text(_TX + 0.110 + 3 * 0.098, _ry, f"{_PC[_i]:+.4f}", fontsize=8.8, color=INK,
            ha="center", fontweight="bold")
    _ry -= 0.040
ax.plot([_TX, _TX + 0.465], [_ry + 0.020] * 2, color=MUTED, lw=0.8, alpha=0.6)
ax.text(_TX + 0.006, _ry - 0.014, f"$+$ bias {CV['bias']:+.4f}", fontsize=8.8, color=MUTED,
        ha="left")
ax.text(_TX + 0.110 + 3 * 0.098, _ry - 0.014,
        f"$\\psi_{{{_o}}} = {CV['psi']:+.4f}$", fontsize=10, color=C_VN, ha="center",
        fontweight="bold")
ax.text(_TX + 0.006, _ry - 0.056, f"ReLU $\\rightarrow {CV['relu']:+.4f}$   (a negative "
                                  f"$\\psi$ would be zeroed)",
        fontsize=8.8, color=MUTED, ha="left")

# ── what one output value costs, and what phi does next
lines_box(ax, 0.530, 0.395, 0.230, 0.400, "what one output value costs", [
    f"reads ${IN_CH}\\times{N_TAP - 1} = {IN_CH * (N_TAP - 1)}$ input values",
    "",
    f"weights  ${OUT_CH}\\times{IN_CH}\\times{N_TAP} = {N_PSI_W:,}$",
    f"bias                       ${OUT_CH}$",
    f"of those, {N_PSI_MASKED} are held at zero",
    f"$\\Rightarrow$ {N_PSI_W - N_PSI_MASKED:,} live weights",
    "",
    f"padding {tuple(PAD)}: at $t\\!=\\!1$, $t\\!=\\!{DT}$",
    "and on the grid border the missing",
    "neighbours are read as ZERO",
], C_VN, tfs=9.6, lfs=8.6)

lines_box(ax, 0.775, 0.395, 0.210, 0.400, f"then $\\varphi$ — kernel 1", [
    f"at each $(t,h,w)$ separately:",
    f"   ${OUT_CH}\\rightarrow{OUT_CH}$, ReLU, "
    f"${OUT_CH}\\rightarrow{C}$",
    "",
    "a per-voxel MLP: it adds depth",
    "but cannot reach across time or",
    "space, so no amount of it can put",
    "back a dependence on the",
    "predicted cell",
    "",
    "the next slide follows all three",
    "layers at this same voxel",
], C_VN, tfs=9.6, lfs=8.6)

ax.text(0.5, 0.075, f"So $\\Phi$ cannot be the identity — that is all the mask has to buy. It "
                    f"is NOT the stronger claim that $\\Phi$ ignores frame $t$: "
                    f"$x(t,h\\pm1,w\\pm1)$ is read freely,\nand {N_TAP - 1}/{N_TAP} of the "
                    f"stencil is live. The paper's Sec. 3.2 wording is "
                    f"$\\psi(x)(s)\\perp x(s)$, with $s$ a space-time POINT.",
        ha="center", va="center", fontsize=9.4, color=INK, linespacing=1.9,
        bbox=dict(fc=C_VN, ec=C_VN, alpha=0.07, pad=6))
save(fig, "math_conv.png")


# ══════════════════════════════════════════ H. the whole prior chain at one voxel
# The psi figure above covers one of 32 features of the first of three weight layers of one of
# two branches. This follows the same voxel all the way to Phi(x) and the residual that enters J.
CN = BD.get("chain")
if CN is None:
    print("[skip] channel_budget.json predates the chain block — rerun the sbatch")
    raise SystemExit(0)
_PR = CN["params"]

fig, ax = canvas(14.2, 6.8)
ax.text(0.5, 0.972, f"The whole prior, at one voxel  —  $(t,h,w) = "
                    f"({CN['voxel'][0]},{CN['voxel'][1]},{CN['voxel'][2]})$ of "
                    f"{BD['day'].split('_')[0]}, following $x$ all the way to $\\Phi(x)$",
        ha="center", fontsize=12.2, color=C_VN, style="italic")
ax.text(0.5, 0.930, f"a branch is THREE weight layers, not one: $\\psi$ then two pointwise "
                    f"convs, with a ReLU after each of the first two.  $\\Phi$ is two such "
                    f"branches plus a learned upsampling.",
        ha="center", fontsize=9.2, color=MUTED)

_SX, _SW, _S4, _RH = 0.115, 0.385, 0.052, 0.050
_NX32, _NX4 = _SX + _SW + 0.012, _SX + 4 * _S4 + 0.012


def strip32(y, vals, label, note):
    v = np.array(vals, float)
    m = np.abs(v).max() or 1.0
    cw = _SW / len(v)
    for _i, _v in enumerate(v):
        dead = _v == 0.0
        ax.add_patch(Rectangle((_SX + _i * cw, y), cw, _RH,
                               fc="white" if dead else (C_VN if _v > 0 else C_EK),
                               ec="#c8ced6" if dead else "none",
                               lw=0.5, alpha=1.0 if dead else 0.18 + 0.72 * abs(_v) / m,
                               zorder=3))
    ax.add_patch(Rectangle((_SX, y), _SW, _RH, fc="none", ec=MUTED, lw=0.7, alpha=0.5,
                           zorder=5))
    ax.text(_SX - 0.010, y + _RH / 2, label, ha="right", va="center", fontsize=9.2, color=INK)
    ax.text(_NX32, y + _RH / 2, note, ha="left", va="center", fontsize=8.2, color=MUTED)
    return cw


def strip4(y, vals, label, note, col=C_VN, bold=False):
    for _i, _v in enumerate(vals):
        ax.add_patch(Rectangle((_SX + _i * _S4, y), _S4, _RH, fc=col, ec=col, lw=0.8,
                               alpha=0.14, zorder=3))
        ax.text(_SX + (_i + 0.5) * _S4, y + _RH / 2, f"{_v:+.4f}", ha="center", va="center",
                fontsize=8.4, color=INK, zorder=6,
                fontweight="bold" if bold else "normal")
    ax.text(_SX - 0.010, y + _RH / 2, label, ha="right", va="center", fontsize=9.2, color=INK)
    ax.text(_NX4, y + _RH / 2, note, ha="left", va="center", fontsize=8.2, color=MUTED)


for _i, _c in enumerate(CH):
    ax.text(_SX + (_i + 0.5) * _S4, 0.872, _c, ha="center", fontsize=8.4, color=MUTED)

_Y = 0.812
strip4(_Y, CN["x"], "$x$", f"the state at that voxel — {len(CH)} numbers")
_Y -= 0.086
_cw = strip32(_Y, CN["psi"], "$\\psi(x)$",
              f"{len(CN['psi'])} features, one $3^3$ conv  ·  {_PR['psi']:,} params")
# mark the feature the previous figure worked out, in the gap above the strip
_fx = _SX + (CV["out_feature"] + 0.5) * _cw
ax.add_patch(Rectangle((_SX + CV["out_feature"] * _cw, _Y), _cw, _RH, fc="none", ec=C_EK,
                       lw=1.4, zorder=7))
ax.text(_fx, _Y + _RH + 0.006, f"feature {CV['out_feature']} $= {CV['psi']:+.4f}$, worked out "
                               f"on the previous slide",
        ha="left", va="bottom", fontsize=7.8, color=C_EK)
_Y -= 0.086
strip32(_Y, CN["psi_relu"], "ReLU",
        f"{CN['n_dead_after_psi']} of {len(CN['psi'])} features go to zero here")
_Y -= 0.086
strip32(_Y, CN["phi0"], "$\\varphi_0(\\cdot)$",
        f"pointwise $32\\!\\rightarrow\\!32$, a matrix multiply  ·  {_PR['phi0']:,}")
_Y -= 0.086
strip32(_Y, CN["phi0_relu"], "ReLU", f"{CN['n_dead_after_phi0']} of {len(CN['phi0'])} go to zero")
_Y -= 0.086
strip4(_Y, CN["fine"], "$\\varphi_2(\\cdot)$",
       f"pointwise $32\\!\\rightarrow\\!{len(CH)}$  —  this is $\\Phi_2(x)$, the FINE branch "
       f"·  {_PR['phi2']}", bold=True)
_Y -= 0.098
strip4(_Y, CN["coarse_up"], "$+$ coarse",
       f"the same three layers on the "
       f"${CN['coarse_grid'][0]}\\times{CN['coarse_grid'][1]}$ pooled grid, then upsampled  ·  "
       f"{_PR['branch']:,} $+$ {_PR['up']}", col=C_EK)
_Y -= 0.086
strip4(_Y, CN["phi"], "$= \\Phi(x)$", "paper Eq. 10  —  the two branches summed", bold=True)
_Y -= 0.086
strip4(_Y, CN["residual"], "$x - \\Phi(x)$", "what the prior term of $J$ squares",
       col=C_OBS, bold=True)

# ── the ledger, so the three layers and the two branches add up on screen
lines_box(ax, 0.735, 0.470, 0.250, 0.325, "the three layers, per branch", [
    f"$\\psi$     Conv3d $({len(CH)}\\!\\rightarrow\\!32)$, $3^3$     {_PR['psi']:,}",
    f"$\\varphi_0$    Conv3d $(32\\!\\rightarrow\\!32)$, $1^3$    {_PR['phi0']:,}",
    f"$\\varphi_2$    Conv3d $(32\\!\\rightarrow\\!{len(CH)})$, $1^3$       {_PR['phi2']}",
    f"                          {_PR['branch']:,}",
    "",
    f"$\\times2$ branches            {2 * _PR['branch']:,}",
    f"$+$ learned upsampling        {_PR['up']}",
    f"$= \\Phi$                    {_PR['total']:,}",
], C_VN, tfs=9.6, lfs=8.6)

ax.text(0.860, 0.408, f"$\\psi$ is the ONLY layer that reaches across time and space.\n"
                      f"Both $\\varphi$ layers have kernel 1, so at a fixed voxel they are\n"
                      f"just $32\\times32$ and ${len(CH)}\\times32$ matrix multiplies — no amount\n"
                      f"of depth after $\\psi$ can reintroduce a dependence on the\n"
                      f"cell being predicted.",
        ha="center", va="top", fontsize=8.6, color=MUTED, linespacing=1.8)

ax.text(0.860, 0.150, f"Both the fine branch and $\\Phi$ itself are asserted against\n"
                      f"the module's own forward in "
                      f"checks/diag\\_channel\\_budget.py,\n"
                      f"so these rows cannot drift from what the model computes.",
        ha="center", va="center", fontsize=8.6, color=INK, linespacing=1.8,
        bbox=dict(fc=C_VN, ec=C_VN, alpha=0.07, pad=5))
save(fig, "math_prior_chain.png")

print(f"  chain: psi {CN['n_dead_after_psi']}/{len(CN['psi'])} dead, "
      f"phi0 {CN['n_dead_after_phi0']}/{len(CN['phi0'])} dead;  "
      f"fine {[round(v, 3) for v in CN['fine']]} + coarse "
      f"{[round(v, 3) for v in CN['coarse_up']]} = Phi {[round(v, 3) for v in CN['phi']]}")


# ══════════════════════════════════════════ I/J: the solver, one step and then all 20
# Both read check_outputs/eval/solver_trace.json from checks/diag_solver_trace.py, which pulls
# the numbers out of the real forward pass with hooks rather than re-implementing the loop.
_TRF = os.path.join(EV, "solver_trace.json")
if not os.path.exists(_TRF):
    print(f"[skip] {_TRF} missing — run sbatch/submit_solver_trace.sbatch first")
    raise SystemExit(0)
TR = _json.load(open(_TRF))
ST, WK, TP = TR["steps"], TR["walk"], TR["params"]
_G = _S.grad_net
GWS = tuple(_G.lstm.gates.weight.shape)
OWS = tuple(_G.out.weight.shape)

# ══════════════════════════════════════════ I. the gate convolution, on its own page
GC = TR["gconv"]
_GP = np.array(GC["per_tap"])


def shape_chain(ax, cy):
    """The (ghat, h) -> gates -> cell -> u chain, drawn identically on both solver pages."""
    for _t, _x, _w, _col in ((f"$\\hat{{g}}$\n$({C}\\!\\cdot\\!{DT}={C * DT},{H},{W})$", 0.020,
                              0.115, C_OBS),
                             (f"$h_{{prev}}$\n$({TR['hidden']},{H},{W})$", 0.155, 0.090, C_EK),
                             (f"concat\n$({GC['in_ch']},{H},{W})$", 0.265, 0.090, C_VN),
                             (f"$\\mathrm{{Conv}}_{{3\\times3}}$\n"
                              f"$\\rightarrow({4 * TR['hidden']},{H},{W})$", 0.375, 0.115,
                              C_VN),
                             (f"$i,f,o,\\tilde{{g}}$\n$4\\times{TR['hidden']}$", 0.510, 0.090,
                              C_VN),
                             (f"cell\n$c,h$", 0.620, 0.070, C_VN),
                             (f"$\\mathrm{{Conv}}_{{1\\times1}}$\n"
                              f"$\\rightarrow({C * DT},{H},{W})$", 0.710, 0.115, C_VN),
                             (f"$x \\leftarrow x-u/{TR['n_iter']}$", 0.845, 0.135, C_OBS)):
        plain(ax, _x, cy, _w, 0.056, _t, _col, 8.0, alpha=0.13)
    for _x1, _x2 in ((0.137, 0.153), (0.247, 0.263), (0.357, 0.373), (0.492, 0.508),
                     (0.602, 0.618), (0.692, 0.708), (0.827, 0.843)):
        arrow(ax, _x1, cy + 0.028, _x2, cy + 0.028, lw=1.1)


fig, ax = canvas(14.2, 6.2)
ax.text(0.5, 0.972, f"The gate convolution  —  {GC['n_terms']:,} terms per output value, and "
                    f"{TP['gates'] / TP['total'] * 100:.1f}% of the solver's parameters",
        ha="center", fontsize=12.2, color=C_VN, style="italic")
shape_chain(ax, 0.856)

ax.text(0.5, 0.766,
        "$\\mathrm{pre}[q,h,w] = \\sum_{p=0}^{" + f"{GC['in_ch'] - 1}" + "}"
        "\\sum_{dh=-1}^{1}\\sum_{dw=-1}^{1} W[q,p,dh,dw]\\;"
        "\\mathrm{in}[p,\\,h\\!+\\!dh,\\,w\\!+\\!dw] \\;+\\; b[q]$",
        fontsize=13.5, color=INK, ha="center", va="center")
ax.text(0.5, 0.706, f"$\\mathrm{{in}} = [\\,\\hat{{g}}\\;;\\,h_{{prev}}\\,]$ — "
                    f"{C * DT} gradient channels then {TR['hidden']} hidden ones.  Worked out "
                    f"below for the forget gate of hidden channel {GC['hidden']} at voxel "
                    f"$({GC['voxel'][0]},{GC['voxel'][1]},{GC['voxel'][2]})$, step "
                    f"{GC['step']}.",
        ha="center", fontsize=8.8, color=MUTED)

lines_box(ax, 0.020, 0.300, 0.300, 0.360, f"where the {GC['n_terms']:,} terms come from", [
    f"$\\hat{{g}}$        {GC['n_grad_terms']:,} terms      {GC['from_grad']:+.4f}",
    f"$h_{{prev}}$      {GC['n_hidden_terms']:,} terms        {GC['from_hidden']:+.4f}",
    f"bias                            {GC['bias']:+.4f}",
    f"                              {GC['total']:+.4f}",
    "",
    f"$\\sigma({GC['total']:.4f}) = {GC['sigmoid']:.6f}$   $= f$",
    "",
    f"by absolute mass the gradient is "
    f"{GC['absmass_grad'] / (GC['absmass_grad'] + GC['absmass_hidden']) * 100:.0f}% —",
    f"this gate is driven by the current",
    f"gradient, not by memory",
], C_VN, tfs=9.8, lfs=8.8)

lines_box(ax, 0.350, 0.300, 0.280, 0.360, "the gradient half, by physical channel", [
    f"   {c:<9s} {v:+.4f}" for c, v in zip(CH, GC["per_channel"])
] + [
    f"   {'total':<9s} {sum(GC['per_channel']):+.4f}",
    "",
    f"each is a sum over {DT} frames",
    f"$\\times$ 9 spatial taps",
    "",
    f"the {C * DT} axis is $c\\!\\cdot\\!T+t$, so",
    f"unfolding it back to $({C},{DT})$ is",
    f"what makes this split possible",
], C_VN, tfs=9.8, lfs=8.8)

ax.text(0.660, 0.634, "the 9 spatial taps", fontsize=9.8, color=C_VN, fontweight="bold",
        ha="left")
_TX, _TY, _TC = 0.700, 0.410, 0.062
_mx = np.abs(_GP).max()
for _i in range(3):
    for _j in range(3):
        _v = _GP[_i, _j]
        ax.add_patch(Rectangle((_TX + _j * _TC, _TY + (2 - _i) * _TC), _TC, _TC,
                               fc=C_VN if _v > 0 else C_EK, ec="#c8ced6", lw=0.6,
                               alpha=0.15 + 0.70 * abs(_v) / _mx, zorder=3))
        ax.text(_TX + (_j + 0.5) * _TC, _TY + (2.5 - _i) * _TC, f"{_v:+.2f}", ha="center",
                va="center", fontsize=8.6, color=INK, zorder=6,
                fontweight="bold" if _i == 1 and _j == 1 else "normal")
ax.text(_TX + 1.5 * _TC, 0.382, f"the centre tap alone carries {_GP[1, 1]:+.2f} of the "
                                f"{GC['total']:+.2f}", fontsize=8.6, color=MUTED, ha="center")
ax.text(_TX + 1.5 * _TC, 0.342, "nothing is masked here — this convolution\n"
                                "belongs to the solver, not to the prior",
        fontsize=8.6, color=C_EK, ha="center", va="top", linespacing=1.7, style="italic")

_fo = abs(GC["other_frames"]) / (abs(GC["frame_t"]) + abs(GC["other_frames"])) * 100
plain(ax, 0.020, 0.070, 0.960, 0.140,
      f"the frame this voxel belongs to, $t={GC['voxel'][0]}$, contributes "
      f"{GC['frame_t']:+.4f}      the other {DT - 1} frames contribute "
      f"{GC['other_frames']:+.4f}\n\n"
      f"{_fo:.1f}% of the drive comes from elsewhere in the window — the all-to-all mixing, "
      f"measured rather than asserted", C_EK, 11.0, alpha=0.11)
save(fig, "math_solver_conv.png")


# ══════════════════════════════════════════ J. the gates, the cell, and the update
fig, ax = canvas(14.2, 6.2)
ax.text(0.5, 0.972, f"From the gates to the update  —  what each of the four decides, then the "
                    f"cell, then the read-out",
        ha="center", fontsize=12.2, color=C_VN, style="italic")
shape_chain(ax, 0.856)

ax.text(0.020, 0.782, f"the gate convolution's ${4 * TR['hidden']}$ outputs are four groups of "
                      f"{TR['hidden']}, and each group has a different job",
        fontsize=10, color=C_VN, fontweight="bold", ha="left")
for _x, _t in ((0.040, ""), (0.100, "gate"), (0.215, "squash"), (0.300, "decides")):
    ax.text(_x, 0.740, _t, fontsize=8.8, color=MUTED, ha="left")
ax.plot([0.020, 0.980], [0.722] * 2, color=MUTED, lw=0.9, alpha=0.6)
_y = 0.688
for _sym, _nm, _sq, _wh in (
        ("$i$", "input", "$\\sigma$", "how much of what this step just computed is written "
                                      "into memory"),
        ("$f$", "forget", "$\\sigma$", "how much of the PREVIOUS step's memory survives"),
        ("$\\tilde{g}$", "candidate", "$\\tanh$", "the content this step computed, before any "
                                                  "gating"),
        ("$o$", "output", "$\\sigma$", "how much of the memory is read out as the update")):
    ax.text(0.040, _y, _sym, fontsize=11, color=C_VN, ha="left", va="center")
    ax.text(0.100, _y, _nm, fontsize=9.4, color=INK, ha="left", va="center")
    ax.text(0.215, _y, _sq, fontsize=10, color=INK, ha="left", va="center")
    ax.text(0.300, _y, _wh, fontsize=9.0, color=MUTED, ha="left", va="center")
    _y -= 0.046
ax.plot([0.020, 0.980], [_y + 0.024] * 2, color=MUTED, lw=0.9, alpha=0.5)
ax.text(0.040, _y - 0.004, f"each is $(B,{TR['hidden']},{H},{W})$ — one value per hidden "
                           f"channel per cell — and each is its own {GC['n_terms']:,}-term "
                           f"convolution over the same {GC['in_ch']} input channels",
        fontsize=8.8, color=MUTED, ha="left", va="top", style="italic")

lines_box(ax, 0.020, 0.150, 0.465, 0.288,
          f"the cell, at hidden channel {WK['hidden']} of voxel "
          f"$({WK['voxel'][0]},{WK['voxel'][1]},{WK['voxel'][2]})$", [
    f"$c_{{prev}} = {WK['c_prev']:+.6f}$      inherited from step {WK['step'] - 1}",
    "",
    f"$f = {WK['f']:.6f}$        $i = {WK['i']:.6f}$",
    f"$\\tilde{{g}} = {WK['g']:+.6f}$       $o = {WK['o']:.6f}$",
    "",
    f"$c \\leftarrow f\\,c_{{prev}} + i\\,\\tilde{{g}} = "
    f"{WK['f']:.4f}\\times{WK['c_prev']:+.4f} + {WK['i']:.4f}\\times{WK['g']:+.4f}$",
    f"        $= {WK['c_new']:+.6f}$",
    "",
    f"$h \\leftarrow o\\,\\tanh(c) = {WK['o']:.4f}\\times\\tanh({WK['c_new']:+.4f}) = "
    f"{WK['h']:+.6f}$",
], C_VN, tfs=9.8, lfs=8.8)

lines_box(ax, 0.515, 0.150, 0.465, 0.288, f"the read-out, and what it moves", [
    f"$u = \\sum_k W_{{out}}[r,k]\\,h[k]$   over all {TR['hidden']} hidden channels, no bias",
    f"      $r = c\\!\\cdot\\!T + t = {WK['out_row']}$   "
    f"({WK['phys_channel']}, $t={WK['voxel'][0]}$)",
    "",
    f"this channel's share:  $W_{{out}}[r,{WK['hidden']}] = {WK['w_out_k']:+.4f}$,"
    f"  $\\times h = {WK['contrib_k']:+.6f}$",
    f"all {TR['hidden']} terms:  $u = {WK['u_2d']:+.6f}$",
    "",
    f"$x \\leftarrow x - u/{TR['n_iter']}$:  ${WK['x_before']:+.6f} - "
    f"{WK['u_scaled']:+.6f} = {WK['x_after']:+.6f}$",
    "",
    f"one step moves this cell by "
    f"{abs(WK['u_scaled']) / max(abs(WK['x_before']), 1e-9) * 100:.1f}% of its value",
], C_VN, tfs=9.8, lfs=8.8)

ax.text(0.5, 0.066, f"$h$ and $c$ are $({TR['hidden']},{H},{W})$ and are carried across the "
                    f"{TR['n_iter']} DESCENT STEPS, not across physical time.      "
                    f"The ${C * DT}$ axis is $c\\!\\cdot\\!T+t$ — the {C} channels and {DT} "
                    f"frames flattened together, which is why $dT$ is fixed by the weight "
                    f"shape.\n"
                    f"{TP['total']:,} parameters in all: {TP['gates']:,} in the gate "
                    f"convolution, {TP['out']:,} in the read-out.",
        ha="center", va="center", fontsize=9.2, color=INK, linespacing=1.9,
        bbox=dict(fc=C_VN, ec=C_VN, alpha=0.07, pad=6))
save(fig, "math_solver_step.png")

print(f"  gate conv: {GC['n_terms']:,} terms, grad {GC['from_grad']:+.4f} + hidden "
      f"{GC['from_hidden']:+.4f} + bias {GC['bias']:+.4f} = {GC['total']:+.4f} -> "
      f"sigma {GC['sigmoid']:.6f};  other frames {_fo:.1f}% of the drive")

# ══════════════════════════════════════════ K. the variance head — how it computes
_VHF = os.path.join(EV, "varhead_trace.json")
if not os.path.exists(_VHF):
    print(f"[skip] {_VHF} missing — run sbatch/submit_varhead_trace.sbatch first")
    raise SystemExit(0)
VH = _json.load(open(_VHF))
_U = np.array(VH["inputs"])
_UN, _RO, _MS, _MB, _SP = VH["unit"], VH["readout"], VH["mass"], VH["members"], VH["spots"]
_GRP = [("$\\hat{x}$", 0, C_VN), ("$\\Omega$", C, C_OBS),
        ("$|\\hat{x}-\\Phi(\\hat{x})|$", 2 * C, C_EK)]

fig, ax = canvas(14.2, 6.0)
ax.text(0.5, 0.968, f"The variance head, step by step  —  both convolutions have kernel 1, so "
                    f"at one voxel each is a matrix-vector product",
        ha="center", fontsize=12.2, color=C_UN, style="italic")
ax.text(0.5, 0.926, f"$3\\!\\times\\!{C} = {VH['n_in']} \\rightarrow {VH['hidden']} "
                    f"\\rightarrow {C}$, then $\\sigma^2 = \\mathrm{{obs\\_var}} + "
                    f"\\mathrm{{softplus}}(\\cdot)$.   {VH['params']['total']} parameters.   "
                    f"Voxel $({VH['voxel'][0]},{VH['voxel'][1]},{VH['voxel'][2]})$, member 0.",
        ha="center", fontsize=8.8, color=MUTED)

# ── the 12 inputs
_cw, _x0, _y0 = 0.056, 0.070, 0.792
for _g, _off, _col in _GRP:
    for _i in range(C):
        ax.add_patch(Rectangle((_x0 + (_off + _i) * _cw, _y0), _cw, 0.052, fc=_col, ec=_col,
                               lw=0.8, alpha=0.14, zorder=3))
        ax.text(_x0 + (_off + _i + 0.5) * _cw, _y0 + 0.026, f"{_U[_off + _i]:+.4f}",
                ha="center", va="center", fontsize=8.6, color=INK, zorder=6)
        ax.text(_x0 + (_off + _i + 0.5) * _cw, _y0 + 0.064, CH[_i], ha="center", fontsize=7.8,
                color=MUTED)
    ax.text(_x0 + (_off + C / 2) * _cw, _y0 - 0.026, _g, ha="center", va="top", fontsize=10,
            color=_col)
ax.text(0.064, _y0 + 0.026, "$u =$", ha="right", va="center", fontsize=11.5, color=INK)
ax.text(0.820, _y0 + 0.026, f"the {VH['n_in']} inputs: the reconstruction,\nthe mask, and the "
                            f"prior residual", ha="left", va="center", fontsize=9.0,
        color=MUTED, linespacing=1.7)

# ── layer 1, one hidden unit, all 12 terms — given the whole left half
frame(ax, 0.020, 0.055, 0.470, 0.640, C_UN, 0.08)
ax.text(0.255, 0.652, f"layer 1, hidden unit {_UN['k']}  —  all {VH['n_in']} terms",
        ha="center", va="center", fontsize=10.2, color=C_UN, fontweight="bold", zorder=6)
ax.plot([0.034, 0.476], [0.620] * 2, color=C_UN, lw=0.8, alpha=0.45, zorder=6)
for _t, _x, _ha in (("input", 0.050, "left"), ("channel", 0.190, "right"),
                    ("$u$", 0.278, "right"), ("$W_0$", 0.364, "right"),
                    ("product", 0.462, "right")):
    ax.text(_x, 0.594, _t, fontsize=8.4, color=MUTED, ha=_ha, zorder=6)
_ry = 0.562
for _i in range(VH["n_in"]):
    _g, _col = _GRP[_i // C][0], _GRP[_i // C][2]
    if _i % C == 0:
        ax.text(0.050, _ry, _g, fontsize=8.6, color=_col, ha="left", zorder=6)
    ax.text(0.190, _ry, CH[_i % C], fontsize=8.4, color=INK, ha="right", zorder=6)
    ax.text(0.278, _ry, f"{_U[_i]:+.4f}", fontsize=8.4, color=INK, ha="right", zorder=6)
    ax.text(0.364, _ry, f"{_UN['w'][_i]:+.4f}", fontsize=8.4, color=INK, ha="right", zorder=6)
    _big = abs(_UN["prod"][_i]) > 1
    ax.text(0.462, _ry, f"{_UN['prod'][_i]:+.4f}", fontsize=8.4, ha="right", zorder=6,
            color=C_EK if _big else INK, fontweight="bold" if _big else "normal")
    _ry -= 0.0315
ax.plot([0.300, 0.476], [_ry + 0.018] * 2, color=MUTED, lw=0.8, alpha=0.6, zorder=6)
for _lab, _val, _dy, _fs, _bold in (("sum", f"{sum(_UN['prod']):+.4f}", 0.000, 8.6, False),
                                    ("$+\\,b_0$", f"{_UN['bias']:+.4f}", 0.032, 8.6, False),
                                    ("$z$, then ReLU", f"{_UN['relu']:+.4f}", 0.068, 9.4,
                                     True)):
    ax.text(0.278, _ry - _dy, _lab, fontsize=_fs, color=C_UN if _bold else MUTED, ha="right",
            zorder=6, fontweight="bold" if _bold else "normal")
    ax.text(0.462, _ry - _dy, _val, fontsize=_fs, color=C_UN if _bold else INK, ha="right",
            zorder=6, fontweight="bold" if _bold else "normal")
ax.text(0.255, 0.078, "all four $\\Omega$ products are negative — being observed pushes this "
                      "unit DOWN",
        ha="center", va="center", fontsize=8.6, color=MUTED, style="italic", zorder=6)

# ── layer 2, then the floor
lines_box(ax, 0.520, 0.400, 0.290, 0.295,
          f"layer 2 $\\rightarrow$ {VH['phys_channel']}  —  {VH['hidden']} terms", [
    f"{VH['n_dead']} of the {VH['hidden']} hidden units are 0 at",
    f"this voxel, so {VH['hidden'] - VH['n_dead']} terms are live",
    "",
    f"$\\sum_h W_2[{CH.index(VH['phys_channel'])},h]\\,a[h] = {sum(_RO['prod']):+.4f}$"
    .replace("\\sum_h", "\\Sigma"),
    f"$+\\,b_2 = {_RO['bias']:+.4f}$",
    "",
    f"$v = {_RO['v']:+.4f}$",
], C_UN, tfs=9.8, lfs=9.0)

lines_box(ax, 0.520, 0.055, 0.290, 0.290, "then softplus", [
    f"$\\sigma^2 = \\mathrm{{softplus}}(v)$",
    "",
    f"$\\mathrm{{softplus}}({_RO['v']:.4f}) = {_RO['sig2']:.6f}$",
    f"$\\sigma = {_RO['sigma']:.5f}$",
    "",
    f"nothing floors this — a very negative $v$",
    f"means the head is claiming confidence",
    f"and has to be right about it",
], C_UN, tfs=9.8, lfs=9.0)

# ── the shape chain, light, on the right
_chain = [(f"$u$", f"{VH['n_in']}"), ("$W_0$, ReLU", f"{VH['hidden']}"),
          ("$W_2$", f"{C}"), ("softplus", f"$\\sigma^2$, {C}")]
_cy2 = 0.600
for _lab, _sh in _chain:
    plain(ax, 0.845, _cy2, 0.140, 0.082, f"{_lab}\n{_sh}", C_UN, 8.8, alpha=0.11)
    if _cy2 > 0.30:
        arrow(ax, 0.915, _cy2 - 0.004, 0.915, _cy2 - 0.048, col=C_UN, lw=1.1)
    _cy2 -= 0.130
ax.text(0.915, 0.185, f"{VH['params']['l1']} $+$ {VH['params']['l2']} $=$ "
                      f"{VH['params']['total']} params\nin the whole head",
        ha="center", va="top", fontsize=8.8, color=MUTED, linespacing=1.7)
save(fig, "math_varhead.png")
# ══════════════════════════════════════════ L. the variance head — what comes out
# Two blocks only. Dropped: the observed-vs-blind cell pair, which made the same point as the
# calibration slide's blind-vs-observed sigma but from a single cell of a single channel, and
# which I had already had to label "not calibration"; and the aleatoric/epistemic percentage
# split, which is per-voxel and invites a comparison with the aggregate that then needs
# explaining away.
fig, ax = canvas(14.2, 4.4)
ax.text(0.5, 0.952, "What the head produces", ha="center", fontsize=12.2, color=C_UN,
        style="italic")

# ── the sigma it actually produces, per channel
frame(ax, 0.020, 0.120, 0.470, 0.740, C_UN, 0.08)
ax.text(0.255, 0.800, "$\\sigma$ across the field, per channel", ha="center", va="center",
        fontsize=10.2, color=C_UN, fontweight="bold", zorder=6)
ax.plot([0.034, 0.476], [0.756] * 2, color=C_UN, lw=0.8, alpha=0.45, zorder=6)
for _t, _x in (("channel", 0.090), ("5th", 0.230), ("median", 0.330), ("95th", 0.430)):
    ax.text(_x, 0.712, _t, fontsize=9.4, color=MUTED, ha="center", zorder=6)
_ry = 0.648
for _c in CH:
    _r = VH["sigma_range"][_c]
    for _v, _x in ((_c, 0.090), (f"{_r['p05']:.4f}", 0.230), (f"{_r['median']:.4f}", 0.330),
                   (f"{_r['p95']:.4f}", 0.430)):
        ax.text(_x, _ry, _v, fontsize=9.4, color=INK, ha="center", zorder=6)
    _ry -= 0.062
ax.text(0.255, 0.330, "read a row as: for density, 5% of cells get a $\\sigma$ below 0.0022,\n"
                      "half below 0.0153, and 95% below 0.2901.",
        ha="center", va="center", fontsize=9.0, color=MUTED, zorder=6, linespacing=1.8)
ax.text(0.255, 0.198, "two orders of magnitude between the cells the head is\n"
                      "confident about and the ones it is not — that spread is\n"
                      "the whole point of learning $\\sigma$ instead of fixing it.",
        ha="center", va="center", fontsize=9.0, color=INK, zorder=6, linespacing=1.8)

# ── the five members, combined
frame(ax, 0.520, 0.120, 0.460, 0.740, C_UN, 0.08)
ax.text(0.750, 0.800, f"the five members, at one voxel ({CH[0]})", ha="center", va="center",
        fontsize=10.2, color=C_UN, fontweight="bold", zorder=6)
ax.plot([0.534, 0.966], [0.756] * 2, color=C_UN, lw=0.8, alpha=0.45, zorder=6)
ax.text(0.552, 0.690, "$\\sigma^2_m$", fontsize=9.8, color=INK, ha="left", zorder=6)
ax.text(0.552, 0.626, "$\\hat{x}_m$", fontsize=9.8, color=INK, ha="left", zorder=6)
for _i in range(5):
    ax.text(0.640 + _i * 0.066, 0.690, f"{_MB['var'][_i]:.5f}", fontsize=9.4, color=INK,
            ha="center", zorder=6)
    ax.text(0.640 + _i * 0.066, 0.626, f"{_MB['mu'][_i]:.3f}", fontsize=9.4, color=INK,
            ha="center", zorder=6)
ax.text(0.750, 0.508, f"$\\sigma^2_* = \\overline{{\\sigma^2_m}} + "
                      f"\\mathrm{{Var}}_m(\\hat{{x}}_m)$",
        fontsize=12, color=INK, ha="center", va="center", zorder=6)
ax.text(0.750, 0.424, f"$= {_MB['mean_var']:.6f} + {_MB['var_of_mu']:.6f} = "
                      f"{_MB['combined']:.6f}$",
        fontsize=11.5, color=INK, ha="center", va="center", zorder=6)
ax.text(0.750, 0.344, f"$\\sigma_* = {_MB['combined'] ** 0.5:.5f}$",
        fontsize=12, color=C_UN, ha="center", va="center", zorder=6, fontweight="bold")
ax.text(0.750, 0.222, "the first term is what each network says about itself,\n"
                      "the second is how far the five are from each other.\n"
                      "This is the paper's Sec. 2.4, written out.",
        ha="center", va="center", fontsize=9.0, color=MUTED, zorder=6, linespacing=1.8)
save(fig, "math_varhead_out.png")

print(f"  var head out: sigma range per channel + the 5-member combination "
      f"({_MB['combined'] ** 0.5:.5f})")
