"""
plot_meeting.py  —  figures for the joint-meeting three-way deck.

Four figures, one per section of the meeting: accuracy, uncertainty scale, uncertainty
usefulness, inference time.

Palette. The project's own C_US/C_EK pair, with one correction: #0e6b8a fails the OKLCH chroma
floor (C=0.092 < 0.10) and reads as grey, so it is replaced by #0074a8 -- the same teal-blue,
properly saturated. The pair passes all six checks in both light and dark
(scratch/validate_palette.py, a Python twin of the dataviz validator since the cluster has no
node). oracle and random are NOT given categorical hues: they are a theoretical bound and a
null model, not methods, so they are grey reference lines like the calibration diagonal already
is. That also removes the CVD collision a third hue caused -- green #4a7a4a sat at deltaE 4.6
from #b5651d under protan/deutan, below the floor of 6.

    python3 checks/plot_meeting.py
"""
from __future__ import annotations
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
EV = os.path.join(ROOT, "check_outputs", "eval")
OUT = os.path.join(ROOT, "check_outputs", "eval")

C_US, C_EK, C_REF = "#0074a8", "#b5651d", "#8a8a8a"
# aleatoric and epistemic are two parts of ONE variance, so they get two steps of one hue
# rather than two categorical hues. #6bb8db is dL 0.215 from #0074a8 (ordinal floor 0.06) and
# 2.15:1 on the surface (light-end floor 2.0).
C_ALE, C_EPI = C_US, "#6bb8db"
INK, INK2 = "#20334d", "#596677"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c8ced6",
                     "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})

J = lambda n: json.load(open(os.path.join(EV, n)))
cov = lambda r, z: r["coverage"][str(z)] * 100


def finish(fig, ax_or_axes, name):
    for ax in np.atleast_1d(ax_or_axes).ravel():
        ax.grid(axis="y", color="#e6eaef", lw=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  {name}", flush=True)


# ── 1. accuracy ────────────────────────────────────────────────────────────────
U, E = J("uncertainty_vsb0.json")["results"], J("uncertainty_enkf_k1.json")["results"]
fig, ax = plt.subplots(figsize=(7.6, 3.9))
groups = ["all cells", "unobserved\n(blind)", "observed"]
keys = ["all", "blind", "observed"]
us = [U[k]["rmse"] for k in keys]
ek = [E[k]["rmse"] for k in keys]
x = np.arange(len(groups)); w = 0.34; g = 0.012   # 2px surface gap between bars
ax.bar(x - w/2 - g, us, w, color=C_US, label="4DVarNet + deep ensemble", zorder=3)
ax.bar(x + w/2 + g, ek, w, color=C_EK, label="EnKF (k=1)", zorder=3)
for xi, (a, b) in enumerate(zip(us, ek)):
    ax.text(xi - w/2 - g, a + 0.004, f"{a:.4f}", ha="center", fontsize=9, color=INK)
    ax.text(xi + w/2 + g, b + 0.004, f"{b:.4f}", ha="center", fontsize=9, color=INK)
ax.annotate("essentially tied (1% apart)", xy=(1, max(us[1], ek[1]) + 0.019), ha="center",
            fontsize=9.5, color=INK2)
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylabel("RMSE  (7 held-out days)")
ax.set_ylim(0, max(us + ek) * 1.30)
ax.legend(frameon=False, loc="upper left", fontsize=9.5)
finish(fig, ax, "mt_rmse.png")

# ── 2. uncertainty scale: the paper's own Sec. A.2 calibration curve ───────────
fig, ax = plt.subplots(figsize=(5.6, 5.1))
zs = np.array(sorted(int(z) for z in U["all"]["coverage"]))
ax.plot([0, 100], [0, 100], "--", color=C_REF, lw=1.5, label="perfectly calibrated", zorder=2)
y = [U["all"]["coverage"][str(z)] * 100 for z in zs]
ax.plot(zs, y, "o-", color=C_US, lw=2.2, ms=5.5, label="our $\\sigma^2$ (5-member ensemble)",
        zorder=3)
ax.fill_between(zs, zs, y, color=C_US, alpha=0.10, zorder=1)
ax.annotate(f"says 10%, catches {cov(U['all'], 10):.0f}%", xy=(10, 61), xytext=(24, 68),
            fontsize=9.5, color=C_US,
            arrowprops=dict(arrowstyle="->", color=C_US, lw=1.1))
# one annotation only: the 90% point and a free-text gloss both said the same thing as the
# shaded gap already does, and both landed on the diagonal
ax.text(70, 24, "the shaded gap is over-caution:\nevery interval is too wide",
        fontsize=9.5, color=INK2, ha="center")
ax.set_xlabel("nominal interval  z%"); ax.set_ylabel("truths actually inside  %")
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13),
          ncol=1, fontsize=9)
finish(fig, ax, "mt_calibration.png")

# ── 3. does sigma know WHICH cell is wrong? ───────────────────────────────────
# One panel, one idea. The sparsification curve that used to sit beside this needs oracle and
# random orderings explained before it can be read; the bars need nothing. The axis runs to 1.0
# so the scale is the message: perfect agreement is the far right, and the bars are slivers.
SP = J("sparsification.json")["per_channel_blind"]
ch = ["density", "vx", "vy", "var"]
val = [SP[c]["spearman"] for c in ch]
fig, ax = plt.subplots(figsize=(8.6, 3.6))
y = np.arange(len(ch))
ax.barh(y, val, 0.52, color=C_US, zorder=3)
for v, yy in zip(val, y):
    ax.text(v + 0.012, yy, f"{v:.3f}", va="center", fontsize=11, color=INK)
ax.axvline(1.0, ls="--", color=C_REF, lw=1.6, zorder=4)
# the y axis is inverted, so the TOP of the plot is the smaller coordinate
ax.text(0.985, -0.62, "perfect agreement", fontsize=10, color=INK2, ha="right")
ax.set_yticks(y); ax.set_yticklabels(ch, fontsize=12)
ax.set_ylim(len(ch) - 0.45, -0.85); ax.invert_yaxis(); ax.invert_yaxis()
ax.set_xlim(0, 1.06); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel("how well $\\sigma$ agrees with where the error actually is\n"
              "0 = no relation at all,   1 = perfect", fontsize=11)
ax.grid(axis="x", color="#e6eaef", lw=0.8); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "mt_agreement.png"), dpi=200, facecolor="white")
plt.close(fig); print("  mt_agreement.png")

# ── 4. inference time ─────────────────────────────────────────────────────────
B = J("bench_speed_vsb0.json")
r = B["runs"]
names = [("enkf_k1", "EnKF (k=1)", C_EK), ("enkf_k4", "EnKF (k=4)", C_EK),
         ("varnet_ens5", "4DVarNet\n5-member ensemble", C_US),
         ("varnet_single", "4DVarNet\n1 member", C_US)]
fig, ax = plt.subplots(figsize=(7.4, 4.0))
v = [r[k]["wall_mean"] for k, _, _ in names]
e = [r[k].get("wall_std", 0) for k, _, _ in names]
yy = np.arange(len(names))
ax.barh(yy, v, 0.58, xerr=e, color=[c for _, _, c in names], zorder=3,
        error_kw=dict(ecolor=INK2, lw=1.1, capsize=3))
for i, (a, b) in enumerate(zip(v, e)):
    ax.text(a + 6, i, f"{a:.1f} s  \u00b1{b:.2f}", va="center", fontsize=9.5, color=INK)
ax.set_yticks(yy); ax.set_yticklabels([n for _, n, _ in names]); ax.invert_yaxis()
ax.set_xlabel("wall-clock for 1,000 frames  (s)  \u2014 lower is better")
ax.set_xlim(0, max(v) * 1.28)
# routed below the value label, which sits just right of the bar at the same y
ax.annotate(f"{v[0] / v[2]:.1f}x faster than the EnKF", xy=(v[2] + 2, 2.28), xytext=(112, 2.62),
            fontsize=10, color=C_US, va="center",
            arrowprops=dict(arrowstyle="->", color=C_US, lw=1.2,
                            connectionstyle="arc3,rad=0.12"))
ax.set_title(f"Same node ({B['hw']['node']}), {B['hw']['cores_avail']} cores, "
             f"interleaved, {B['repeats']} timed repeats",
             fontsize=9.5, color=INK2, pad=8)
ax.grid(axis="x", color="#e6eaef", lw=0.8); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "mt_time.png"), dpi=200, facecolor="white")
plt.close(fig); print("  mt_time.png")

# ── 5. Sec. 2.4 decomposition — where the variance actually comes from ────────
CO = J("collapse.json")
D, M = CO["decomposition"], CO["members"]
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.5),
                         gridspec_kw={"width_ratios": [1.35, 1]})
ax = axes[0]
ale, epi = D["aleatoric_share_of_variance"], D["epistemic_share_of_variance"]
ax.barh([0], [ale * 100], 0.5, color=C_ALE, zorder=3)
ax.barh([0], [epi * 100], 0.5, left=[ale * 100 + 0.35], color=C_EPI, zorder=3)  # 2px gap
ax.text(ale * 50, 0, f"learnt $\\overline{{\\sigma^2_m}}$   {ale*100:.1f}%",
        ha="center", va="center", fontsize=11, color="white", fontweight="bold")
ax.annotate(f"5-member disagreement\n$\\mathrm{{Var}}_m(\\mu_m)$   {epi*100:.1f}%",
            xy=(99, 0.18), xytext=(72, 0.55), fontsize=9.5, color=INK,
            arrowprops=dict(arrowstyle="->", color=INK2, lw=1.1))
ax.set_yticks([]); ax.set_xlim(0, 108); ax.set_ylim(-0.55, 0.8)
ax.set_xlabel("share of the total predictive variance  (%)")
ax.spines["left"].set_visible(False)
ax.set_title("Sec. 2.4: the two halves", fontsize=10.5, color=INK, pad=8)

ax = axes[1]
lab = ["learnt\n$\\overline{\\sigma^2_m}$", "ensemble\n$\\mathrm{Var}_m(\\mu_m)$",
       "total"]
val = [D["aleatoric_spread_skill"], D["epistemic_spread_skill"], D["total_spread_skill"]]
ax.bar(range(3), val, 0.55, color=[C_ALE, C_EPI, C_US], zorder=3)
ax.axhline(1.0, ls="--", color=C_REF, lw=1.5, zorder=4)
ax.text(2.42, 1.09, "calibrated", fontsize=9, color=INK2, ha="right", va="bottom")
for i, v in enumerate(val):
    ax.text(i, v + 0.09, f"{v:.2f}", ha="center", fontsize=10, color=INK)
ax.set_xticks(range(3)); ax.set_xticklabels(lab, fontsize=9.5)
ax.set_ylabel("spread / skill"); ax.set_ylim(0, max(val) * 1.22)
ax.set_title("...and how each is scaled", fontsize=10.5, color=INK, pad=8)
finish(fig, axes, "mt_decomposition.png")

# ── 6. is sigma the right size? observed / unobserved / total ─────────────────
# Three cases the audience already understands -- the robots either measured that cell or they
# did not -- with the two magnitudes side by side in each. No ratio, no coverage curve: both say
# the same thing and both have to be defined before they can be quoted.
UU = J("uncertainty_vsb0.json")["results"]
grp = [("observed\nrobots measured it", "observed"),
       ("unobserved\nno measurement", "blind"),
       ("total\nevery cell", "all")]
said = [UU[k]["sigma_mean"] for _, k in grp]
act = [UU[k]["rmse"] for _, k in grp]
fig, ax = plt.subplots(figsize=(8.4, 3.9))
x = np.arange(len(grp)); w = 0.33; g = 0.012
ax.bar(x - w/2 - g, said, w, color=C_EPI, label="what the model says its error is", zorder=3)
ax.bar(x + w/2 + g, act, w, color=C_US, label="what its error actually is", zorder=3)
for xi, (p_, q_) in enumerate(zip(said, act)):
    ax.text(xi - w/2 - g, p_ + 0.011, f"{p_:.2f}", ha="center", fontsize=11.5, color=INK)
    ax.text(xi + w/2 + g, q_ + 0.011, f"{q_:.2f}", ha="center", fontsize=11.5, color=INK)
ax.set_xticks(x); ax.set_xticklabels([n for n, _ in grp], fontsize=11)
ax.set_ylabel("size of the error")
ax.set_ylim(0, max(said) * 1.34)
ax.legend(frameon=False, fontsize=10.5, loc="upper center",
          bbox_to_anchor=(0.5, 1.03), ncol=2, columnspacing=1.8)
finish(fig, ax, "mt_size.png")

print("[done]")
