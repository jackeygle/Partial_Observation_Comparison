"""
plot_uncertainty.py  —  the three figures for the uncertainty comparison

  unc_scores.png       accuracy and CRPS side by side: what each method gets right
  unc_reliability.png       the reliability diagram, ours vs EnKF vs the ideal diagonal
  unc_reliability_ours.png  the same, ours only, for the deck that omits the filter
  unc_spread_decay.png why the EnKF's ensemble cannot hold a spread

Every number is read from check_outputs/eval/*.json, so a slide and a figure cannot disagree.

Why CRPS and not NLL on the score figure: the EnKF's sigma is ~1% of its own error, which sends
the NLL's (x-mu)^2/(2 sigma^2) term to ~1e18. That is a real result -- it is reported in the
tables -- but it cannot be drawn next to a number of order 1. CRPS is a proper scoring rule
too, is bounded, and carries the units of the state, so it goes on the figure.
"""
from __future__ import annotations
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = os.path.join(ROOT, "check_outputs", "eval")
J = lambda n: json.load(open(os.path.join(EV, n)))

ML = J("uncertainty_ml5.json")["results"]
EK = {k: J(f"uncertainty_enkf_k{k}.json")["results"] for k in (1,)}   # k=4 dropped 2026-09-08

INK, MUTED = "#20334d", "#5b6a7d"
C_US, C_EK, C_IDEAL = "#0e6b8a", "#b5651d", "#8a8a8a"


def save(fig, name, pad=0.09):
    fig.tight_layout(rect=[0, pad, 1, 1])
    p = os.path.join(EV, name)
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[figure] {p}")


# ───────────────────────────────────────────── 1. accuracy + CRPS
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5))
for ax, key, lab in ((axes[0], "rmse", "RMSE  (reconstruction error)"),
                     (axes[1], "crps", "CRPS  (error AND calibration)")):
    v = [ML["all"][key], EK[1]["all"][key]]
    b = ax.bar(["4DVarNet\n+ deep ensemble", "EnKF"], v, 0.5, color=[C_US, C_EK])
    for r, x in zip(b, v):
        ax.text(r.get_x() + r.get_width() / 2, x + max(v) * 0.02, f"{x:.4f}",
                ha="center", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(v) * 1.22)
    ax.set_title(lab, fontsize=12, pad=10)
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.5, 0.90, f"{(v[1] - v[0]) / v[1] * 100:.0f}% better", transform=ax.transAxes,
            ha="center", fontsize=11, color=C_US, fontweight="bold")
fig.text(0.012, 0.02,
         f"7 held-out test days, identical observations and evaluation frames for both methods. "
         f"CRPS is a proper scoring rule in the units of the state: it penalises a wrong mean and "
         f"a wrong spread at once, so a method cannot score well by being merely accurate or "
         f"merely uncertain. NLL is reported in the tables but not drawn -- the EnKF's is ~1e18 "
         f"because its sigma approaches zero.",
         fontsize=7.6, color=MUTED, wrap=True)
save(fig, "unc_scores.png")


# ───────────────────────────────────────────── 2. reliability diagram
fig, ax = plt.subplots(figsize=(6.0, 5.6))
zs = sorted(int(z) for z in ML["all"]["coverage"])
ax.plot([0, 100], [0, 100], "--", color=C_IDEAL, lw=1.6, label="ideal (perfectly calibrated)")
for res, lab, col, mk in ((ML["all"], "4DVarNet + deep ensemble", C_US, "o"),
                          (EK[1]["all"], "EnKF", C_EK, "s")):
    y = [res["coverage"][str(z)] * 100 for z in zs]
    ax.plot(zs, y, mk + "-", color=col, lw=2, ms=6, label=lab)
ax.set_xlabel("stated confidence  (%)", fontsize=11)
ax.set_ylabel("truths actually inside the interval  (%)", fontsize=11)
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.set_xticks(zs); ax.set_yticks(range(0, 101, 20))
ax.grid(alpha=0.3)
ax.legend(fontsize=9.5, frameon=False, loc="upper left")
ax.set_title("Does the stated confidence hold up?", fontsize=13, pad=12)
ax.annotate(f"says 90%, delivers {EK[1]['all']['coverage']['90'] * 100:.1f}%",
            xy=(90, EK[1]["all"]["coverage"]["90"] * 100), xytext=(46, 13),
            fontsize=9.5, color=C_EK,
            arrowprops=dict(arrowstyle="->", color=C_EK, lw=1.2))
fig.text(0.012, 0.02,
         f"Protocol from Lakshminarayanan et al. App. A.2. A point on the diagonal means the "
         f"method's stated confidence is exactly right. Above it is conservative (intervals "
         f"wider than needed); below it is overconfident. The EnKF's curve is nearly flat: "
         f"widening its interval 9x moves coverage only from "
         f"{EK[1]['all']['coverage']['10'] * 100:.2f}% to "
         f"{EK[1]['all']['coverage']['90'] * 100:.2f}%, so its spread barely tracks its error "
         f"and no rescaling would fix it.",
         fontsize=7.6, color=MUTED, wrap=True)
save(fig, "unc_reliability.png", pad=0.13)


# ───────────────────────────────────────────── 2b. the same diagram, ours only
# The three-way deck no longer describes the filter, so it needs a version of this figure with
# only our own curve. The full two-curve version above is kept for the decks that do compare.
fig, ax = plt.subplots(figsize=(6.0, 5.6))
ax.plot([0, 100], [0, 100], "--", color=C_IDEAL, lw=1.6, label="ideal (perfectly calibrated)")
_y = [ML["all"]["coverage"][str(z)] * 100 for z in zs]
ax.plot(zs, _y, "o-", color=C_US, lw=2, ms=6, label="4DVarNet + deep ensemble")
ax.fill_between(zs, zs, _y, color=C_US, alpha=0.10)
ax.set_xlabel("stated confidence  (%)", fontsize=11)
ax.set_ylabel("truths actually inside the interval  (%)", fontsize=11)
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.set_xticks(zs); ax.set_yticks(range(0, 101, 20))
ax.grid(alpha=0.3)
ax.legend(fontsize=9.5, frameon=False, loc="upper left")
ax.set_title("Does the stated confidence hold up?", fontsize=13, pad=12)
ax.annotate(f"says 90%, delivers {ML['all']['coverage']['90'] * 100:.1f}%",
            xy=(90, ML["all"]["coverage"]["90"] * 100), xytext=(44, 62),
            fontsize=9.5, color=C_US,
            arrowprops=dict(arrowstyle="->", color=C_US, lw=1.2))
fig.text(0.012, 0.02,
         f"Protocol from Lakshminarayanan et al. App. A.2. A point on the diagonal means the "
         f"stated confidence is exactly right. Above it the intervals are wider than they need "
         f"to be, which is the harmless direction; below it the model is overconfident. Our "
         f"curve sits above the diagonal at every level, so the stated intervals hold up. Note "
         f"that the RMS spread/skill ratio is {ML['all']['spread_skill']:.2f}, i.e. below 1 — "
         f"coverage and spread/skill disagree because the shortfall is in the tail: most errors "
         f"are comfortably inside the interval while a few large ones are not.",
         fontsize=7.6, color=MUTED, wrap=True)
save(fig, "unc_reliability_ours.png", pad=0.13)


# ───────────────────────────────────────────── 3. why the EnKF ensemble collapses
G = J("enkf_spread_growth.json")
fig, ax = plt.subplots(figsize=(7.6, 4.6))
st = [d["step"] for d in G["steps"]]
sp = [d["spread"] for d in G["steps"]]
ax.semilogy(st, sp, "o-", color=C_EK, lw=2, ms=4)
ax.axhline(G["injected_per_step"], ls="--", color=C_IDEAL, lw=1.4)
ax.text(len(st) * 0.55, G["injected_per_step"] * 1.5,
        f"noise injected per step  ({G['injected_per_step']:.4f})", fontsize=9, color=MUTED)
ax.set_xlabel("forecast steps  (no noise, no analysis)", fontsize=11)
ax.set_ylabel("ensemble spread", fontsize=11)
ax.grid(alpha=0.3, which="both")
ax.set_title("The EnKF's forecast model erases member disagreement", fontsize=13, pad=12)
one = G["steps"][1]["ratio_to_start"]
ax.annotate(f"one step removes {100 * (1 - one):.0f}%", xy=(1, sp[1]), xytext=(4.5, sp[0] * 0.6),
            fontsize=10, color=C_EK, arrowprops=dict(arrowstyle="->", color=C_EK, lw=1.2))
fig.text(0.012, 0.02,
         f"An ensemble of {G['ensemble']} members is given a spread of {sp[0]:.3f} -- the full "
         f"configured process noise, 100x what the filter actually injects -- and then advanced "
         f"through the forecast model alone: no process noise, no Kalman analysis, no inflation. "
         f"The spread decays to {sp[-1]:.1e} in {st[-1]} steps. The model is contractive, so the "
         f"collapse is not a matter of retuning: raising the injected noise or the inflation "
         f"would buy spread only by degrading the forecast.",
         fontsize=7.6, color=MUTED, wrap=True)
save(fig, "unc_spread_decay.png", pad=0.15)

print(f"\n  ML-5   RMSE {ML['all']['rmse']:.4f}  CRPS {ML['all']['crps']:.4f}  "
      f"sp/sk {ML['all']['spread_skill']:.2f}  90% {ML['all']['coverage']['90']:.1%}")
print(f"  EnKF   RMSE {EK[1]['all']['rmse']:.4f}  CRPS {EK[1]['all']['crps']:.4f}  "
      f"sp/sk {EK[1]['all']['spread_skill']:.3f}  90% {EK[1]['all']['coverage']['90']:.2%}")
