"""
plot_uncertainty.py  —  why the EnKF's ensemble cannot hold a spread

  unc_spread_decay.png   ensemble spread under the forecast model alone, step by step

Every number is read from check_outputs/eval/enkf_spread_growth.json.

This file used to draw three more figures -- CRPS/RMSE bars, a reliability diagram, and an
ours-only reliability diagram -- comparing the EnKF against `uncertainty_ml5.json`, the retired
variance-head design. They were deleted on 2026-09-13: ml5 is in no table (SUPERSEDED.md says
why), so redrawing them would put a superseded model back into a figure. The uncertainty
comparison that is reported is the table in the repository README, built from
uncertainty_{vsb0,aug0}.json, uncertainty_enkf_k1.json and uncertainty_dincae.json.
"""
from __future__ import annotations
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = os.path.join(ROOT, "check_outputs", "eval")
J = lambda n: json.load(open(os.path.join(EV, n)))

INK, MUTED = "#20334d", "#5b6a7d"
C_EK, C_IDEAL = "#b5651d", "#8a8a8a"


def save(fig, name, pad=0.09):
    fig.tight_layout(rect=[0, pad, 1, 1])
    p = os.path.join(EV, name)
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[figure] {p}")


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
