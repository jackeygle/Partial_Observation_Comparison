"""
plot_results.py  —  the two result figures for the model presentation

  results_accuracy.png   RMSE on the held-out test set, 4DVarNet vs EnKF, at both densities
  results_speed.png      per-frame cost on ONE CPU, with the filter's cost split into the
                         two things it actually does

Why the speed figure is a stacked bar rather than a single number per method: the filter
pays for two separate operations every window — a forecast (the neural surrogate advanced
for all 100 ensemble members) and, on frames that carry an observation, a Kalman analysis.
Only the analysis scales with observation density, so lumping them into one bar hides the
reason the k=1 and k=4 bars differ. The split is measured, not apportioned: see
check_outputs/eval/enkf_time_split.json.

All numbers are read from the evaluation and benchmark jsons; nothing here is typed in.
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

# 4DVarNet: the model of record (runs/varnet_a2_k*) — RMSE per day first, then averaged,
# because sqrt of a mean is not the mean of sqrts.
# The model of record for the presentation: B0 = hidden 32 / kt 3, i.e. the
# config.yaml defaults. A2 (hidden 64, kt 5) scores ~2% better but is a capacity experiment;
# the deck presents the baseline configuration, so accuracy and speed must both come from it.
RUN = "b0"
VAR = {}
for k in (1, 4):
    d = J(f"test_metrics_{RUN}_k{k}.json")
    fu = np.array([r["full_mse"] for r in d["per_day"]], float)
    VAR[k] = dict(rmse=np.sqrt(fu).mean(), std=np.sqrt(fu).std(),
                  days=len(d["per_day"]),
                  frames=sum(r["n_windows"] for r in d["per_day"]) * d["dT"])
ENK = {k: J(f"test_metrics_enkf_k{k}.json") for k in (1, 4)}
SPD = J(f"bench_speed_{RUN}.json")
SPL = J("enkf_time_split.json")
ms = lambda k: SPD["runs"][k]["per_frame_s"] * 1000

INK, MUTED = "#20334d", "#5b6a7d"
C_VAR, C_FORE, C_ANAL = "#0e6b8a", "#e0a370", "#b5651d"

# ───────────────────────────────────────────────── accuracy
fig, ax = plt.subplots(figsize=(9.2, 4.9))
x = np.arange(2)
w = 0.34
v = [VAR[4]["rmse"], VAR[1]["rmse"]]
e = [ENK[4]["rmse_mean"], ENK[1]["rmse_mean"]]
ax.bar(x - w / 2, v, w, label="4DVarNet", color=C_VAR)
ax.bar(x + w / 2, e, w, label="EnKF", color=C_ANAL)
for xi, a, b in zip(x, v, e):
    ax.text(xi - w / 2, a + 0.004, f"{a:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.text(xi + w / 2, b + 0.004, f"{b:.3f}", ha="center", fontsize=11, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["observing every 4th frame", "observing every frame"], fontsize=11)
ax.set_ylabel("RMSE over the whole state  (lower is better)", fontsize=10.5)
ax.set_ylim(0, max(e) * 1.22)
ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=10.5, frameon=False)
ax.set_title("Reconstruction accuracy — held-out test days", fontsize=13, pad=12)
fig.text(0.012, 0.02,
         f"{VAR[1]['days']} test days (2013-08-11 to 2013-09-29), {VAR[1]['frames']:,} frames the "
         f"models never trained on. Same days, same frames, same physical bounds for both "
         f"methods; RMSE taken per day and then averaged. The EnKF runs its optimised "
         f"implementation, verified bit-identical to the original over all 14 full-day runs.",
         fontsize=7.6, color=MUTED, wrap=True)
fig.tight_layout(rect=[0, 0.07, 1, 1])
p = os.path.join(EV, "results_accuracy.png")
fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fig)
print(f"[figure] {p}")

# ───────────────────────────────────────────────── speed
fig, ax = plt.subplots(figsize=(10.6, 4.9))
rows, fore, anal, single = [], [], [], []
for k, lab in ((1, "EnKF\nevery frame observed"), (4, "EnKF\nevery 4th frame observed")):
    tot = ms(f"enkf_k{k}")
    sh = SPL["runs"][f"k{k}"]
    rows.append(lab)
    fore.append(tot * sh["forecast_share"])           # measured share, applied to this
    anal.append(tot * sh["analysis_share"])           # benchmark's total for the same setting
    single.append(0.0)
# Not "either density" — the state's first channel is literally called density, so that
# label reads as "crowd density" to anyone seeing this cold. Name the two settings.
rows.append("4DVarNet\nevery frame or every 4th")
fore.append(0.0); anal.append(0.0); single.append(ms("varnet_k1"))

y = np.arange(len(rows))[::-1]
ax.barh(y, fore, 0.55, color=C_FORE, label="EnKF — forecast (surrogate on 100 members)")
ax.barh(y, anal, 0.55, left=fore, color=C_ANAL, label="EnKF — Kalman analysis")
ax.barh(y, single, 0.55, color=C_VAR, label="4DVarNet — 20 unrolled solver iterations")
for yi, f_, a_, s_ in zip(y, fore, anal, single):
    if s_:
        ax.text(s_ + 3, yi, f"{s_:.1f} ms", va="center", fontsize=11, fontweight="bold")
    else:
        if f_ > 12:
            ax.text(f_ / 2, yi, f"{f_:.0f}", va="center", ha="center", fontsize=9.5, color="#3a2a12")
        ax.text(f_ + a_ / 2, yi, f"{a_:.0f}", va="center", ha="center", fontsize=9.5, color="white")
        ax.text(f_ + a_ + 3, yi, f"{f_ + a_:.1f} ms", va="center", fontsize=11, fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(rows, fontsize=10.5)
ax.set_xlabel("wall-clock per reconstructed frame  (ms, lower is better)", fontsize=10.5)
ax.set_xlim(0, max(np.array(fore) + np.array(anal)) * 1.22)
ax.grid(axis="x", alpha=0.3)
ax.legend(fontsize=9, frameon=False, loc="lower right")
ax.set_title(f"Inference cost — one node, {SPD['hw']['cores_avail']} cores, "
             f"{SPD['frames']} frames, {SPD['repeats']} rounds", fontsize=13, pad=12)
# Run-to-run spread belongs on the figure. The nodes are shared, so the same unchanged EnKF has
# measured anywhere from 150 to 171 ms/frame across sessions purely from co-tenant load. Within
# ONE json the comparison is still sound (all methods interleaved on one node), but a reader
# should not take the mean for a constant.
spread = max(r["wall_std"] / r["wall_mean"] for r in SPD["runs"].values())
fig.text(0.012, 0.02,
         f"{SPD['hw']['cpu']} · {SPD['hw']['cores_avail']} cores"
         f"{', node ' + SPD['hw']['node'] if SPD['hw'].get('node') else ''}. Both methods on the "
         f"same CPU, in one job, interleaved: no GPU is involved, and the EnKF is its optimised "
         f"implementation. Only the analysis step scales with observation density — the forecast "
         f"runs on every frame either way — which is why the two EnKF bars differ while "
         f"4DVarNet's cost does not ({ms('varnet_k1'):.2f} vs {ms('varnet_k4'):.2f} ms). Split "
         f"measured separately under the same conditions. Largest run-to-run spread across the "
         f"{SPD['repeats']} rounds: {spread:.0%} of the mean — the node was shared, so read these "
         f"as ratios between methods rather than as absolute constants.",
         fontsize=7.6, color=MUTED, wrap=True)
fig.tight_layout(rect=[0, 0.09, 1, 1])
p = os.path.join(EV, "results_speed.png")
fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fig)
print(f"[figure] {p}")

print(f"\n  accuracy  4DVarNet {VAR[4]['rmse']:.4f} / {VAR[1]['rmse']:.4f}   "
      f"EnKF {ENK[4]['rmse_mean']:.4f} / {ENK[1]['rmse_mean']:.4f}")
print(f"  speed     4DVarNet {ms('varnet_k1'):.2f} ms   "
      f"EnKF k=1 {ms('enkf_k1'):.1f} ({SPL['runs']['k1']['forecast_share']:.0%} forecast) "
      f"k=4 {ms('enkf_k4'):.1f} ({SPL['runs']['k4']['forecast_share']:.0%} forecast)")
