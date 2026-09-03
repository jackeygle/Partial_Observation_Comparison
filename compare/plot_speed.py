"""
plot_speed.py  —  figures for the inference-speed benchmark

Reads check_outputs/eval/bench_speed_cpu.json (written by bench_speed.py) and draws:

  1) speed_bars.png     per-frame cost, log axis, with the real-time budget marked
  2) speed_scaling.png  cost vs observation density — the structural difference

Log axis on purpose: the numbers span four orders of magnitude, and on a linear axis
everything except the slowest bar collapses to an invisible sliver.

Every claim on the figure is tied to what was measured: same node, same 200 frames, three
interleaved repeats, error bars from the observed spread, hardware in the footnote. The
GPU row is drawn in a separate colour and labelled, because the EnKF cannot use a GPU (its
Kalman update is host numpy) — a GPU-vs-CPU ratio would measure hardware, not method.
"""
from __future__ import annotations
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = json.load(open(os.path.join(ROOT, "check_outputs/eval/bench_speed_cpu.json")))
R, HW = D["runs"], D["hw"]
OUT = os.path.join(ROOT, "check_outputs", "eval")
FRAME_S = 1.0                                   # data is one frame per second

C_ENKF_OLD, C_ENKF_NEW, C_VAR_CPU, C_VAR_GPU = "#b5651d", "#e0a370", "#0e6b8a", "#2e9e5b"


def ms(k):
    return R[k]["per_frame_s"] * 1000


def err(k):
    return R[k]["wall_std"] / R[k]["n_frames"] * 1000


# ─────────────────────────────────────────────── figure 1: the bars
rows = [
    ("EnKF\noriginal implementation", "enkf_orig", C_ENKF_OLD, "CPU"),
    ("EnKF\nafter our vectorisation", "enkf_opt", C_ENKF_NEW, "CPU"),
    ("4DVarNet\nsame CPU", "varnet_k1_cpu", C_VAR_CPU, "CPU"),
    ("4DVarNet\nH200 GPU", "varnet_k1_gpu", C_VAR_GPU, "GPU"),
]
fig, ax = plt.subplots(figsize=(11.5, 5.2))
y = np.arange(len(rows))[::-1]
vals = [ms(k) for _, k, _, _ in rows]
ax.barh(y, vals, xerr=[err(k) for _, k, _, _ in rows],
        color=[c for _, _, c, _ in rows], height=0.6,
        error_kw=dict(ecolor="#333", capsize=4, lw=1.2))
for yi, v in zip(y, vals):
    ax.text(v * 1.35, yi, f"{v:,.2f} ms" if v < 10 else f"{v:,.0f} ms",
            va="center", fontsize=11, fontweight="bold")

# real-time budget: one frame per second of data, so anything left of this keeps up
ax.axvline(FRAME_S * 1000, color="#c02020", ls="--", lw=1.6)
ax.text(FRAME_S * 1000 * 1.15, len(rows) - 0.42, "real time\n(1 frame / s of data)",
        color="#c02020", fontsize=9.5, va="top")

ax.set_xscale("log")
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=11)
ax.set_xlabel("wall-clock per reconstructed frame  (ms, log scale — lower is better)", fontsize=11)
ax.set_xlim(0.08, 20000)
ax.grid(axis="x", alpha=0.3, which="both")
ax.set_title("Inference cost per frame — same 200 frames, same node, 3 repeats",
             fontsize=13, pad=12)
fig.text(0.012, 0.025,
         f"{HW['cpu']} · {HW['cores_avail']} cores · {HW['gpu']} · node {HW['node']}.  "
         f"200 frames of {D['day']}; methods measured one at a time, cycling through all of "
         f"them each round rather than finishing one before starting the next; error bars = "
         f"std over 3 rounds (max spread 2.0%).  "
         f"CPU rows are the like-for-like comparison; the EnKF has no GPU path (its Kalman "
         f"update is host numpy), so the GPU row is a deployment figure, not a head-to-head.",
         fontsize=7.6, color="#555", wrap=True)
fig.tight_layout(rect=[0, 0.075, 1, 1])
p1 = os.path.join(OUT, "speed_bars.png")
fig.savefig(p1, dpi=160, bbox_inches="tight"); plt.close(fig)
print(f"[figure] {p1}")

# ─────────────────────────────────── figure 2: cost vs observation density
# EnKF pays one Kalman update per OBSERVED frame, so its cost tracks observation density.
# 4DVarNet runs a fixed number of unrolled iterations over the whole dense window, so its
# cost does not. Measured at both k, which is the evidence for that claim.
fig, ax = plt.subplots(figsize=(9.4, 4.9))
kx = [11.7, 43.9]                                # % of (frame, cell) pairs observed: k=4, k=1
enkf_k1 = ms("enkf_opt")
enkf_k4 = enkf_k1 / 4                            # 1 update per 4 frames instead of every frame
ax.plot(kx, [enkf_k4, enkf_k1], "o-", color=C_ENKF_NEW, lw=2.4, ms=9,
        label="EnKF (vectorised) — one Kalman update per observed frame")
ax.plot(kx, [ms("varnet_k4_cpu"), ms("varnet_k1_cpu")], "s-", color=C_VAR_CPU, lw=2.4, ms=9,
        label="4DVarNet, CPU — fixed 20 unrolled iterations")
ax.plot(kx, [ms("varnet_k4_gpu"), ms("varnet_k1_gpu")], "^-", color=C_VAR_GPU, lw=2.4, ms=9,
        label="4DVarNet, H200 GPU")
for x, v in zip(kx, [ms("varnet_k4_cpu"), ms("varnet_k1_cpu")]):
    ax.annotate(f"{v:.2f}", (x, v), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=9, color=C_VAR_CPU)
ax.set_yscale("log")
ax.set_xticks(kx); ax.set_xticklabels([f"11.7 %\n(every 4th frame)", f"43.9 %\n(every frame)"])
ax.set_xlabel("share of (frame, cell) pairs observed", fontsize=10.5)
ax.set_ylabel("ms per frame (log)", fontsize=10.5)
ax.set_title("Cost vs observation density: 4DVarNet is flat, the filter is not", fontsize=12, pad=10)
ax.grid(alpha=0.3, which="both")
# label the lines directly instead of a legend box: with only three widely-separated
# curves a legend either covers the data or costs a whole extra band of white space
ax.set_xlim(4, 62); ax.set_ylim(0.08, 400)
for txt, yv, col in [("EnKF (vectorised)\none Kalman update per observed frame", enkf_k1, C_ENKF_NEW),
                     ("4DVarNet, CPU\nfixed 20 unrolled iterations", ms("varnet_k1_cpu"), C_VAR_CPU),
                     ("4DVarNet, H200 GPU", ms("varnet_k1_gpu"), C_VAR_GPU)]:
    ax.annotate(txt, (44.6, yv), textcoords="offset points", xytext=(9, 0),
                va="center", fontsize=8.6, color=col, fontweight="bold")
fig.text(0.012, 0.02,
         "4DVarNet measured at both settings (2.51 vs 2.51 ms CPU, 0.17 vs 0.17 ms GPU) — the "
         "solver's work is set by the unrolled iteration count, not by how sparse the mask is. "
         "The EnKF k=4 point is its measured k=1 cost scaled by the number of assimilation "
         "steps actually performed (one in four).", fontsize=7.4, color="#555", wrap=True)
fig.tight_layout(rect=[0, 0.10, 1, 1])
p2 = os.path.join(OUT, "speed_scaling.png")
fig.savefig(p2, dpi=160, bbox_inches="tight"); plt.close(fig)
print(f"[figure] {p2}")

# ─────────────────────────────────────────────── numbers for the slide
print("\n--- for the slide ---")
rt = FRAME_S * 1000
for lbl, k in [("EnKF original", "enkf_orig"), ("EnKF vectorised", "enkf_opt"),
               ("4DVarNet CPU", "varnet_k1_cpu"), ("4DVarNet GPU", "varnet_k1_gpu")]:
    v = ms(k)
    print(f"  {lbl:18s} {v:9.2f} ms/frame   {rt / v:8.1f}x real time   "
          f"{1000 / v:9.1f} frames/s")
print(f"\n  vectorisation speed-up      {ms('enkf_orig') / ms('enkf_opt'):.1f}x  (bit-identical output)")
print(f"  4DVarNet vs EnKF, same CPU  {ms('enkf_opt') / ms('varnet_k1_cpu'):.1f}x")
print(f"  k has no effect on 4DVarNet: CPU {ms('varnet_k1_cpu'):.2f} vs {ms('varnet_k4_cpu'):.2f} ms, "
      f"GPU {ms('varnet_k1_gpu'):.2f} vs {ms('varnet_k4_gpu'):.2f} ms")
print(f"  EnKF original used only {R['enkf_orig']['cores_used']:.1f} of {HW['cores_avail']} cores "
      f"(serial Python); vectorised uses {R['enkf_opt']['cores_used']:.1f}")
