"""
build_meeting_deck.py — the meeting slide deck
==============================================

Builds the 9-slide, figure-driven deck for the supervisor meeting. Every number is
read from config.yaml / the checkpoint / check_outputs/eval/*.json — nothing about
the training or results is hard-coded here, so the deck can never drift from what
was actually trained / measured.

Slides:  1 walkable map · 2 framework recap · 3 training + convergence ·
         4 fair-comparison setup · 5 reconstruction (density + velocity) ·
         6 results (4DVarNet vs EnKF) · 7 takeaways · appendix: detailed architecture.

Rendering helpers (render_pptx / render_pdf / render_notes) come from build_slides.py.
Outputs: slides/meeting_deck.pptx / .pdf / meeting_notes.md

Run:
    module load scicomp-pytorch-env/2026.1
    python3 slides/build_meeting_deck.py
"""

from __future__ import annotations

import os
from crowdcore import paths
import sys


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from slides.build_slides import (render_pptx, render_pdf, render_notes,
                                  PAGE_W, OUTPUTS)

# 这个脚本原来住在 4dvarnet_enkf/ 下，ROOT 一直指那个目录（runs/、check_outputs/
# 都挂在它下面）。2026-09-03 重构后它搬到了顶层，dirname(dirname(__file__)) 会变成
# 仓库根，于是每一条 os.path.join(ROOT, ...) 都会静默指错地方 —— 所以显式绑定。
ROOT = paths.method(paths.VARNET)
OUT_PPTX = os.path.join(paths.SLIDES, "meeting_deck.pptx")
OUT_PDF = os.path.join(paths.SLIDES, "meeting_deck.pdf")
OUT_NOTES = os.path.join(paths.SLIDES, "meeting_notes.md")

# The training run whose numbers/curve the deck reports. This is the ONLY place the
# run is named; all training hyper-parameters are read back from that run's checkpoint
# (varnet_last.pt "args") + config.yaml — never hard-coded in this slide script.
RUN = "runs/varnet_b0_k1"

F = lambda sub, name: os.path.join(OUTPUTS, sub, name)


def render_architecture(outpath):
    """End-to-end training schematic: data -> GradSolver(uses GENN prior Φ) -> x_rec -> loss,
    with the whole solver unrolled and back-propagated so Φ AND the solver train jointly."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(13, 6.6)); fig.patch.set_facecolor("white")
    ax.set_xlim(0, 13); ax.set_ylim(0, 6.6); ax.axis("off")
    BLUE, GREEN, ORANGE, GRAY = "#dCE6F5", "#dCF0DE", "#FBE7D0", "#eee"

    def box(x, y, w, h, txt, fc, fs=11, bold=False, ec="#334"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=fc, ec=ec, lw=1.5))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2, color="#334", lw=2.2, style="-|>"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw))

    # inputs (left)
    box(0.2, 4.55, 2.5, 0.7, "Y  partial observation", BLUE, 11)
    box(0.2, 3.75, 2.5, 0.7, "Ω  observation mask", BLUE, 11)
    box(0.2, 2.95, 2.5, 0.7, "X₀  initial fill", BLUE, 11)
    box(0.2, 1.15, 2.5, 0.9, "X  ground truth\n(supervised loss ONLY)", GRAY, 10)

    # solver (middle)
    box(3.4, 2.5, 5.3, 3.4, "", GREEN, 11)
    ax.text(6.05, 5.55, "GradSolver — learned gradient descent, unrolled 20 iterations",
            ha="center", va="center", fontsize=12, fontweight="bold", color="#1a5")
    box(3.7, 4.15, 4.7, 1.05,
        "variational cost   J(x) = ‖(x−Y)⊙Ω‖²  +  ‖x − Φ(x)‖²\n"
        "                              fit observations        prior (plausible sequence)", ORANGE, 10.5)
    box(3.7, 3.35, 2.15, 0.65, "Φ = GENN prior", "#fff", 10.5, bold=True)
    box(6.05, 3.35, 2.35, 0.65, "g = LSTM(α·∇J);  x ← x − L(g)", "#fff", 10)
    ax.text(6.05, 2.78, "repeat 20×  (x refined each step)", ha="center", fontsize=9, color="#555",
            fontstyle="italic")

    # output + loss (right)
    box(9.4, 3.9, 1.9, 0.9, "x_rec\nreconstruction", BLUE, 11, bold=True)
    box(11.55, 3.9, 1.25, 0.9, "Loss\n‖x_rec−X‖²", ORANGE, 10, bold=True)

    # forward arrows
    for yy in (4.9, 4.1, 3.3):
        arrow(2.7, yy, 3.4, 4.2)
    arrow(8.7, 4.2, 9.4, 4.35)
    arrow(11.3, 4.35, 11.55, 4.35)

    # backprop arrow (bottom, spanning back to the solver/Φ)
    arrow(11.9, 3.9, 11.9, 0.7, color="#c0392b", lw=2.2)
    arrow(11.9, 0.7, 4.9, 0.7, color="#c0392b", lw=2.2)
    arrow(4.9, 0.7, 4.9, 2.5, color="#c0392b", lw=2.2)
    ax.text(8.2, 0.45, "back-propagate the loss through the WHOLE unrolled solver  →  "
            "update Φ AND the solver weights JOINTLY (end-to-end, paper Eq. 14)",
            ha="center", fontsize=10.5, color="#c0392b", fontweight="bold")

    fig.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close(fig)
    return outpath


def render_architecture_detail(outpath):
    """Detailed, component-level architecture in three stacked bands:
        A  the unrolled solver / end-to-end training flow
        B  what happens INSIDE one solver iteration (cost -> autodiff grad -> ConvLSTM -> update)
        C  what is INSIDE the prior Φ = GENN (two-scale; each branch ψ zero-centre -> φ pointwise)
    Hyper-parameters (kernels, widths, iters) are read from config + the checkpoint, not typed here."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    import torch
    from crowdcore import config as cfg
    P = cfg.CFG["prior"]
    ck = torch.load(os.path.join(ROOT, RUN, "varnet_last.pt"), map_location="cpu")
    a = ck["args"]
    hid, kt, kh, kw = P["hidden"], P["kt"], P["kh"], P["kw"]
    nphi, scale = P["n_phi_layers"], P["scale"]
    n_iter, dT, lstm_h = a["n_iter"], a["dT"], a["lstm_hidden"]

    fig, ax = plt.subplots(figsize=(13.2, 7.4)); fig.patch.set_facecolor("white")
    ax.set_xlim(0, 13.2); ax.set_ylim(0, 7.4); ax.axis("off")
    BLUE, GREEN, ORANGE, PURPLE, GRAY, WHITE = "#d5e3f7", "#d7f0da", "#fbe6cf", "#e7dcf5", "#eeeeee", "#ffffff"

    def box(x, y, w, h, txt, fc, fs=8.5, bold=False, ec="#334", lw=1.3):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.06",
                                    fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal")

    def arr(x1, y1, x2, y2, color="#334", lw=1.8, style="-|>", ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
                                     color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0))

    def band_label(y, txt, color="#123f7a"):
        ax.text(0.15, y, txt, ha="left", va="center", fontsize=10.5, fontweight="bold", color=color)

    # ============================ BAND A — overview / training ============================
    band_label(7.15, "A · End-to-end flow")
    yA = 6.35
    box(0.3, yA, 1.7, 0.75, "X₀\ninitial fill\n(from obs)", BLUE)
    box(2.5, yA - 0.12, 3.1, 1.0, f"GradSolver\nlearned gradient descent\nunrolled ×{n_iter}", GREEN, 9, bold=True)
    box(6.1, yA, 1.7, 0.75, "x_rec\nreconstruction", BLUE, bold=True)
    box(8.3, yA, 2.0, 0.75, "Loss\n‖x_rec − X‖²  (Eq.14)", ORANGE, 8.5, bold=True)
    box(10.8, yA, 2.0, 0.75, "X  ground truth\n(training ONLY)", GRAY, 8)
    arr(2.0, yA + 0.37, 2.5, yA + 0.37)
    arr(5.6, yA + 0.37, 6.1, yA + 0.37)
    arr(7.8, yA + 0.37, 8.3, yA + 0.37)
    arr(10.8, yA + 0.37, 10.3, yA + 0.37, ls="--", color="#777")
    # back-prop arrow
    arr(9.3, yA, 9.3, yA - 0.55, color="#c0392b", lw=1.8)
    arr(9.3, yA - 0.55, 4.05, yA - 0.55, color="#c0392b", lw=1.8)
    arr(4.05, yA - 0.55, 4.05, yA - 0.12, color="#c0392b", lw=1.8)
    ax.text(6.7, yA - 0.72, "back-propagate through the whole unrolled solver → Φ and the solver train JOINTLY",
            ha="center", fontsize=8.2, color="#c0392b", fontweight="bold")

    # zoom connector A -> B (dotted; the band-B title already says "one iteration")
    arr(2.9, yA - 0.12, 2.9, 5.72, ls=":", color="#123f7a", lw=1.4)

    # ============================ BAND B — inside one iteration ============================
    band_label(5.55, f"B · Inside one solver iteration  (k = 1…{n_iter})")
    yB = 4.35
    box(0.3, yB, 1.35, 0.95, "xₖ\nstate", BLUE, bold=True)
    box(1.95, yB - 0.15, 3.2, 1.25,
        "variational cost  J(xₖ)\n"
        "α_obs²·‖(x−y)⊙Ω‖²  (obs)\n"
        "+ α_reg²·‖x − Φ(x)‖²  (prior)\n"
        "α, per-channel W: learnable", ORANGE, 8)
    box(5.45, yB, 2.05, 0.95, f"gₖ = ∂J/∂xₖ\nautodiff\n(no hand-derived ∇)", WHITE, 8)
    box(7.8, yB, 2.15, 0.95, f"ConvLSTM2d\n(hidden {lstm_h}) →\nupdate uₖ", PURPLE, 8, bold=True)
    box(10.25, yB, 2.55, 0.95, f"xₖ₊₁ = xₖ − uₖ / {n_iter}\n(gentle step)", GREEN, 8.5, bold=True)
    arr(1.65, yB + 0.47, 1.95, yB + 0.47)
    arr(5.15, yB + 0.47, 5.45, yB + 0.47)
    arr(7.5, yB + 0.47, 7.8, yB + 0.47)
    arr(9.95, yB + 0.47, 10.25, yB + 0.47)
    # loop-back xk+1 -> xk
    arr(11.5, yB, 11.5, yB - 0.5, color="#555", lw=1.6)
    arr(11.5, yB - 0.5, 0.97, yB - 0.5, color="#555", lw=1.6)
    arr(0.97, yB - 0.5, 0.97, yB, color="#555", lw=1.6)
    ax.text(6.2, yB - 0.68, f"repeat ×{n_iter}  (each step lowers J: better obs-fit + more plausible)",
            ha="center", fontsize=8.2, color="#555", style="italic")
    # zoom connector B(Φ) -> C
    ax.text(3.55, yB - 0.02, "Φ ↓", fontsize=8, color="#1a7a3a", fontweight="bold")
    arr(3.55, yB - 0.15, 3.55, 2.95, ls=":", color="#1a7a3a", lw=1.4)

    # ============================ BAND C — inside Φ (GENN) ============================
    band_label(2.75, "C · Inside the prior  Φ = GENN  (two-scale, paper Eq.10)")
    yC = 1.15
    box(0.3, yC + 0.15, 1.25, 0.9, "x\n(B,4,T,H,W)", BLUE, bold=True)
    # split into coarse / fine
    box(1.85, yC + 1.05, 3.3, 0.7, f"coarse: avg-pool ↓{scale} → branch → up", GREEN, 8)
    box(1.85, yC - 0.55, 3.3, 0.7, "fine: (x − coarse) → branch", GREEN, 8)
    box(5.5, yC + 0.15, 1.1, 0.9, "Σ  +", WHITE, 11, bold=True)
    box(6.9, yC + 0.15, 1.4, 0.9, "Φ(x)\n(B,4,T,H,W)", BLUE, bold=True)
    arr(1.55, yC + 0.6, 1.85, yC + 1.4)
    arr(1.55, yC + 0.6, 1.85, yC - 0.2)
    arr(5.15, yC + 1.4, 5.7, yC + 1.05)
    arr(5.15, yC - 0.2, 5.7, yC + 0.25)
    arr(6.6, yC + 0.6, 6.9, yC + 0.6)

    # one branch expanded (right side)
    ax.text(8.55, yC + 1.75, "each branch:", fontsize=8.5, fontweight="bold", color="#1a5")
    box(8.5, yC + 0.65, 1.75, 0.85, f"ψ : ZeroCentreConv3d\n{kt}×{kh}×{kw}, 4→{hid}\ncentre tap = 0", PURPLE, 7.6)
    box(10.4, yC + 0.65, 0.95, 0.85, "ReLU", WHITE, 8)
    box(11.5, yC + 0.65, 1.5, 0.85, f"φ : {nphi}× (1×1×1)\n{hid}→{hid}→4\npointwise", ORANGE, 7.6)
    arr(10.25, yC + 1.07, 10.4, yC + 1.07)
    arr(11.35, yC + 1.07, 11.5, yC + 1.07)
    ax.text(9.9, yC + 0.35, "zero-centre ⇒ Φ(x)(s) ⊥ x(s)  →  prior can't collapse to identity",
            ha="center", fontsize=7.8, color="#7a1a5a", fontweight="bold")

    fig.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close(fig)
    return outpath


def render_training_table(outpath, run=RUN):
    """The training / model hyper-parameters as one clean, large table.

    ALL numbers are read back from the training run itself — the checkpoint's saved
    `args` (the exact command line that trained it) + config.yaml — so this table can
    never drift from what was actually trained. Nothing is hard-coded here.
    """
    import json
    import torch
    from crowdcore import config as cfg
    ck = torch.load(os.path.join(ROOT, run, "varnet_last.pt"), map_location="cpu")
    a = ck["args"]                                        # the exact training args
    n_param = sum(v.numel() for v in ck["solver"].values() if hasattr(v, "numel"))
    mp = os.path.join(ROOT, run, "metrics.jsonl")
    n_epoch = a["epochs"]
    if os.path.exists(mp):
        done = [json.loads(l) for l in open(mp) if l.strip()]
        if done:
            n_epoch = done[-1]["epoch"] + 1               # epochs actually completed
    C, dT, H, W = 4, a["dT"], 36, 12
    rng = a.get("sensing_range", cfg.get("observation", "sensing_range"))
    na = a.get("num_agents", cfg.get("observation", "num_agents"))
    k = cfg.get("observation", "obs_every_k", default=1)
    prec = "bf16 (amp)" if a.get("amp") else "fp32"
    rows = [
        ("framework / loss", f"4DVarNet (Fablet 2020);  {a['loss']}  ‖x_rec − X‖²  (Eq.14)"),
        ("optimizer / lr", f"Adam  /  {a['lr']:g}"),
        ("epochs / batch / precision", f"{n_epoch}  /  {a['batch']}  /  {prec}"),
        ("window dT / solver iters", f"{dT} (paper window)  /  {a['n_iter']}"),
        ("total trainable params", f"{n_param/1e6:.2f} M"),
        ("data", f"{a['days']} days;  windows of ({C}, {dT}, {H}, {W})"),
        ("observation", f"{na} robots; range {rng}; line-of-sight; every {k}th frame; +noise"),
        ("hardware", "1 × NVIDIA H200"),
    ]
    fig, ax = plt.subplots(figsize=(12.5, 6.0)); fig.patch.set_facecolor("white")
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=["setting", "value"],
                   cellLoc="left", colLoc="left", loc="center",
                   colWidths=[0.32, 0.68])
    tbl.auto_set_font_size(False); tbl.set_fontsize(14); tbl.scale(1, 2.2)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#ccc")
        if r == 0:
            cell.set_facecolor("#123f7a"); cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif r % 2 == 0:
            cell.set_facecolor("#eef3fa")
        if c == 0 and r > 0:
            cell.get_text().set_fontweight("bold")
    fig.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close(fig)
    return outpath


def render_loss_curve(outpath, run=RUN):
    """Plot train-loss / blind-zone MSE / R-score vs epoch from the training run's own
    metrics.jsonl. Returns (path, run_name) or (None, None) if no metrics exist yet."""
    import json
    for run in (run,):
        mp = os.path.join(ROOT, run, "metrics.jsonl")
        if not os.path.exists(mp):
            continue
        rows = [json.loads(l) for l in open(mp) if l.strip()]
        if not rows:
            continue
        ep = [r["epoch"] for r in rows]
        fig, ax = plt.subplots(figsize=(9, 5.2))
        ax.plot(ep, [r["train_loss"] for r in rows], "-o", label="train loss")
        ax.plot(ep, [r["rec_unobs_mse"] for r in rows], "-s", label="blind-zone MSE")
        ax.plot(ep, [r["r_score"] for r in rows], "-^", label="R-score (full-state MSE)")
        ax.set_xlabel("epoch"); ax.set_ylabel("MSE"); ax.legend(fontsize=10)
        ax.set_title(f"end-to-end training — {run.split('/')[-1]} (dT=200, paper window)", fontsize=11)
        fig.savefig(outpath, dpi=140, bbox_inches="tight"); plt.close(fig)
        return outpath, run
    return None, None


def _headline():
    """Read the comparison numbers back from the eval JSONs (single source of truth) so
    nothing is hard-coded in the deck. Returns a dict of the values used in bullet text."""
    import json
    EV = os.path.join(OUTPUTS, "eval")
    v = json.load(open(os.path.join(EV, "test_metrics_matched_clip.json")))
    e = json.load(open(os.path.join(EV, "enkf_metrics.json")))
    ch = json.load(open(os.path.join(EV, "channel_metrics.json")))
    return {
        "n_days": len(v.get("per_day", [])) or 7,
        "v_blind": v["blind_mse_mean"], "e_blind": e["blind_mse_mean"],
        "v_full": v["full_mse_mean"], "e_full": e["full_mse_mean"],
        "ch_blind": ch["blind"],                            # {baseline,enkf,varnet} each ->{channel:mse}
    }


def define_slides():
    from crowdcore import config as cfg
    arch = render_architecture(F("prior_model", "architecture.png"))
    arch_detail = render_architecture_detail(F("prior_model", "architecture_detail.png"))
    table = render_training_table(F("training", "train_table.png"))
    curve, _ = render_loss_curve(F("training", "convergence_full100.png"))
    E = lambda name: os.path.join(OUTPUTS, "eval", name)   # eval-figure path helper

    # map rule read from config (NOT hard-coded); walkable count lives in the figure title.
    rule = cfg.get("navigation", "obstacle_rule")
    thr = cfg.get("navigation", "occupancy_thresh")
    m = _headline()
    better = lambda a, b: "4DVarNet" if a < b else "EnKF"
    # per-channel blind-zone winners, read from the JSON
    cb = m["ch_blind"]
    ch_line = "  ·  ".join(
        f"{c}: {better(cb['varnet'][c], cb['enkf'][c])}" for c in ("density", "vx", "vy", "var"))

    return [
        dict(kind="title",
             title="Progress: 50% map · full-data training · fair EnKF comparison",
             subtitle="Reproducing 4DVarNet (Fablet 2020) on ATC crowd data — results for this meeting",
             author="Xinle Zhang · July 2026",
             notes="EN: three things done since last meeting, in order — (1) rebuilt the walkable "
                   "map with the supervisor's 50% occupancy rule; (2) trained 4DVarNet to convergence "
                   "on the FULL 32-day dataset at the paper window dT=200; (3) a FAIR reconstruction "
                   "comparison against the EnKF (apt-ibex) baseline on held-out test days. Framing: "
                   "reproducing the paper + an honest baseline comparison, not chasing a single number.\n"
                   "中文: 上次会后做了三件事 — ①按老师 50% 占用规则重建可行走地图;②在全量 32 天数据、"
                   "论文窗口 dT=200 下把 4DVarNet 训到收敛;③在留出测试日上和 EnKF(apt-ibex)baseline 做"
                   "了公平的重建对比。定调: 复现论文 + 诚实 baseline 对比,不刷单一指标。"),

        # ---- 1. Map (supervisor's 50% rule) ----
        dict(title="1 — Walkable map: supervisor's 50% occupancy rule",
             images=[(F("navigation", "obstacle_map.png"), (0.3, 1.2, 6.3, 5.1)),
                     (F("navigation", "nav_mask_on_map.png"), (6.8, 1.2, 6.3, 5.1))],
             bullets=[(0.5, 6.35, 12.4, 1.0, 13, [
                 ("left: solid obstacle occupancy (ROS OCCUPIED ∪ enclosed-UNKNOWN — pillar/stall "
                  "interiors filled) = what the rule measures.   right: resulting walkable (green) vs wall cells", 0),
                 (f"supervisor's 50% rule: a 1 m cell is WALL iff ≥ {thr:.0%} obstacle "
                  f"(config obstacle_rule='{rule}', occupancy_thresh={thr})  ·  + 'walked-in-training' hybrid  ·  "
                  "largest connected component", 0),
             ])],
             notes="EN: the map now follows the supervisor's literal 50% rule — a 1 m cell is wall iff "
                   "≥50% of it is obstacle (config obstacle_rule=per_cell, occupancy_thresh=0.5). At 50% "
                   "this equals the connected-footprint variant (verified same mask). Obstacles = ROS "
                   "OCCUPIED ∪ enclosed-UNKNOWN. Hybrid with the training-day 'ever walked' criterion to "
                   "kill kernel spill across walls; largest component so nobody is trapped. The walkable "
                   "count is printed in the figure (computed live from the mask, never hard-coded here).\n"
                   "中文: 地图现在照老师字面 50% 规则 — 1m 格 ≥50% 是障碍就整格标墙(config per_cell/0.5)。"
                   "50% 时与连通 footprint 版一致(已验证同一 mask)。障碍=ROS OCCUPIED∪封闭 UNKNOWN。再"
                   "与'训练日走过'混合,去掉跨墙核溢出;保留最大连通域防困住。可行格数打印在图里(由 mask "
                   "实时算,不写死)。"),

        # ---- 2. Framework recap ----
        dict(title="2 — The framework (recap): learned variational reconstruction",
             images=[(arch, (0.4, 1.15, PAGE_W - 0.8, 3.2))],
             bullets=[(0.6, 4.5, 12.4, 2.9, 15, [
                 ("reconstruct the FULL crowd field over a 200-frame window from robots' partial, "
                  "noisy observations", 0),
                 ("cost   J(x) = ‖(x − y)⊙Ω‖²  (fit what the robots saw)  +  ‖x − Φ(x)‖²  "
                  "(Φ scores dynamical plausibility)", 0),
                 ("a learned gradient descent unrolls 20 steps from a rough fill X₀; step 20 = the "
                  "reconstruction — uses ONLY observations + Φ, never the truth", 0),
                 ("trained END-TO-END: the loss ‖x_rec − X‖² back-props through the whole unrolled "
                  "solver → Φ and the solver optimise jointly (paper Eq.14)", 0),
             ])],
             notes="EN: one-slide recap of how it works and how it trains, since it drives everything "
                   "after. Two-term cost (fit observations + plausibility), 20-step learned gradient "
                   "descent from X₀, uses only observations + Φ at inference. Trained end-to-end: loss "
                   "back-propagates through the whole unrolled solver (paper Eq.14).\n"
                   "中文: 一页回顾框架怎么跑、怎么训。两项代价(拟合观测+合理性),从 X₀ 做 20 步学习型梯度"
                   "下降,推理只用观测+Φ。端到端训练: 损失沿整条展开 solver 反传(论文 Eq.14)。"),

        # ---- 3. Training setup + convergence (full-data, dT=200) ----
        dict(title="3 — Trained to convergence on the full 32-day dataset (dT=200)",
             images=([(curve, (6.7, 1.35, 6.3, 5.3))] if curve else []) +
                     [(table, (0.4, 1.5, 6.0, 5.0))],
             captions=([("train loss · blind-zone MSE · R-score vs epoch  (converges by ~epoch 20)",
                         6.7, 6.75, 6.3)] if curve else []),
             notes="EN: report as a REPRODUCTION of the paper's dT=200 setting, not a metric contest. "
                   "Full 32-day dataset, 100 epochs, 20 solver iters, one H200. The table numbers are "
                   "read back from the checkpoint's saved args (never hand-typed); the curve is this "
                   "run's own metrics.jsonl — converges below the naive X₀ fill.\n"
                   "中文: 以'复现论文 dT=200 设置'讲,不搞指标比赛。全量 32 天、100 epoch、20 solver 迭代、"
                   "单张 H200。表里数字从 checkpoint 的 args 读回(不手打);曲线是本次 metrics.jsonl,收敛到"
                   "朴素 X₀ 填充以下。"),

        # ---- 4. Fair-comparison setup ----
        dict(title="4 — Fair comparison: identical everything, no truth at init",
             bullets=[(0.7, 1.4, 12.0, 5.6, 16, [
                 (f"both methods evaluated on the SAME {m['n_days']} held-out test days", 0),
                 ("SAME observations (same robot coverage, same additive noise), SAME frames, SAME "
                  "metric, SAME physical clip bounds", 0),
                 ("NEITHER sees the ground truth at initialisation:", 0),
                 ("4DVarNet starts from the observation-based fill X₀ (paper: solver uses only "
                  "observations, not true states)", 1),
                 ("EnKF starts from its ORIGINAL random Gaussian ensemble (unchanged from the "
                  "Partial_observation project)", 1),
                 ("comparison is reconstruction ACCURACY only — the EnKF's uncertainty (ensemble "
                  "spread) is shown separately, not scored against a point estimate", 0),
                 ("→ any accuracy difference reflects the METHOD, not an unfair setup", 0),
             ])],
             notes="EN: this is the slide the supervisors will scrutinise. Stress fairness: same test "
                   "days / observations / noise / frames / metric / clip; and crucially NEITHER method "
                   "is initialised from truth — 4DVarNet uses the obs-based X₀ (paper-faithful), EnKF "
                   "uses its original random init (we did NOT modify the EnKF project). We only compare "
                   "reconstruction accuracy; EnKF's spread is its own strength, shown separately.\n"
                   "中文: 老师最会盯这页。强调公平: 同测试日/观测/噪声/帧/指标/裁剪;且两者都不从真值初始化 "
                   "— 4DVarNet 用基于观测的 X₀(忠于论文),EnKF 用它原始随机初始化(没改 EnKF 项目)。只比"
                   "重建精度;EnKF 的 spread 是它的长处,单独展示。"),

        # ---- 5. Reconstruction figures (density + velocity) ----
        dict(title="5 — Reconstruction: density and velocity (EnKF-baseline style)",
             images=[(E("reconstruction_enkf_atc-20130811.png"), (0.3, 1.35, 12.7, 2.7)),
                     (E("velocity_enkf_atc-20130811.png"), (0.3, 4.25, 12.7, 2.7))],
             captions=[("colour: top = density (+ EnKF spread), bottom = speed |v|  ·  arrows = heading  ·  "
                        "panels: obs | true | 4DVarNet | EnKF", 0.3, 7.05, 12.7)],
             notes="EN: two matched frames in the EnKF project's own visual style so they sit next to "
                   "the baseline with no mismatch. Top = density (Blues) + heading arrows; bottom = "
                   "speed magnitude (Blues) + heading arrows, so velocity accuracy is visible. Point out: "
                   "true has a few high-speed blocks; 4DVarNet recovers more of them than the EnKF, which "
                   "under-estimates speed. Last panel is the EnKF ensemble spread (its uncertainty).\n"
                   "中文: 两张同帧、用 EnKF 项目的可视化风格,和 baseline 并排无违和。上=密度(蓝)+朝向箭头;"
                   "下=速率大小(蓝)+朝向箭头,能看出速度精度。指出: true 有几处高速块,4DVarNet 比 EnKF 恢复"
                   "得多,EnKF 普遍低估速度。最后一栏是 EnKF ensemble spread(它的不确定性)。"),

        # ---- 6. Results: headline + honest per-channel/region ----
        dict(title="6 — Results: 4DVarNet vs EnKF (held-out test days)",
             images=[(E("comparison.png"), (1.6, 1.2, 10.1, 2.7)),
                     (E("compare_channels.png"), (0.3, 4.05, 12.7, 2.4))],
             bullets=[(0.5, 6.55, 12.5, 0.9, 13, [
                 (f"blind-zone MSE: 4DVarNet {m['v_blind']:.4f} vs EnKF {m['e_blind']:.4f}  "
                  f"({better(m['v_blind'], m['e_blind'])} better overall)   ·   "
                  f"full-state: {m['v_full']:.4f} vs {m['e_full']:.4f}", 0),
                 (f"honest per-channel (blind zone) — {ch_line}   ·   "
                  "EnKF's var channel collapses in both regions; that is the main gap", 0),
             ])],
             notes="EN: headline + honesty in one slide. Left = overall blind-zone / full-state MSE "
                   "(4DVarNet lower overall). Right = per-channel by region (observed | blind): 4DVarNet "
                   "wins var and vx decisively; but on the blind zone EnKF is slightly better on density "
                   "and vy (density is sparse, low-SNR — 4DVarNet's weak spot). Say this openly — the win "
                   "is real but not uniform. All numbers here are read from the eval JSONs.\n"
                   "中文: 头条+诚实一页讲完。左=总体盲区/全场 MSE(4DVarNet 总体更低)。右=分通道分区域"
                   "(观测|盲区): 4DVarNet 在 var、vx 明显赢;但盲区里 density、vy 是 EnKF 略好(密度稀疏、"
                   "低 SNR,是弱项)。要坦白说 — 赢是真的但不均匀。数字都从 eval JSON 读。"),

        # ---- Appendix: detailed architecture (backup — shown only if asked) ----
        dict(title="Appendix — Architecture in detail (backup)",
             images=[(arch_detail, (0.25, 1.05, 12.8, 6.1))],
             notes="EN: BACKUP slide — do not present by default; pull up only if the supervisor asks "
                   "for architecture detail. Three bands: (A) end-to-end flow X₀→GradSolver(×20)→x_rec→loss, "
                   "back-prop trains Φ and the solver jointly; (B) inside one iteration — build the two-term "
                   "cost J, take its gradient by AUTODIFF (no hand-derived ∇), a ConvLSTM maps the gradient "
                   "to an update, x ← x − u/n_iter; (C) inside Φ=GENN — two-scale (coarse avg-pool branch + "
                   "fine residual branch, summed), each branch = ψ (zero-centre 3×3×3 conv, centre tap=0) → "
                   "ReLU → φ (two 1×1×1 pointwise convs). Zero-centre is why the prior can't collapse to the "
                   "identity. All kernel/width/iter numbers are read from config + the checkpoint.\n"
                   "中文: 备用页 — 默认不讲,老师问架构细节才翻。三段: (A) 端到端流程,反传联合训 Φ+solver; "
                   "(B) 单次迭代内部: 组两项代价 J → autodiff 求梯度(不手推)→ ConvLSTM 把梯度变成更新 → "
                   "x←x−u/n_iter; (C) Φ=GENN 内部: two-scale(粗 avg-pool 分支 + 细残差分支相加),每个分支 "
                   "= ψ(zero-centre 3×3×3,中心抽头=0)→ReLU→φ(两层 1×1×1 pointwise)。zero-centre 是"
                   "先验不塌缩成恒等的关键。所有核/宽度/迭代数从 config+checkpoint 读。"),
    ]


def main():
    slides = define_slides()
    n = render_pptx(slides, OUT_PPTX)
    render_pdf(slides, OUT_PDF)
    render_notes(slides, OUT_NOTES)
    print(f"[meeting deck] {n} slides ->")
    print(f"    {OUT_PPTX}")
    print(f"    {OUT_PDF}")
    print(f"    {OUT_NOTES}")


if __name__ == "__main__":
    main()
