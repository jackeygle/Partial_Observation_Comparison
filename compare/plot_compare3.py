"""
plot_compare3.py — 三方逐通道对比图
====================================

读 `check_outputs/eval/compare3.json`，画盲区 MSE 的三方对比。

**为什么是小多组而不是分组柱状图**：四个通道量级差一个数量级（vx 0.079、var 0.006）。
放在同一根 y 轴上，density/vy/var 三个面板会被压成看不见的细条，读者只能看出 vx 的
差异——而那恰好是唯一 4DVarNet 领先的通道，图会给出与数据相反的印象。
每通道一个面板、各自 y 轴，比的是"该通道内谁赢"，这正是要传达的信息。
面板间不可比这一点写在副标题里。

配色取自数据可视化规范的默认分类色板 1/2/3 号槽位（固定顺序，不循环）。

用法: python3 checks/plot_compare3.py    (纯 matplotlib, 登录节点即可)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 分类色板槽位 1/2/3（light mode）
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8a85"
SURFACE = "#fcfcfb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="check_outputs/eval/compare3.json")
    ap.add_argument("--out", default="check_outputs/eval/compare3.png")
    args = ap.parse_args()

    d = json.load(open(args.json))
    chans = d["channels"]
    methods = list(d["results"].keys())
    R = d["results"]

    # 各通道占「盲区总误差」的比重，按 Senseiver 算（用于说明 vx 的主导地位）
    ref = R[methods[0]]["per_channel"]
    tot = sum(ref.values())
    share = {c: ref[c] / tot * 100 for c in chans}

    panels = chans + ["overall"]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)

    for ax, p in zip(axes, panels):
        vals = [R[m]["overall"] if p == "overall" else R[m]["per_channel"][p] for m in methods]
        bars = ax.bar(range(len(methods)), vals, color=COLORS, width=0.62, zorder=3)
        best = min(range(len(vals)), key=lambda i: vals[i])
        for i, (b, v) in enumerate(zip(bars, vals)):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom",
                    fontsize=8.5, color=INK if i == best else INK2,
                    fontweight="bold" if i == best else "normal")
        ax.set_facecolor(SURFACE)
        ax.set_ylim(0, max(vals) * 1.28)
        ax.set_xticks([])
        sub = "unweighted mean" if p == "overall" else f"{share[p]:.0f}% of blind-zone error"
        ax.set_title(p, fontsize=11, color=INK, pad=13, fontweight="bold")
        ax.text(0.5, 1.015, sub, transform=ax.transAxes, ha="center",
                fontsize=8, color=MUTED)
        # 刻度：固定 3 位小数 + 限制刻度数，否则两位小数会撞出重复标签(0.04 出现两次)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, min_n_ticks=4))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
        ax.grid(axis="y", color="#e6e5e1", lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color("#d8d7d2")
        ax.tick_params(colors=MUTED, labelsize=8, length=0)

    axes[0].set_ylabel("blind-zone MSE  (lower is better)", fontsize=9.5, color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
    fig.legend(handles, methods, loc="lower center", ncol=len(methods), frameon=False,
               fontsize=9.5, labelcolor=INK2, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Blind-zone reconstruction error, per channel", fontsize=13.5, color=INK,
                 y=0.985, fontweight="bold")
    fig.text(0.5, 0.925, f"{d['n_days']} held-out days, {d['protocol']}  |  "
             f"each panel has its own y-axis - panels are not comparable to each other",
             ha="center", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.05, 1, 0.90])
    fig.savefig(args.out, facecolor=SURFACE, bbox_inches="tight")
    print(f"[out] {args.out}")

    print("\n每通道的赢家:")
    for p in panels:
        vals = {m: (R[m]["overall"] if p == "overall" else R[m]["per_channel"][p]) for m in methods}
        w = min(vals, key=vals.get)
        srt = sorted(vals.values())
        print(f"  {p:<9} {w:<18} {srt[0]:.4f}  (次优 {srt[1]:.4f}, 领先 {(srt[1]-srt[0])/srt[1]*100:.1f}%)")


if __name__ == "__main__":
    main()
