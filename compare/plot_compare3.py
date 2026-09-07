"""
plot_compare3.py — three-way per-channel comparison figure
====================================

Reads `check_outputs/eval/compare3.json`, plots the three-way comparison of blind
MSE.

**Why small multiples rather than a grouped bar chart**: the four channels differ
by an order of magnitude (vx 0.079, var 0.006). On a shared y-axis, the density/vy/var
panels would flatten into invisible slivers, leaving the reader only able to see
the difference in vx -- which happens to be the one channel where 4DVarNet leads,
giving an impression opposite to the data. One panel per channel with its own
y-axis compares "who wins within this channel," which is exactly the information
to convey. That panels are not comparable to each other is stated in the subtitle.

Colours are the default categorical palette's slots 1/2/3 from a dataviz spec
(fixed order, no cycling).

Usage: python3 checks/plot_compare3.py    (pure matplotlib, the login node is fine)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical palette slots 1/2/3 (light mode)
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

    # Each channel's share of "total blind error", computed from Senseiver (used
    # to explain vx's dominance)
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
        # Ticks: fixed 3 decimals + a bounded tick count, otherwise 2 decimals
        # produce duplicate labels (0.04 showing up twice)
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

    print("\nWinner per channel:")
    for p in panels:
        vals = {m: (R[m]["overall"] if p == "overall" else R[m]["per_channel"][p]) for m in methods}
        w = min(vals, key=vals.get)
        srt = sorted(vals.values())
        print(f"  {p:<9} {w:<18} {srt[0]:.4f}  (runner-up {srt[1]:.4f}, lead {(srt[1]-srt[0])/srt[1]*100:.1f}%)")


if __name__ == "__main__":
    main()
