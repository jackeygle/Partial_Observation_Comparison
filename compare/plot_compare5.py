"""
plot_compare5.py — the main results figure: 5 methods x 4 channels x 2 conventions

Reads `compare/results/compare5.json`, draws one figure telling the whole main
result, plus the fact that "the ranking flips".

**Everything is a single model, no ensembling** -- the paper never mentions
ensembling. The two 4DVarNet arms each have 5 training seeds; bar height is the
cross-seed mean, error bars are +/-1 std: "single model" is not one number,
reporting the best seed would be systematically over-optimistic, reporting s0
would be arbitrary. The other three methods each have only one model, no error
bars.

Layout: **2 rows x 5 columns of small multiples**

    rows = convention   top: channel-defined cells ∩ walkable ∩ blind (comparable across all five)
                         bottom: all cells ∩ blind (the old convention, for reference)
    columns = channel   density / vx / vy / var / total

Why small multiples rather than a grouped bar chart: the four channels differ by
an order of magnitude (vx 0.50, var 0.04). On a shared y-axis, the density/vy/var
panels would flatten into invisible slivers, leaving the reader only able to see
the difference in vx -- which happens to be the one channel where 4DVarNet leads,
giving an impression opposite to the data. One panel per channel with its own
y-axis compares "who wins within this channel," which is exactly the information
to convey. That panels are not comparable to each other is stated in the subtitle.

Why the two rows must be in one figure: **the ranking flips with the convention**
(top row DINCAE first, bottom row it comes last, 12x worse). Splitting into two
figures in different sections would leave the reader believing "the ranking
flips" only because the text says so; side by side, the flip is visible.

The y-axis is always linear, auto-scaled per panel, never log, never truncated. In
the bottom row's vy panel DINCAE is 1.65 while the rest are 0.02, so the other four
bars sit nearly flat against the axis -- that is simply the fact, and every bar has
its value labelled on top, so no information is lost. Switching to log to make the
small bars look nicer would soften the "80x difference" conclusion.

Usage (the login node is fine, pure matplotlib):
    source sbatch/_env.sh
    python3 -m compare.plot_compare5
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from compare import plotstyle as ps
from crowdcore import paths

#: The main table plots these five methods, **all single models** -- the paper
#: never mentions ensembling.
#: The two 4DVarNet arms each have 5 seeds; take the cross-seed mean and draw
#: +/-std error bars: "single model" is not one number, reporting the best seed
#: would be systematically over-optimistic, reporting s0 would be arbitrary.
#:
#: Triple = (how to read it from compare5.json, legend label, key used for colour)
#:   ("row",  row-name prefix)   read directly from results[...]
#:   ("seed", arm label)         read mean+/-std from seed_summary[...]
ROWS = [
    ("row",  "DINCAE",     "DINCAE",                      "DINCAE"),
    ("seed", "MSE",        "4DVarNet (MSE loss)",         "4DVarNet MSE"),
    ("seed", "NLL",        "4DVarNet + uncertainty head", "4DVarNet NLL"),
    ("row",  "Senseiver",  "Senseiver",                   "Senseiver"),
    ("row",  "EnKF k1",    "EnKF",                        "EnKF k1"),
]

PANELS = [
    ("defined",  "channel-defined cells\n(blind, walkable)"),
    ("allcells", "all cells\n(blind, old convention)"),
]


def pick(results: dict, prefix: str):
    """Fetch a row by prefix, tolerating member-count differences like ens5/ens2."""
    for k in results:
        if k.startswith(prefix):
            return k, results[k]
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(paths.COMPARE, "results", "compare5.json"))
    ap.add_argument("--out", default=os.path.join(paths.COMPARE, "results", "compare5.png"))
    args = ap.parse_args()

    with open(args.json) as f:
        doc = json.load(f)
    results, chans = doc["results"], doc["channels"]
    cols = chans + ["total"]

    seeds = doc.get("seed_summary", {})
    rows = []                      # (label, color, kind, payload)
    for kind, ref, label, ckey in ROWS:
        if kind == "seed":
            if ref not in seeds:
                print(f"  [warn] {args.json} has no {ref} in seed_summary, skipping this row")
                continue
            rows.append((label, ps.method_color(ckey), "seed", seeds[ref]))
        else:
            key, entry = pick(results, ref)
            if entry is None:
                print(f"  [warn] {args.json} has no {ref}*, skipping this row")
                continue
            rows.append((label, ps.method_color(key), "row", entry))

    ps.use()
    # wspace given generously: each panel has its own y-axis, and the tick labels
    # need horizontal room, or they collide with the neighbouring panel's bars.
    # The first version didn't set this, and vy's "0.150" sat right on top of
    # vx's bars.
    fig, axes = ps.figure(ncols=len(cols), nrows=len(PANELS),
                          width=ps.TEXT_W * 1.75, rows_h=2.9,
                          gridspec_kw={"wspace": 0.52, "hspace": 0.22})
    axes = np.atleast_2d(axes)

    for r, (conv, conv_title) in enumerate(PANELS):
        for c, ch in enumerate(cols):
            ax = axes[r, c]
            vals, errs, colors = [], [], []
            for label, color, kind, payload in rows:
                q = payload.get(conv)
                if q is None:
                    continue
                if kind == "seed":
                    vals.append(q["overall_mean"] if ch == "total"
                                else q["per_channel_mean"][ch])
                    errs.append(q["overall_std"] if ch == "total"
                                else q["per_channel_std"][ch])
                else:
                    vals.append(q["overall"] if ch == "total" else q["per_channel"][ch])
                    errs.append(0.0)                  # single-model methods have no seed spread
                colors.append(color)

            bars = ax.bar(range(len(vals)), vals, yerr=errs, color=colors, width=0.66,
                          zorder=3, error_kw=dict(ecolor=ps.INK, lw=0.9, capsize=2.5))
            hi = max(v + e for v, e in zip(vals, errs)) if vals else 1.0
            # Values labelled vertically: 5 bars x 4 decimal places will always
            # collide horizontally (the first version collided into
            # ".0242.0246"). Vertical labels each only take up one bar's width.
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + hi * 0.04, f"{v:.4f}".lstrip("0"),
                        ha="center", va="bottom", fontsize=6.2, color=ps.INK_MUTED,
                        rotation=90)
            ax.set_ylim(0, hi * 1.42)          # leave enough room for the vertical labels
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels([], )
            ax.tick_params(axis="x", length=0)
            ax.set_title(ch if r == 0 else "", fontsize=ps.FS_LABEL, fontweight="bold",
                         color=ps.INK, pad=8)
            if c == 0:
                ax.set_ylabel(f"{conv_title}\nblind MSE", fontsize=7.4, color=ps.INK)

    # Legend placed in a row at the bottom, one colour swatch per method -- no
    # need to repeat labels inside each panel
    handles = [__import__("matplotlib").patches.Patch(facecolor=col, label=lab)
               for lab, col, _, _ in rows]
    fig.legend(handles=handles, loc="lower center", ncol=len(rows), frameon=False,
               bbox_to_anchor=(0.5, -0.035), fontsize=ps.FS_TICK)

    n_days = doc.get("protocol", {}).get("n_days", "?")
    fig.suptitle(
        f"Per-channel reconstruction error, five methods  —  {n_days} held-out days, "
        "full day, obs_every_k=1, identical clipping\n"
        "Single models throughout (no ensembling). Error bars on the two 4DVarNet arms are "
        "±1 s.d. over 5 training seeds.\n"
        "Each panel has its own y axis: the four channels differ by an order of magnitude, "
        "so a shared axis would flatten density/vy/var under vx. Panels are not comparable "
        "to each other.",
        fontsize=ps.FS_TICK, color=ps.INK_MUTED, y=1.04)

    ps.save(fig, args.out)


if __name__ == "__main__":
    main()
