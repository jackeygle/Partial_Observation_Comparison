"""
plot_compare5.py — the main results figure: 4 methods x 4 channels, two walkable scopes

Reads `compare/results/compare5_final.json` and draws both reported scopes, per channel:
blind walkable cells (top row) and all walkable cells, observed ones included (bottom row).

**One model per method, no ensembling, no seed averaging.** DINCAE, Senseiver
and the EnKF each contribute one model; 4DVarNet contributes the MSE seed whose
checkpoint scored lowest on the VALIDATION split, recorded by compare5 as
`single_model["MSE"]`. Averaging 4DVarNet's 5 seeds against everyone else's single
model would give it five times the training. The uncertainty designs (vsb0, aug0)
are not in this figure: they answer a different question and are compared, as
ensembles, in the README's uncertainty table.

Layout: **2 rows x 5 columns of small multiples** -- one row per scope; density / vx / vy / var / total.

Why small multiples rather than a grouped bar chart: the four channels differ by
an order of magnitude (vx 0.50, var 0.04). On a shared y-axis, the density/vy/var
panels would flatten into invisible slivers, leaving the reader only able to see
the difference in vx -- which happens to be the one channel where 4DVarNet leads,
giving an impression opposite to the data. One panel per channel with its own
y-axis compares "who wins within this channel," which is exactly the information
to convey. That panels are not comparable to each other is stated in the subtitle.

The y-axis is always linear, auto-scaled per panel, never log, never truncated, and every
bar has its value labelled on top.

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

#: The four reconstruction methods, one model each.
#:   ("row",    row name prefix)  read directly from results[...]
#:   ("single", arm label)        the arm's validation-best seed, from single_model[...]
ROWS = [
    ("row",    "DINCAE",    "DINCAE",              "DINCAE"),
    ("single", "MSE",       "4DVarNet (MSE loss)", "4DVarNet MSE"),
    ("row",    "Senseiver", "Senseiver",           "Senseiver"),
    ("row",    "EnKF k1",   "EnKF",                "EnKF k1"),
]

PANELS = [
    ("walkable",      "blind walkable cells"),
    ("walkable_full", "all walkable cells"),
]


def pick(results: dict, prefix: str):
    """Fetch a row by prefix, tolerating member-count differences like ens5/ens2."""
    for k in results:
        if k.startswith(prefix):
            return k, results[k]
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(paths.COMPARE, "results", "compare5_final.json"))
    ap.add_argument("--out", default=os.path.join(paths.COMPARE, "results", "compare5.png"))
    args = ap.parse_args()

    with open(args.json) as f:
        doc = json.load(f)
    results, chans = doc["results"], doc["channels"]
    cols = chans + ["total"]

    single = doc.get("single_model", {})
    rows = []                      # (label, color, payload)
    for kind, ref, label, ckey in ROWS:
        if kind == "single":
            sm = single.get(ref)
            if sm is None or sm["row"] not in results:
                print(f"  [warn] {args.json} has no single_model[{ref!r}], skipping this row")
                continue
            rows.append((f"{label}, seed {sm['seed']}", ps.method_color(sm["row"]),
                         results[sm["row"]]))
        else:
            key, entry = pick(results, ref)
            if entry is None:
                print(f"  [warn] {args.json} has no {ref}*, skipping this row")
                continue
            rows.append((label, ps.method_color(key), entry))

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
            vals, colors = [], []
            for label, color, payload in rows:
                q = payload.get(conv)
                if q is None:
                    continue
                vals.append(q["overall"] if ch == "total" else q["per_channel"][ch])
                colors.append(color)

            bars = ax.bar(range(len(vals)), vals, color=colors, width=0.66, zorder=3)
            hi = max(vals) if vals else 1.0
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
                ax.set_ylabel(f"{conv_title}\nMSE", fontsize=7.4, color=ps.INK)

    # Legend placed in a row at the bottom, one colour swatch per method -- no
    # need to repeat labels inside each panel
    handles = [__import__("matplotlib").patches.Patch(facecolor=col, label=lab)
               for lab, col, _ in rows]
    fig.legend(handles=handles, loc="lower center", ncol=len(rows), frameon=False,
               bbox_to_anchor=(0.5, -0.035), fontsize=ps.FS_TICK)

    n_days = doc.get("protocol", {}).get("n_days", "?")
    fig.suptitle(
        f"Per-channel reconstruction error  —  {n_days} held-out days, "
        "full day, obs_every_k=1, identical clipping\n"
        "One model per method, no ensembling or seed averaging; 4DVarNet is its "
        "validation-best MSE seed.\n"
        "Each panel has its own y axis: the four channels differ by an order of magnitude, "
        "so a shared axis would flatten density/vy/var under vx. Panels are not comparable "
        "to each other.",
        # With a single row the figure is short, so a fixed y=1.04 put the third subtitle line on
        # top of the panel titles. Push it up by an amount that shrinks as rows are added.
        fontsize=ps.FS_TICK, color=ps.INK_MUTED, y=1.04 + 0.16 / len(PANELS))

    ps.save(fig, args.out)


if __name__ == "__main__":
    main()
