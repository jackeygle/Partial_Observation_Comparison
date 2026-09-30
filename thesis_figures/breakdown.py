"""Thesis figure: where the reconstruction error comes from.

    python3 -m thesis_figures.breakdown [breakdown_dir]   # -> thesis_figures/out/breakdown.{pdf,png}

Reads supervisor_evaluation/outputs/full/breakdown/{summary.json, per_frame.npz}
(supervisor_evaluation/breakdown.py). All panels: pooled RMSE over the four channels on
blind walkable cells, the same common frames for every method.
(a) by the time since the blind cell was last observed; the grey bars give the share of
    blind cells in each bin.
(b) by the number of people in the corridor at that second (frames split into five
    groups of equal size).
(c) on each of the seven test days.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from compare import plotstyle as ps
from thesis_figures.common import FS, FS_SMALL, GREY, method_color

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LABEL = {"Senseiver-A": "Senseiver-A", "Senseiver-G": "Senseiver-G (ours)", "DINCAE": "DINCAE",
         "4DVarNet": "4DVarNet", "4DVarNet aug. var.": "4DVarNet aug. var. (ours)", "EnKF": "EnKF"}
MARK = {"Senseiver-A": "o", "Senseiver-G": "s", "DINCAE": "^", "4DVarNet": "D",
        "4DVarNet aug. var.": "v", "EnKF": "P"}


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "supervisor_evaluation/outputs/full/breakdown")
    S = json.load(open(os.path.join(src, "summary.json")))
    F = np.load(os.path.join(src, "per_frame.npz"))
    methods = S["methods"]
    ps.use(dpi=300)
    plt.rcParams.update({"font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS,
                         "xtick.labelsize": FS_SMALL, "ytick.labelsize": FS_SMALL,
                         "legend.fontsize": FS_SMALL, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "grid.linewidth": 0.5})
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(16 / 2.54, 6.2 / 2.54),
                                  gridspec_kw={"width_ratios": [1.15, 1.0, 1.15]},
                                  constrained_layout=True)
    kw = dict(lw=0.9, ms=3.0, mew=0)

    # (a) age of the blind cell -------------------------------------------------------------
    labs = S["age_labels"][:-1]                          # "never" holds a handful of cells
    n = np.asarray(S["age_cells"][:-1]); share = n / n.sum()
    x = np.arange(len(labs))
    a2 = a.twinx()
    a2.bar(x, share, 0.62, color="#e3e3e3", zorder=0)
    a2.set_ylim(0, 1.0); a2.set_yticks([]); a2.spines[:].set_visible(False)
    a.set_zorder(a2.get_zorder() + 1); a.patch.set_visible(False)
    for m in methods:
        a.plot(x, S["age_rmse"][m][:-1], marker=MARK[m], color=method_color(LABEL[m]), **kw)
    for xi, s_ in zip(x, share):
        a2.text(xi, s_ + 0.015, f"{100 * s_:.0f}%", ha="center", va="bottom", fontsize=FS_SMALL - 1.5,
                color=GREY)
    a.set_xticks(x, [r"$>$200" if l == ">200" else l for l in labs])
    a.set_xlabel("seconds since last observed")
    a.set_ylabel("RMSE")
    a.set_title("(a) by age of the blind cell", loc="left")
    a.set_ylim(0, None)

    # (b) people in the corridor ------------------------------------------------------------
    people, nb = F["people"], F["n_blind"]
    edges = np.quantile(people, np.linspace(0, 1, 6))
    grp = np.clip(np.searchsorted(edges, people, side="right") - 1, 0, 4)
    xs = [float(np.mean(people[grp == g])) for g in range(5)]
    for i, m in enumerate(methods):
        se = F[f"se_{i}"]
        y = [np.sqrt(se[grp == g].sum() / (4 * nb[grp == g].sum())) for g in range(5)]
        b.plot(xs, y, marker=MARK[m], color=method_color(LABEL[m]), **kw)
    b.set_xlabel("people in the corridor")
    b.set_title("(b) by crowd size", loc="left")
    b.set_ylim(0, None)

    # (c) per test day ----------------------------------------------------------------------
    days = sorted(set(F["day"].tolist()))
    xd = np.arange(len(days))
    for i, m in enumerate(methods):
        se = F[f"se_{i}"]
        y = [np.sqrt(se[F["day"] == d].sum() / (4 * nb[F["day"] == d].sum())) for d in days]
        c.plot(xd, y, marker=MARK[m], color=method_color(LABEL[m]), label=LABEL[m], **kw)
    mon = {"08": "Aug", "09": "Sep"}
    c.set_xticks(xd, [f"{int(d[10:12])}\n{mon[d[8:10]]}" for d in days])
    c.set_xlabel("test day (2013)")
    c.set_title("(c) by test day", loc="left")
    c.set_ylim(0, None)

    fig.legend(*c.get_legend_handles_labels(), loc="outside upper center", ncol=6,
               handlelength=1.6, columnspacing=1.0, handletextpad=0.4)
    out = os.path.join(HERE, "out", "breakdown")
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png")
    print(f"[figure] {out}.pdf", flush=True)


if __name__ == "__main__":
    main()
