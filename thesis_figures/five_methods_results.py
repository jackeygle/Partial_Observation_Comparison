"""Result figures (accuracy, uncertainty, latency) for the separate five-method thesis.

    python3 -m thesis_figures.five_methods_results   # -> thesis_figures/out_5methods/{accuracy,uncertainty,latency}.{pdf,png}

Copies the final evaluation's CSVs without the aug. var. 4DVarNet rows into
out_5methods/results/ and redraws them with evaluate.plot_summary, so the numbers are
exactly those of the evaluation (common.RESULTS) and nothing there is changed.
"""
from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path

from supervisor_evaluation import evaluate as ev

HERE = Path(__file__).resolve().parent
from thesis_figures.common import OUT_DIR, RESULTS, SIX_ALLWALK  # noqa: E402
SRC = HERE.parent / RESULTS
OUT = HERE / OUT_DIR      # out_5methods, or out_6methods with THESIS_VARIANT=6methods
AUG = ev.FINAL_CONFIG["varnet_aughead"]["label"]
CSVS = ("accuracy.csv", "uncertainty.csv", "uncertainty_by_channel.csv",
        "inference_time.csv", "inference_time_controlled.csv", "accuracy_all_walkable.csv",
        "uncertainty_all_walkable.csv", "uncertainty_by_channel_all_walkable.csv")
STEMS = {"accuracy_rmse": "accuracy", "uncertainty_summary": "uncertainty",
         "inference_latency": "latency"}


def accuracy_two_panels(work: Path) -> None:
    """accuracy.pdf with (a) all walkable cells and (b) blind cells, on one RMSE axis.

    Same bars, colours and labels as evaluate.plot_summary's accuracy figure; (a) reads
    accuracy_all_walkable.csv (supervisor_evaluation/checks/accuracy_all_walkable.py)."""
    import matplotlib.pyplot as plt
    import numpy as np
    from compare import plotstyle as ps

    ps.use(dpi=300)
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "grid.linewidth": 0.5, "lines.linewidth": 0.8})
    short = {ev.FINAL_CONFIG[k]["label"]: ev.SHORT_LABEL[k] for k in ev.FINAL_CONFIG}
    labels = ["All", "Density", "$v_x$", "$v_y$", "Vel. var."]
    chans = ["density_rmse", "vx_rmse", "vy_rmse", "variance_rmse"]
    blind = list(csv.DictReader((work / "accuracy.csv").open()))
    allw = {r["method"]: r for r in csv.DictReader((work / "accuracy_all_walkable.csv").open())}
    panels = [("(a) all walkable cells, observed and blind",
               [[float(allw[r["method"]]["all_walkable_rmse"])]
                + [float(allw[r["method"]][c]) for c in chans] for r in blind]),
              ("(b) blind cells", [[float(r["blind_rmse"])] + [float(r[c]) for c in chans]
                                   for r in blind])]
    fig, axes = ps.figure(nrows=2, rows_h=2.3, sharey=True, constrained_layout=True)
    x, n = np.arange(len(labels)), len(blind)
    w = 0.8 / n
    for ax, (title, values) in zip(axes, panels):
        for i, (r, v) in enumerate(zip(blind, values)):
            ax.bar(x + (i - (n - 1) / 2) * w, v, w, color=ps.method_color(r["method"]),
                   edgecolor="white", linewidth=0.6, label=short.get(r["method"], r["method"]))
        ax.set_xticks(x, labels)
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel("RMSE (lower is better)")
        ax.set_title(title, loc="left", fontsize=8)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=5,
               handlelength=1.2, columnspacing=1.5)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"accuracy.{ext}")
    plt.close(fig)
    print(f"[figure] {OUT / 'accuracy.pdf'} (two panels)")


def uncertainty_two_scopes(work: Path) -> None:
    """uncertainty.pdf: CRPS (left) and RMS spread / RMSE (right), on all walkable cells
    (top, a-b) and on the blind cells (bottom, c-d); same bars and colours as
    evaluate.plot_summary's uncertainty figure, error bars = day-bootstrap 95% intervals."""
    import matplotlib.pyplot as plt
    import numpy as np
    from compare import plotstyle as ps

    ps.use(dpi=300)
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "grid.linewidth": 0.5, "lines.linewidth": 0.8})
    short = {ev.FINAL_CONFIG[k]["label"]: ev.SHORT_LABEL[k] for k in ev.FINAL_CONFIG}
    chans = list(ev.CHANNELS)
    glabels = ["All", "Density", "$v_x$", "$v_y$", "Vel. var."]
    scopes = [("all walkable cells, observed and blind", "uncertainty_all_walkable.csv",
               "uncertainty_by_channel_all_walkable.csv"),
              ("blind cells", "uncertainty.csv", "uncertainty_by_channel.csv")]
    fig, axes = ps.figure(nrows=2, ncols=2, rows_h=2.3, constrained_layout=True)
    letters = iter("abcd")
    for row, (scope, f_pool, f_ch) in enumerate(scopes):
        pooled = {r["method"]: r for r in csv.DictReader((work / f_pool).open())}
        per_ch = {(r["method"], r["channel"]): r for r in csv.DictReader((work / f_ch).open())}
        first = ("DINCAE",)                    # the order of the uncertainty table
        names = sorted(pooled, key=lambda m: (m not in first, m == ev.FINAL_CONFIG["enkf"]["label"]))
        x = np.arange(len(glabels)); w = 0.8 / len(names)
        for col, (key, ylabel, ref) in enumerate((
                ("crps", "CRPS (lower is better)", 0.0),
                ("spread_rmse", "RMS spread / RMSE\n(reference scale: 1)", 1.0))):
            ax = axes[row, col]
            for i, m in enumerate(names):
                v = np.array([float(pooled[m][key])] + [float(per_ch[(m, c)][key]) for c in chans])
                xs = x + (i - (len(names) - 1) / 2) * w
                ax.bar(xs, v, w, color=ps.method_color(m), edgecolor="white", linewidth=0.6,
                       label=short.get(m, m))
                lo, hi = float(pooled[m][f"{key}_lo"]), float(pooled[m][f"{key}_hi"])
                ax.errorbar(xs[0], v[0], yerr=[[max(v[0] - lo, 0.0)], [max(hi - v[0], 0.0)]],
                            fmt="none", ecolor=ps.INK, elinewidth=0.6, capsize=1.5, capthick=0.6)
            ax.axhline(ref, ls="--" if ref else "-", lw=0.6, color=ps.INK_MUTED if ref else ps.AXIS)
            ax.axvline(0.5, color=ps.RULE, lw=0.6)
            ax.set_xticks(x, glabels)
            ax.tick_params(axis="x", length=0)
            ax.set_ylabel(ylabel)
            ax.set_title(f"({next(letters)}) {scope}", loc="left", fontsize=8)
    for col in range(2):                       # one y scale per metric across both scopes
        top = max(axes[r, col].get_ylim()[1] for r in range(2))
        for r in range(2):
            axes[r, col].set_ylim(0, top)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper center", ncol=2,
               handlelength=1.2, columnspacing=1.5)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"uncertainty.{ext}")
    print(f"[figure] {OUT / 'uncertainty'}.pdf (two scopes)", flush=True)


def main() -> None:
    work = OUT / "results"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    for name in CSVS:
        src = SRC / name
        if not src.exists():
            continue
        with src.open() as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if SIX_ALLWALK or r["method"] != AUG]
            fields = reader.fieldnames
        with (work / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)
        print(f"[csv] {name}: {len(rows)} methods")
    ev.plot_summary(work)
    for stem, new in STEMS.items():
        for ext in ("pdf", "png"):
            shutil.copy(work / "figures" / f"{stem}.{ext}", OUT / f"{new}.{ext}")
        print(f"[figure] {OUT / new}.pdf")
    if (work / "accuracy_all_walkable.csv").exists():
        accuracy_two_panels(work)              # replaces the one-panel accuracy figure
    if os.environ.get("UNC_TWO_SCOPES") == "1" and (work / "uncertainty_all_walkable.csv").exists():
        uncertainty_two_scopes(work)           # opt-in: the theses report the blind cells only


if __name__ == "__main__":
    main()
