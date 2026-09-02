"""
plot_comparison.py  —  4DVarNet vs EnKF reconstruction-accuracy table + bar chart
=================================================================================

Reads the two evaluation summaries (both scored on the SAME held-out test frames
with the SAME metric) and renders a clean comparison:

    check_outputs/eval/test_metrics_matched_clip.json   4DVarNet (matched frames)
    check_outputs/eval/enkf_metrics.json           Localized EnKF + apt-ibex

Outputs check_outputs/eval/comparison.png (table + grouped bar of blind-zone /
full-state MSE). Reconstruction accuracy only — uncertainty is shown separately
(EnKF's advantage), not compared here.

Run:  python3 checks/plot_comparison.py
"""
from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = os.path.join(ROOT, "check_outputs", "eval")


def main():
    v = json.load(open(os.path.join(EV, "test_metrics_matched_clip.json")))
    e = json.load(open(os.path.join(EV, "enkf_metrics.json")))

    # Report RMSE (= sqrt of the MSE the JSONs store), matching how the EnKF project scores
    # its own filter (ENKF.py evaluate_enkf: rmse = sqrt(mean((est-true)^2))). RMSE is in the
    # state's own units, so a velocity error reads directly in m/s. Per-day RMSE is taken
    # first, then mean/std over the 7 days — sqrt of a mean is not the mean of sqrts.
    def rmse_stats(d, key):
        r = np.sqrt(np.array([p[key] for p in d["per_day"]], dtype=float))
        return float(r.mean()), float(r.std())

    methods = ["4DVarNet\n(ours)", "EnKF\n(apt-ibex)"]
    (vb, vbs), (eb, ebs) = rmse_stats(v, "blind_mse"), rmse_stats(e, "blind_mse")
    (vf, vfs), (ef, efs) = rmse_stats(v, "full_mse"), rmse_stats(e, "full_mse")
    blind, blind_sd = [vb, eb], [vbs, ebs]
    full, full_sd = [vf, ef], [vfs, efs]

    fig, (axt, axb) = plt.subplots(1, 2, figsize=(15, 5.5),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # ---- table ----
    axt.axis("off")
    rows = [
        ["metric (held-out test days)", "4DVarNet", "EnKF"],
        ["blind-zone RMSE", f"{blind[0]:.4f} ± {blind_sd[0]:.4f}", f"{blind[1]:.4f} ± {blind_sd[1]:.4f}"],
        ["full-state RMSE", f"{full[0]:.4f} ± {full_sd[0]:.4f}", f"{full[1]:.4f} ± {full_sd[1]:.4f}"],
    ]
    tbl = axt.table(cellText=rows, cellLoc="center", loc="center", colWidths=[0.42, 0.29, 0.29])
    tbl.auto_set_font_size(False); tbl.set_fontsize(13); tbl.scale(1, 2.4)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#ccc")
        if r == 0:
            cell.set_facecolor("#0e6b8a"); cell.get_text().set_color("white"); cell.get_text().set_fontweight("bold")
        elif c == 0:
            cell.get_text().set_fontweight("bold"); cell.set_facecolor("#eef3fa")
    axt.set_title("Reconstruction accuracy (same test frames, same metric)", fontsize=12, pad=12)

    # ---- grouped bar ----
    x = np.arange(2); w = 0.35
    axb.bar(x - w/2, blind, w, yerr=blind_sd, capsize=4, color="#0e6b8a", label="blind-zone RMSE")
    axb.bar(x + w/2, full, w, yerr=full_sd, capsize=4, color="#2e9e46", label="full-state RMSE")
    axb.set_xticks(x); axb.set_xticklabels(methods)
    axb.set_ylabel("RMSE"); axb.legend(fontsize=10); axb.grid(axis="y", alpha=0.25)
    axb.set_title("lower is better", fontsize=11)

    fig.suptitle("4DVarNet vs EnKF — reconstruction RMSE on held-out test days", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(EV, "comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[figure] {out}")
    print(f"  4DVarNet blind RMSE {blind[0]:.4f} | EnKF blind RMSE {blind[1]:.4f}  "
          f"({'4DVarNet' if blind[0]<blind[1] else 'EnKF'} better)")


if __name__ == "__main__":
    main()
