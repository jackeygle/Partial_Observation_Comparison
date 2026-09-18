"""
plot_why_sigma_fails.py — three plain figures, one causal step each.

The question is narrow: why is the EnKF's uncertainty bad? The chain is

    ensemble collapses  ->  sigma becomes an echo of the injected noise  ->  it carries nothing

so there are three figures, and each one is a single panel with at most two lines. An earlier
version of this had six heatmap panels and log-log axes; it showed more and communicated less.

    A  the symptom    sigma is flat all day while the actual error swings with the crowd
    B  the cause      the ensemble collapses in ~5 frames onto the level that is injected
    C  the cost       sigma ends up worse than a constant (vsb0's too, for the opposite reason)

Figure text is English: Triton's matplotlib has no CJK font (see compare/plotstyle.py).

    python3 -m methods.enkf.checks.plot_why_sigma_fails
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from crowdcore import paths                                              # noqa: E402
from compare import plotstyle as ps                                      # noqa: E402

CH = ("density", "vx", "vy", "var")
PROC_STD = np.array([0.02829307, 0.31263075, 0.12325809, 0.41680932])
INJECTED = 0.01 * PROC_STD          # what forecast() actually adds each step


def fig_a(S, X, Est, out, ch=0, every=20):
    """Symptom. Per frame: the field-wide RMS error, and the field-wide mean sigma. One is the
    thing to be estimated, the other is the estimate. Thinned by `every` frames so the day fits
    without turning into a solid block of ink."""
    fig, ax = ps.figure(rows_h=2.9)
    t = np.arange(0, len(S), every)
    err = np.sqrt(((Est[t, ch] - X[t, ch]) ** 2).reshape(len(t), -1).mean(axis=1))
    sig = S[t, ch].reshape(len(t), -1).mean(axis=1)
    hrs = t / 3600.0
    ax.plot(hrs, err, lw=0.8, color="0.35", label="actual error (RMS over the grid)")
    ax.plot(hrs, sig, lw=1.8, color=ps.METHOD_COLORS["EnKF"],
            label=r"$\sigma$ the EnKF reports")
    ax.set_xlabel("hours into the day")
    ax.set_ylabel(f"{CH[ch]}")
    ax.set_title(r"The error moves with the crowd all day. $\sigma$ does not move at all.")
    ax.legend(frameon=False, loc="upper left")
    ps.save(fig, out)


def fig_b(S, out, n=40):
    """Cause. Each channel's sigma divided by the noise injected into that channel each step.
    Every curve lands on 1-2 within about five frames: whatever the ensemble knew at the start is
    gone, and what is left is the injection."""
    fig, ax = ps.figure(rows_h=2.9)
    t = np.arange(n)
    for c in range(4):
        m = S[:n, c].reshape(n, -1).mean(axis=1) / INJECTED[c]
        ax.plot(t, m, lw=1.4, label=CH[c])
    ax.axhspan(1.0, 2.0, color="0.85", zorder=0)
    ax.text(n * 0.62, 1.5, "the injected level", fontsize=8, color="0.35", va="center")
    ax.set_xlabel("frame")
    ax.set_ylabel(r"$\sigma$  /  noise injected per step")
    ax.set_title("Within ~5 frames the ensemble has forgotten everything\n"
                 "except the noise that was just added")
    ax.set_ylim(0, 12)
    ax.legend(ncol=4, frameon=False, loc="upper right")
    ps.save(fig, out)


#: The reported sigma-hat of each method: the same JSONs the README's uncertainty tables read.
#: (label, json, colour key); the 4DVarNet rows are the 5-seed ensembles the README reports.
COST_SRC = [("EnKF",          "methods/enkf/check_outputs/eval/uncertainty_enkf_k1.json",  "EnKF k1"),
            ("4DVarNet vsb0", "methods/varnet/check_outputs/eval/uncertainty_vsb0.json",   "4DVarNet NLL ens5"),
            ("4DVarNet aug0", "methods/varnet/check_outputs/eval/uncertainty_aug0.json",   "4DVarNet AUG ens5"),
            ("DINCAE",        "methods/dincae/check_outputs/eval/uncertainty_dincae.json", "DINCAE")]
#: The two cell sets the README reports, under the names the uncertainty JSONs use.
COST_SCOPES = [("walkable_blind", "blind walkable cells"), ("walkable", "all walkable cells")]


def cost_ratios():
    """{label: {scope: CRPS / that method's own constant-sigma null-model CRPS}}"""
    out = {}
    for nm, p, _ in COST_SRC:
        r = json.load(open(os.path.join(paths.ROOT, p)))["results"]
        out[nm] = {s: r[s]["crps"] / r[f"{s}_constant_sigma_baseline"]["crps"] for s, _ in COST_SCOPES}
    return out


def fig_c(out):
    """Cost. CRPS divided by each method's OWN constant-sigma null model -- a null that knows the
    average error size and nothing about where it falls. The ratio is the only cross-method
    comparable form (DINCAE scores in a transformed space; the space cancels in a ratio).
    Above 1 means sigma carries less than a constant of the right size. Solid bars are blind
    walkable cells, light bars all walkable cells, as in the README."""
    from matplotlib.patches import Patch
    R = cost_ratios()
    fig, ax = ps.figure(rows_h=2.9)
    x, w = np.arange(len(COST_SRC)), 0.38
    for i, (s, _) in enumerate(COST_SCOPES):
        vals = [R[nm][s] for nm, _, _ in COST_SRC]
        bars = ax.bar(x + (i - 0.5) * w, vals, w * 0.92, alpha=1.0 if i == 0 else 0.5,
                      color=[ps.method_color(key) for _, _, key in COST_SRC])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=8)
    ax.axhline(1.0, ls="--", lw=1.0, color="0.3")
    ax.set_xticks(x, [nm for nm, _, _ in COST_SRC])
    ax.set_xlim(-0.6, len(COST_SRC) + 0.35)
    ax.text(len(COST_SRC) - 0.4, 1.03, "worse than\na constant", fontsize=7.5, color="0.3", va="bottom")
    ax.text(len(COST_SRC) - 0.4, 0.97, "better than\na constant", fontsize=7.5, color="0.3", va="top")
    ax.set_ylabel("CRPS  /  CRPS of a constant")
    ax.set_title(r"Two $\sigma$s are worse than a plain constant: the EnKF's and vsb0's")
    ax.set_ylim(0, max(max(v.values()) for v in R.values()) * 1.2)
    ax.legend(handles=[Patch(facecolor="0.35", label=COST_SCOPES[0][1]),
                       Patch(facecolor="0.35", alpha=0.5, label=COST_SCOPES[1][1])],
              frameon=False, loc="upper center", ncol=2, fontsize=8)
    ps.save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default="atc-20130811")
    ap.add_argument("--outdir", default=os.path.join(paths.eval_out(paths.ENKF), "figs"))
    ap.add_argument("--figs", default="ABC",
                    help="which figures to draw; C alone reads only the uncertainty JSONs")
    a = ap.parse_args()
    ps.use()
    if "A" in a.figs or "B" in a.figs:
        src = paths.enkf_export("enkf_k1_full")
        with np.load(os.path.join(src, f"est_{a.day}.npz")) as e:
            S, Est = e["Spread"], e["Est"]
        with np.load(os.path.join(src, f"obs_{a.day}.npz")) as o:
            X = o["X_true"][:len(S)]
        if "A" in a.figs:
            fig_a(S, X, Est, os.path.join(a.outdir, "A_symptom.png"))
        if "B" in a.figs:
            fig_b(S, os.path.join(a.outdir, "B_cause.png"))
    if "C" in a.figs:
        fig_c(os.path.join(a.outdir, "C_cost.png"))


if __name__ == "__main__":
    main()
