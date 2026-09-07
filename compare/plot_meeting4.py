"""
plot_meeting4.py — the four figures for the 2026-09-07 meeting

One throughline, three checks: **the ranking of the four methods depends on a
scoring convention that most papers never state.**

    fig1  capability matrix   what each of the four methods actually provides
                               (uncertainty is not a shared property)
    fig2  accuracy            main convention vs. the control convention, ranking flips
    fig3  uncertainty         structural vs. bolted-on, robust across conventions
    fig4  speed               throughput vs. latency, ranking reverses

All figure text is English: Triton's matplotlib has only 24 fonts and no CJK ones,
so Chinese would render as tofu boxes. Colours come from compare/plotstyle.py,
consistent for each method across every figure.

Usage (the login node is fine, pure matplotlib):
    source sbatch/_env.sh && python3 -m compare.plot_meeting4
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from compare import plotstyle as ps
from crowdcore import paths

OUT = os.path.join(paths.method(paths.VARNET), "check_outputs", "eval")
CH = ("density", "vx", "vy", "var")


def _rmse(entry, conv):
    """Pooled RMSE: sum squared error and cell count per channel first, divide
    once at the end (same convention as compare5)."""
    q = entry[conv]
    se = sum(q["per_channel"][c] * q["n_per_channel"][c] for c in CH)
    n = sum(q["n_per_channel"][c] for c in CH)
    return math.sqrt(se / n)


# --------------------------------------------------------------------------- #
def fig_capability(path):
    """What the four methods provide -- uncertainty is not a shared property;
    forcing it into a symmetric table would be dishonest."""
    ps.use()
    fig, ax = ps.figure(width=ps.SLIDE_W, rows_h=3.0)
    ax.axis("off")
    rows = [("Senseiver", "yes", "no", "—"),
            ("DINCAE", "yes", "yes", "part of the method\n(information form)"),
            ("EnKF", "yes", "yes", "part of the method\n(ensemble spread)"),
            ("4DVarNet", "yes", "yes", "ADDED BY US\n(2 designs; not in the paper)")]
    cols = ["Method", "Reconstruction", "Uncertainty", "Where sigma-hat comes from"]
    # Column spacing set by the longest header, "Reconstruction" -- the first
    # version's 0.26/0.44 made it collide with "Uncertainty".
    xs = [0.03, 0.28, 0.50, 0.66]
    for x, c in zip(xs, cols):
        ax.text(x, 0.88, c, fontsize=11, fontweight="bold", color=ps.INK,
                transform=ax.transAxes)
    ax.plot([0.02, 0.98], [0.84, 0.84], color=ps.INK, lw=1.1, transform=ax.transAxes)
    for i, r in enumerate(rows):
        y = 0.70 - i * 0.175
        col = ps.method_color(r[0])
        ax.add_patch(__import__("matplotlib").patches.Rectangle(
            (0.02, y - 0.035), 0.012, 0.10, color=col, transform=ax.transAxes, clip_on=False))
        for x, t in zip(xs, r):
            bold = (x == xs[0])
            ax.text(x + (0.022 if x == xs[0] else 0), y, t, fontsize=10.2,
                    fontweight="bold" if bold else "normal",
                    color=ps.INK if bold else ps.INK_MUTED,
                    va="center", transform=ax.transAxes)
    ax.text(0.02, -0.02,
            "Accuracy compares 4 methods. Uncertainty compares 3 — and one of those three "
            "has a sigma-hat we added ourselves.",
            fontsize=9.6, style="italic", color=ps.INK_MUTED, transform=ax.transAxes)
    ps.save(fig, path)


# --------------------------------------------------------------------------- #
def fig_accuracy(path, jf):
    """Check 1: the same predictions, switch the convention, first and last place swap.

    **2x2, not two bars stacked in one panel.** The first version encoded
    "full field / blind" as two shades of the same colour and used a blue legend
    to explain it -- but each method already has its own colour, so the reader
    sees an orange pair of bars with no way to map it to the blue legend, and ends
    up thinking "blue = blind". Changed to rows = convention, columns = cell
    scope, each panel four bars in plain method colours -- no legend, no
    shade-coding needed.

    The ablation row (DINCAE full-field) is not in this figure -- it belongs to
    the closing slide; mixed into the main list it would be read as a fifth
    method.
    """
    R = json.load(open(jf))["results"]
    rows = [("DINCAE", "DINCAE"), ("Senseiver", "Senseiver"),
            ("4DVarNet", "4DVarNet MSE s0"), ("EnKF", "EnKF k1")]

    ps.use()
    fig, axes = ps.figure(ncols=2, nrows=2, width=ps.SLIDE_W, rows_h=2.35,
                          gridspec_kw={"wspace": 0.30, "hspace": 0.62})
    panels = [((0, 0), "defined_full", "full field"),
              ((0, 1), "defined",      "blind cells only"),
              ((1, 0), "full",         "full field"),
              ((1, 1), "allcells",     "blind cells only")]
    conv_title = {0: "Convention A — channel-defined cells   (velocity scored only where people are)",
                  1: "Convention B — all cells   (velocity also scored on empty cells)"}
    for (r, c), conv, scope in panels:
        ax = axes[r][c]
        vals = sorted(((L, _rmse(R[k], conv)) for L, k in rows), key=lambda t: t[1])
        labs = [t[0] for t in vals]; v = [t[1] for t in vals]
        y = list(range(len(v)))[::-1]
        ax.barh(y, v, color=[ps.method_color(l) for l in labs], height=0.62, zorder=3)
        for i, x in zip(y, v):
            ax.text(x + max(v) * 0.02, i, f"{x:.3f}", va="center", fontsize=9.0, color=ps.INK)
        ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=9.2)
        ax.set_xlim(0, max(v) * 1.22)
        ax.set_title(scope, fontsize=9.8, fontweight="bold", color=ps.INK, pad=5)
        ax.tick_params(axis="x", labelsize=8.0)
        ax.grid(axis="y", visible=False)
        if c == 0:      # write the convention on the left of each row, spanning both panels
            ax.text(0.0, 1.30, conv_title[r], transform=ax.transAxes, fontsize=10.4,
                    fontweight="bold", color=ps.INK)
    for c in (0, 1):
        axes[1][c].set_xlabel("RMSE   (lower is better)", fontsize=9.0)
    fig.suptitle("Same predictions, same 7 held-out days — only the scoring convention differs",
                 fontsize=11.6, fontweight="bold", color=ps.INK, y=1.06)
    ps.save(fig, path)


# --------------------------------------------------------------------------- #
def fig_uncertainty(path):
    """Check 2: structural vs. bolted-on. Verdict = CRPS relative to that slice's
    own constant-sigma null model."""
    src = [("4DVarNet aug0", "augmented state", paths.method(paths.VARNET),
            "uncertainty_aug0.json", True),
           ("DINCAE", "information form", paths.method(paths.DINCAE),
            "uncertainty_dincae.json", True),
           ("4DVarNet vsb0", "read-out head", paths.method(paths.VARNET),
            "uncertainty_vsb0.json", False),
           ("EnKF", "ensemble spread", paths.method(paths.ENKF),
            "uncertainty_enkf_k1.json", False)]
    rows = []
    for lab, mech, root, f, structural in src:
        R = json.load(open(os.path.join(root, "check_outputs", "eval", f)))["results"]
        m, b = R["defined_blind"], R["defined_blind_constant_sigma_baseline"]
        rows.append((lab, mech, 100 * (m["crps"] / b["crps"] - 1),
                     m["spread_skill"], 100 * m["coverage"]["90"], structural))

    ps.use()
    fig, axes = ps.figure(ncols=2, width=ps.SLIDE_W, rows_h=3.7,
                          gridspec_kw={"wspace": 0.30})
    y = range(len(rows))[::-1]
    labs = [f"{r[0]}\n{r[1]}" for r in rows]

    ax = axes[0]                                     # verdict: CRPS vs. the constant-sigma null model
    v = [r[2] for r in rows]
    cols = ["#2a9d5c" if x < 0 else "#c0392b" for x in v]
    bb = ax.barh(y, v, color=cols, height=0.62, zorder=3)
    # Values are labelled **inside** the bar by default: a negative bar extends
    # left, and labelling outside it would collide with the y-axis method names
    # (the first version did exactly that). But when a bar is short enough that
    # the label text is wider than the bar itself (the "+2.6%" one), the label
    # moves outside instead.
    span = max(abs(min(v)), abs(max(v)))
    for b_, x in zip(bb, v):
        yc = b_.get_y() + b_.get_height() / 2
        if abs(x) >= 0.30 * span:
            ax.text(x * 0.5, yc, f"{x:+.1f}%", va="center", ha="center",
                    fontsize=10.4, fontweight="bold", color="white")
        else:
            off = 0.035 * span
            ax.text(x + (off if x > 0 else -off), yc, f"{x:+.1f}%", va="center",
                    ha="left" if x > 0 else "right",
                    fontsize=10.4, fontweight="bold", color=ps.INK)
    ax.axvline(0, color=ps.INK, lw=1.2)
    ax.set_yticks(list(y)); ax.set_yticklabels(labs, fontsize=9.4)
    pad = max(abs(min(v)), abs(max(v))) * 0.22
    ax.set_xlim(min(v) - pad, max(v) + pad)
    ax.set_xlabel("CRPS relative to the constant-sigma null model\n"
                  "negative = the learnt sigma knows WHERE the error is", fontsize=9.2)
    ax.set_title("Verdict", fontsize=10.6, fontweight="bold", color=ps.INK, pad=9)
    ax.grid(axis="y", visible=False)

    ax = axes[1]                                     # calibration: spread/skill and 90% coverage
    sk = [r[3] for r in rows]
    cov = [r[4] / 100 for r in rows]
    ax.barh([i + 0.17 for i in y], sk, height=0.32, color="#4269d0", zorder=3,
            label="spread / skill      (1.0 = right scale)")
    ax.barh([i - 0.19 for i in y], cov, height=0.32, color="#efb118", zorder=3,
            label="90% coverage      (0.90 = nominal)")
    ax.axvline(1.0, color=ps.INK, ls="--", lw=1.2, zorder=2)
    ax.axvline(0.9, color="#efb118", ls=":", lw=1.4, zorder=2)
    for i, (sv, cv) in zip(y, zip(sk, cov)):
        ax.text(sv + 0.03, i + 0.17, f"{sv:.3f}", va="center", fontsize=8.8, color=ps.INK)
        ax.text(cv + 0.03, i - 0.19, f"{100 * cv:.0f}%", va="center", fontsize=8.8, color=ps.INK)
    ax.set_yticks(list(y)); ax.set_yticklabels([r[0] for r in rows], fontsize=9.4)
    ax.set_xlim(0, 1.62)
    ax.set_xlabel("ratio", fontsize=9.2)
    ax.set_title("Calibration", fontsize=10.6, fontweight="bold", color=ps.INK, pad=9)
    # Legend moved below the panel: placing it at lower right would cover the
    # EnKF bars, which sit almost flat against the axis.
    ax.legend(fontsize=8.4, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.20), ncol=1, handlelength=1.4)
    ax.grid(axis="y", visible=False)

    fig.suptitle("Uncertainty built INTO the model beats the constant-sigma null model.\n"
                 "Bolted on afterwards, or derived from ensemble spread, does not.",
                 fontsize=11.4, fontweight="bold", color=ps.INK, y=1.10)
    ps.save(fig, path)


# --------------------------------------------------------------------------- #
def fig_speed(path):
    """Inference time: per frame, mean +/- s.d., all four methods measured
    interleaved on the same node over the same frames.

    Plots only **throughput** (total time / frame count) -- the latency column
    was dropped per request.

    A footnote explains what the 4DVarNet number actually is: it solves a
    dT=200-frame window at once, so 2.74 ms is an amortised figure, not "the
    delay to produce one frame" (that would be 0.55 s). The other three methods
    are per-frame, so for them the two numbers are the same to begin with.
    Without this note, quoting this number on its own would reproduce the exact
    problem raised last week.
    """
    d = json.load(open(os.path.join(OUT, "bench_speed_four.json")))
    keep = [("Senseiver", "senseiver"), ("DINCAE", "dincae"),
            ("4DVarNet", "varnet_single"), ("EnKF", "enkf_k1")]
    rows = [(L, d["runs"][k]) for L, k in keep if k in d["runs"]]

    ps.use()
    fig, ax = ps.figure(width=ps.SLIDE_W * 0.80, rows_h=3.5)
    vals = [1000 * r["per_frame_s"] for _, r in rows]
    errs = [1000 * r["per_frame_std_s"] for _, r in rows]
    labs = [L for L, _ in rows]
    order = np.argsort(vals)
    vals = [vals[i] for i in order]; errs = [errs[i] for i in order]
    labs = [labs[i] for i in order]
    y = list(range(len(vals)))[::-1]
    b = ax.barh(y, vals, xerr=errs, color=[ps.method_color(l) for l in labs],
                height=0.60, zorder=3, error_kw=dict(ecolor=ps.INK, lw=1.0, capsize=3))
    for bb, v, e in zip(b, vals, errs):
        ax.text((v + e) * 1.28, bb.get_y() + bb.get_height() / 2,
                f"{v:,.2f} ± {e:,.3f} ms", va="center", fontsize=9.8, color=ps.INK)
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=10.4)
    ax.set_xscale("log")
    ax.set_xlim(min(vals) * 0.45, max(vals) * 22)
    ax.set_xlabel("milliseconds per frame   (log scale)", fontsize=9.8)
    ax.grid(axis="y", visible=False)
    n = d.get("frames", "?"); rp = d.get("repeats", "?")
    fig.suptitle("Inference time per frame — 4 methods, same node, same frames",
                 fontsize=11.8, fontweight="bold", color=ps.INK, y=1.03)
    ax.text(0.0, -0.235,
            f"mean ± s.d. over {rp} interleaved repeats of {n} frames, {d['hw']['cores_avail']} CPU "
            f"cores. Nothing printed inside the timed region.\n"
            "4DVarNet solves a whole 200-frame window at once, so its figure is amortised over "
            "the window rather than the wait for one frame.",
            transform=ax.transAxes, fontsize=8.4, color=ps.INK_MUTED)
    ps.save(fig, path)


def main():
    os.makedirs(OUT, exist_ok=True)
    jf = os.path.join(paths.COMPARE, "results", "compare5_final.json")
    fig_capability(os.path.join(OUT, "m4_capability.png"))
    fig_accuracy(os.path.join(OUT, "m4_accuracy.png"), jf)
    fig_uncertainty(os.path.join(OUT, "m4_uncertainty.png"))
    fig_speed(os.path.join(OUT, "m4_speed.png"))


if __name__ == "__main__":
    main()
