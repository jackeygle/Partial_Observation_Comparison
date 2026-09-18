"""
plot_sigma_is_constant.py — the four figures behind "the EnKF's sigma is a constant".

One claim, four pieces of evidence, one figure each:

    F1  it is constant in TIME     -- reaches a floor in ~5 frames, flat for the rest of the day,
                                      and that floor is the per-step injected noise
    F2  it is constant in SPACE    -- the sigma field is flat where the truth and the error are not
    F3  the floor IS the constant  -- raise the hard-coded injection 100x and every calibration
                                      metric moves to where it should be
    F4  what that costs in the     -- CRPS against a constant-sigma null model: the EnKF loses to
        method comparison             a constant (so does vsb0, whose sigma is too large)

Everything reads results already on disk; no filter is re-run.

Figure text is English on purpose -- Triton's matplotlib ships no CJK font, Chinese renders as
tofu (see compare/plotstyle.py).

    python3 -m methods.enkf.checks.plot_sigma_is_constant
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
#: forecast() injects rng.normal(0, 0.01 * proc_noise_vec) every step. PROC_STD is not a free
#: knob -- estimate_noise_from_data() measured it as the surrogate's own one-step error.
PROC_STD = np.array([0.02829307, 0.31263075, 0.12325809, 0.41680932])
INJECTED = 0.01 * PROC_STD


def f1_time(S, out):
    """Mean sigma per channel against frame index, log-x so the 5-frame collapse and the
    39,800-frame plateau both fit on one axis. Dashed line = the injected level."""
    import matplotlib.pyplot as plt
    fig, ax = ps.figure(rows_h=2.8)
    t = np.arange(1, len(S))
    for c in range(4):
        m = S[1:, c].reshape(len(t), -1).mean(axis=1)
        line, = ax.plot(t, m, lw=1.1, label=CH[c])
        ax.axhline(INJECTED[c], color=line.get_color(), ls=":", lw=0.8, alpha=0.7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("frame  (1 Hz, one full day)")
    ax.set_ylabel(r"ensemble spread  $\sigma$")
    ax.set_title(r"$\sigma$ hits a floor in ~5 frames and never moves again"
                 "\n(dotted = the noise injected per step)")
    ax.legend(ncol=4, frameon=False, loc="upper right")
    ps.save(fig, out)


def f2_space(S, X, Est, out, frame, ch=0):
    """One frame, three panels: the truth has structure, the error has structure, sigma does not.

    Every panel is divided by its OWN mean and drawn on ONE shared colour scale, so a colour
    means the same thing everywhere: "this many times the field's own average". Auto-scaling each
    panel to its own range would stretch sigma's 5% wobble across the full colormap and make a
    flat field look structured -- the opposite of the truth. CV (std/mean within the frame) is
    printed under each title; that is the claim, quantified."""
    import matplotlib.pyplot as plt
    # Two channels, deliberately: density is where sigma has the MOST spatial structure
    # (whole-day CV 0.455) and vx a typical one (0.142). Showing only the flattest channel
    # would be picking the best case for the claim.
    fig, axes = ps.figure(ncols=4, nrows=2, rows_h=2.3)
    for row, c in enumerate((0, 1)):
        err = np.abs(Est[frame, c] - X[frame, c])
        panels = [(X[frame, c], "truth"), (err, "|error|"),
                  (S[frame, c], r"EnKF's $\sigma$")]
        for col, (img, title) in enumerate(panels):
            ax = axes[row][col]
            cv = img.std() / (np.abs(img).mean() + 1e-12)
            im = ax.imshow((img / (np.abs(img).mean() + 1e-12)).T, cmap="magma_r",
                           aspect="auto", origin="lower", vmin=0, vmax=3)
            ax.set_title(f"{title}   CV={cv:.2f}", fontsize=8, pad=4)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(CH[c], fontsize=9)
        axes[row][3].axis("off")
    cb = fig.colorbar(im, ax=[axes[0][3], axes[1][3]], fraction=0.5, pad=0.02, aspect=28)
    cb.set_label(r"value / that panel's own mean  ($\times$)", fontsize=8)
    fig.suptitle(f"frame {frame} — identical colour scale everywhere:  the field and its error "
                 r"are structured, $\sigma$ is not", fontsize=9.5, y=1.0)
    fig.subplots_adjust(top=0.86, hspace=0.35)
    ps.save(fig, out)


def f3_knob(d, out):
    """Calibration against the injection multiplier. Two metrics, both with their ideal marked:
    spread/skill should be 1, coverage of the nominal 90% interval should be 0.90."""
    import matplotlib.pyplot as plt
    rows = []
    for f in sorted(os.listdir(d)):
        if not (f.startswith("proc_scale_") and f.endswith(".json")) or "FIXED" in f:
            continue
        r = json.load(open(os.path.join(d, f)))
        s = list(r["arms"])[0]
        a = r["arms"][s]["all"]
        rows.append((float(s), a["spread_skill"], a["coverage"]["90"]))
    rows.sort()
    x = [r[0] for r in rows]
    fig, axes = ps.figure(ncols=2, rows_h=2.6)
    for ax, idx, ylab, ideal, ttl in (
            (axes[0], 1, "spread / skill", 1.0, r"mean $\sigma$ / RMSE"),
            (axes[1], 2, "coverage of the 90% interval", 0.90, "how much truth it contains")):
        ax.plot(x, [r[idx] for r in rows], "o-", color=ps.METHOD_COLORS["EnKF"], lw=1.4)
        ax.axhline(ideal, ls="--", lw=0.9, color="0.4")
        ax.text(x[0], ideal, " ideal", va="bottom", ha="left", fontsize=7, color="0.4")
        ax.set_xscale("log"); ax.set_xlabel("process-noise injection multiplier")
        ax.set_ylabel(ylab); ax.set_title(ttl)
        ax.annotate("shipped\n(hard-coded 0.01)", xy=(x[0], [r[idx] for r in rows][0]),
                    xytext=(0.22, 0.75), textcoords="axes fraction", fontsize=7,
                    arrowprops=dict(arrowstyle="->", lw=0.7, color="0.4"))
    fig.suptitle("Turn the hard-coded constant up 100x and the calibration comes back", y=1.04)
    ps.save(fig, out)


def f4_methods(out):
    """CRPS divided by each method's OWN constant-sigma null model. The ratio is the only
    cross-method-comparable form: DINCAE lives in log1p + per-channel standardised space, so raw
    CRPS is not comparable, but numerator and denominator share that space and it cancels.
    >1 means sigma carries less information than a constant of the right magnitude."""
    # One definition of the method comparison (plot_why_sigma_fails.fig_c), so the two
    # figure sets cannot disagree about which models and cells they show.
    from methods.enkf.checks.plot_why_sigma_fails import fig_c
    fig_c(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default="atc-20130811")
    ap.add_argument("--frame", type=int, default=20000, help="frame shown in F2")
    ap.add_argument("--outdir", default=os.path.join(paths.eval_out(paths.ENKF), "figs"))
    a = ap.parse_args()
    ps.use()

    src = paths.enkf_export("enkf_k1_full")
    with np.load(os.path.join(src, f"est_{a.day}.npz")) as e:
        S, Est = e["Spread"], e["Est"]
    with np.load(os.path.join(src, f"obs_{a.day}.npz")) as o:
        X = o["X_true"][:len(S)]

    f1_time(S, os.path.join(a.outdir, "F1_sigma_vs_time.png"))
    f2_space(S, X, Est, os.path.join(a.outdir, "F2_sigma_vs_space.png"), a.frame)
    f3_knob(paths.eval_out(paths.ENKF), os.path.join(a.outdir, "F3_injection_knob.png"))
    f4_methods(os.path.join(a.outdir, "F4_vs_null_model.png"))


if __name__ == "__main__":
    main()
