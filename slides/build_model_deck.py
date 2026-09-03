"""
build_model_deck.py  —  the model-architecture presentation
===========================================================

Scope, after trimming: explain the PRIOR in detail, then show where the results stand.

The earlier version also had a framework-overview slide, a solver slide, and two
"why it is built this way" summary slides. Those were dropped on purpose — the prior is
the part that needs explaining, and it needs more than one slide to do it, so the single
GENN slide became four: what Φ is for, ψ, φ, and the two-scale composition. Each has a
diagram plus the reasoning that cannot be read off the code.

Every number is read from the live checkpoint (runs/varnet_a2_k1) or from the evaluation and
benchmark jsons, so a slide and a figure can never disagree about what was trained. The four
prior diagrams come from checks/plot_genn_detail.py, which loads the same checkpoint; the two
result figures come from checks/plot_results.py.

    python3 slides/build_model_deck.py
"""
from __future__ import annotations

import json
import os
from crowdcore import paths
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
from slides.build_slides import OUTPUTS, ROOT, render_notes, render_pdf, render_pptx  # noqa: E402

import numpy as np  # noqa: E402
from methods.varnet.checks.model_io import load_solver  # noqa: E402

# One model throughout: architecture slides, accuracy and speed all come from B0 — the
# config.yaml defaults (hidden 32, kt 3), so psi's kernel is 3x3x3. Keep it in step with the
# CKPT constant in checks/plot_genn_detail.py, which is where the figures read their shapes.
RUN = "b0"
ARCH_RUN = f"runs/varnet_{RUN}_k1"
S, A, CK = load_solver(os.path.join(ROOT, ARCH_RUN, "varnet_best.pt"), "cpu")
NP = lambda m: sum(p.numel() for p in m.parameters())
C, T, H, W = 4, A["dT"], 36, 12
HID, KT, KH, KW = A["hidden"], A.get("kt", 3), A.get("kh", 3), A.get("kw", 3)
BR = S.phi.branch_fine
P_TOT, P_PRIOR, P_SOLV = NP(S), NP(S.phi), NP(S.grad_net)

J = lambda n: json.load(open(os.path.join(OUTPUTS, "eval", n)))
VAR = {}
for k in (1, 4):
    d = J(f"test_metrics_{RUN}_k{k}.json")
    fu = np.array([r["full_mse"] for r in d["per_day"]], float)
    VAR[k] = dict(rmse=np.sqrt(fu).mean(), days=len(d["per_day"]),
                  frames=sum(r["n_windows"] for r in d["per_day"]) * d["dT"])
ENK = {k: J(f"test_metrics_enkf_k{k}.json") for k in (1, 4)}
SPD = J(f"bench_speed_{RUN}.json")
SPL = J("enkf_time_split.json")
ms = lambda k: SPD["runs"][k]["per_frame_s"] * 1000

F = lambda *p: os.path.join(OUTPUTS, *p)

slides = [
    # ───────────────────────────────────────────────────────────── 1
    dict(kind="title",
         title="The dynamical prior  Φ",
         subtitle=f"{P_PRIOR:,} parameters   ·   {P_PRIOR / P_TOT * 100:.1f}% of the model",
         author="Xinle Zhang"),

    # ───────────────────────────────────────────────────────────── 2
    dict(title="What the prior is for",
         images=[(F("eval", "genn_role.png"), (0.37, 1.30, 12.6, 3.77))],
         bullets=[
             (0.9, 5.45, 11.6, 1.7, 15, [
                 ("Not a forecaster: given x, it says what each frame SHOULD look like, judged "
                  "from the frames around it.", 0),
                 ("Plausible x → x ≈ Φ(x), term small.  Impossible x → term large, solver "
                  "pushed away.", 0),
                 ("The only route to cells no robot saw — the observation term is zero there.", 0),
             ]),
         ],
         notes="Say the last bullet out loud even if nobody asks: it is the reason the prior "
               "matters at all for our problem. Roughly 38% of walkable cells carry no "
               "observation in a given frame, and their reconstruction comes only from Φ.\n\n"
               "α_obs and α_reg are learned alongside everything else, so the balance between "
               "the two terms is not hand-tuned either."),

    # ───────────────────────────────────────────────────────────── 3
    dict(title="Inside Φ, part 1 — ψ, the one convolution that sees neighbours",
         images=[(F("eval", "genn_psi.png"), (0.37, 1.22, 12.6, 4.07))],
         bullets=[
             (0.9, 5.65, 11.6, 1.5, 15, [
                 ("The output at a point never reads the input at that point.", 0),
                 ("Without it, Φ = identity drives ‖x − Φ(x)‖² to zero for EVERY x — the prior "
                  "would constrain nothing.", 0),
                 ("The mask is re-applied every forward pass, so the optimiser cannot undo it.", 0),
             ]),
         ],
         notes="This is the single most important slide in the deck — everything else in the "
               "prior's design exists to protect this property.\n\nIf asked how far it reaches: "
               f"kernel {KT}x{KH}x{KW} means +-{KT // 2} frames in time and +-{KH // 2} cells "
               f"in space. Because phi is pointwise, that is the ENTIRE receptive field of one "
               f"branch."),

    # ───────────────────────────────────────────────────────────── 4
    dict(title="Inside Φ, part 2 — φ, pointwise on purpose",
         images=[(F("eval", "genn_phi.png"), (0.37, 1.40, 12.6, 3.57))],
         bullets=[
             (0.9, 5.45, 11.6, 1.7, 15, [
                 ("Paper Sec. 3.2 requires it: every kernel size is 1.", 0),
                 ("A wider kernel would pull in a neighbour — which does depend on x(s) — and "
                  "make the identity reachable again.", 0),
                 ("So all spatial and temporal context comes from ψ alone. Widening φ adds "
                  "capacity, not reach.", 0),
             ]),
         ],
         notes="The order in the code is psi -> ReLU -> Conv(1x1x1) -> ReLU -> Conv(1x1x1); "
               f"the {NP(BR.phi):,} parameters are the two convolutions.."),

    # ───────────────────────────────────────────────────────────── 5
    dict(title="Inside Φ, part 3 — two scales  (Eq. 10)",
         images=[(F("eval", "genn_twoscale.png"), (0.37, 1.18, 12.6, 4.16))],
         bullets=[
             (0.9, 5.65, 11.6, 1.5, 15, [
                 (f"Each branch is a full ψ + φ ({NP(BR):,} params), both zero-centre. "
                  f"Prior total {P_PRIOR:,}.", 0),
                 (f"Φ₁ runs AT {H // 2}×{W // 2}, so its {KH}×{KW} kernel covers "
                  f"{KH * 2}×{KW * 2} fine cells — genuinely different scales.", 0),
                 ("Φ₂ takes x itself, not a high-pass residual; Up is learned, not "
                  "interpolation.", 0),
             ]),
         ],
         notes="The mistake we made first and then fixed: blurring x, feeding blur and "
               "high-pass to two FULL-resolution branches. Both branches then had the same "
               "receptive field in real terms, which defeats the purpose. Now the coarse "
               "branch genuinely runs on the smaller grid.\n\nCaveat if pressed: with two "
               "scales the zero-centre property is no longer exact. Pooling folds x(s) into a "
               "coarse cell and Up spreads it back, so d|Phi(x)(s)|/dx(s) is 2e-2 on this "
               "model — 2% of what the identity would give, against exactly 0 for a single "
               "branch. Still far from reachable, but say 'effectively excluded', not "
               "'excluded'. It is a property of Eq.10 itself; the paper does not quantify it. "
               "Measured by checks/check_paper_conformance.py."),

    # ───────────────────────────────────────────────────────────── 6
    # One figure per slide. The two used to share a slide and it was too crowded to read
    # either of them; they also carry different arguments, so they do not belong together.
    dict(title="Where it stands — accuracy",
         images=[(F("eval", "results_accuracy.png"), (2.30, 1.15, 8.74, 4.60))],
         bullets=[
             (0.9, 6.00, 11.6, 1.1, 15, [
                 (f"{VAR[1]['days']} held-out test days, {VAR[1]['frames']:,} frames. Same "
                  f"frames and bounds for both methods; RMSE over the whole state.", 0),
             ]),
         ],
         notes="More observations help us far more than they help the filter (0.188 -> 0.167 "
               "against 0.243 -> 0.239).\n\nIf asked why the filter barely improves: its "
               "ensemble spread is about 90x smaller than its actual error, so the Kalman gain "
               "is near zero and observations are largely ignored — measured, not inferred. "
               "Mention only if asked, and add that it is their configuration and we did not "
               "change it."),

    # ───────────────────────────────────────────────────────────── 7
    dict(title="Where it stands — inference cost",
         images=[(F("eval", "results_speed.png"), (1.65, 1.15, 10.03, 4.60))],
         bullets=[
             (0.9, 6.00, 11.6, 1.1, 15, [
                 (f"One node, {SPD['hw']['cores_avail']} cores, no GPU either side. EnKF is its "
                  f"optimised implementation, bit-identical to the original.", 0),
             ]),
         ],
         notes="If asked how solid the absolute numbers are, say it plainly: the EnKF's "
               "per-frame cost has measured anywhere from 130 to 290 ms across sessions on "
               "identical code, because every batch-csl node is shared and its k=1 pass runs "
               "for minutes, so it absorbs whatever else is on the node. 4DVarNet's is stable "
               "(3.3-3.8 ms) because each pass is seconds. The ratio between the methods is "
               "what survives that, and it is 40-80x in every session. No exclusive node was "
               "available to do better.\n\nThe structural point the figure makes: the filter's cost tracks observation "
               f"density because it pays a Kalman analysis per observed frame, while its "
               f"forecast runs every frame either way "
               f"({SPL['runs']['k1']['forecast_ms']:.0f} vs "
               f"{SPL['runs']['k4']['forecast_ms']:.0f} ms). The solver's cost does not move at "
               f"all ({ms('varnet_k1'):.2f} vs {ms('varnet_k4'):.2f} ms) — it runs a fixed "
               f"{S.n_iter} iterations over the whole dense window.\n\nSo denser observations "
               f"widen the gap in speed as well as in accuracy."),
]

if __name__ == "__main__":
    out = paths.SLIDES
    render_pptx(slides, os.path.join(out, "model_deck.pptx"))
    render_pdf(slides, os.path.join(out, "model_deck.pdf"))
    render_notes(slides, os.path.join(out, "model_deck_notes.md"))
    print(f"\n  {len(slides)} slides   every number from runs/varnet_{RUN}_k* "
          f"(epoch {CK.get('epoch')}, hidden {HID}, kt {KT}, kernel {KT}x{KH}x{KW})")
    print(f"  prior    {P_PRIOR:,} = branch {NP(BR):,} x2 + up {NP(S.phi.up)}   "
          f"(psi {NP(BR.psi):,} + phi {NP(BR.phi):,} per branch)")
    print(f"  accuracy 4DVarNet {VAR[4]['rmse']:.4f} / {VAR[1]['rmse']:.4f}   "
          f"EnKF {ENK[4]['rmse_mean']:.4f} / {ENK[1]['rmse_mean']:.4f}")
    print(f"  speed    4DVarNet {ms('varnet_k1'):.2f} ms   EnKF {ms('enkf_k1'):.1f} / "
          f"{ms('enkf_k4'):.1f} ms")
