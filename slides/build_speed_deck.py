"""
build_speed_deck.py  —  the inference-speed meeting deck
========================================================

Answers the supervisor's question from last time:

    "If 4DVarNet's query time turns out to be too slow, using it directly may be
     impractical; the more sensible route is to take useful elements from it into our
     existing system."

Every number comes from ONE benchmark run: check_outputs/eval/bench_speed_cpu.json,
200 frames of atc-20130811, six measurements run one at a time on one node, cycling through
all six each round rather than finishing one before starting the next; three timed repeats
each, max run-to-run spread 2.0%.

Three things the deck must not overstate, and does not:
  * the headline is NOT "15000x faster than the EnKF" — that would compare a GPU against an
    unoptimised CPU implementation, unfair twice over. The like-for-like number is 52x, same
    CPU, both implementations optimised;
  * the EnKF's 2631 ms is stated as an IMPLEMENTATION cost, not an algorithmic one, and the
    comparison uses the 131 ms vectorised figure;
  * the node was shared, not exclusive — controlled for by interleaving, and the residual
    spread is reported rather than hidden.

Slide 6 (test-set accuracy) is intentionally a placeholder: the per-epoch numbers quoted
so far were measured on the TRAINING split (train_varnet.py evaluates X[:n_eval] of
--split train), so they cannot be reported as test results. The real test-set evaluation
is running; this slide gets filled in when it lands.

    python3 slides/build_speed_deck.py
"""
from __future__ import annotations

import glob
import json
import os
from crowdcore import paths
import sys

from slides.build_slides import OUTPUTS, ROOT, render_notes, render_pdf, render_pptx

import numpy as np

B = json.load(open(os.path.join(OUTPUTS, "eval", "bench_speed_cpu.json")))
# Test-set accuracy, read from the eval jsons so the slide can never drift from the run.
# RMSE is taken per day and THEN averaged: sqrt of a mean is not the mean of sqrts.
# Observation coverage, computed from the exported masks over ALL 7 test days rather than
# hand-written: an earlier 43.9% came from a 400-frame sample and was wrong (46.9%).
# Denominator is all 36x12 cells, matching the denominator of the RMSE reported here.
# The robots follow a fixed seeded A* route over one fixed map, so coverage is identical
# day to day (verified: 11.7% and 46.9% on every one of the 7 days).
COV = {}
for _k, _d in ((4, "enkf_k4_full"), (1, "enkf_k1_full")):
    _fs = sorted(glob.glob(os.path.join(OUTPUTS, _d, "obs_*.npz")))
    _c = [float(np.load(_f)["Omega"].mean()) for _f in _fs]
    COV[_k] = float(np.mean(_c)) if _c else float("nan")

ACC, TRAIN = {}, {}
for _k in (1, 4):
    # r_score in metrics.jsonl is the full-state MSE on the per-epoch eval subset, which is
    # taken from the TRAINING split — used here only to show how much of the gain survives
    # on unseen days, never as a reported result.
    _rows = [json.loads(_l) for _l in open(os.path.join(ROOT, f"runs/varnet_b0_k{_k}/metrics.jsonl"))]
    TRAIN[_k] = min(_rows, key=lambda r: r["rec_unobs_mse"])["r_score"]
    _p = os.path.join(OUTPUTS, "eval", f"test_metrics_b0_k{_k}.json")
    _d = json.load(open(_p))
    _fu = np.array([r["full_mse"] for r in _d["per_day"]], float)   # WHOLE state, all cells
    _bl = np.array([r["blind_mse"] for r in _d["per_day"]], float)  # unobserved cells only
    ACC[_k] = dict(mse=_fu.mean(), mse_std=_fu.std(),
                   rmse=np.sqrt(_fu).mean(), rmse_std=np.sqrt(_fu).std(),
                   blind_rmse=np.sqrt(_bl).mean(), blind_mse=_bl.mean(),
                   epoch=_d["epoch"],
                   days=len(_d["per_day"]),
                   frames=sum(r["n_windows"] for r in _d["per_day"]) * _d["dT"])
# EnKF, same test days, same frames per day, same physical clip as the 4DVarNet rows.
ENKF4 = json.load(open(os.path.join(OUTPUTS, "eval", "test_metrics_enkf_k4.json")))
R, HW = B["runs"], B["hw"]
ms = lambda k: R[k]["per_frame_s"] * 1000
RT = 1000.0                                        # real-time budget: 1 frame per second of data

ORIG, OPT, VCPU, VGPU = ms("enkf_orig"), ms("enkf_opt"), ms("varnet_k1_cpu"), ms("varnet_k1_gpu")
SPEEDUP_IMPL = ORIG / OPT                          # our vectorisation, bit-identical output
SPEEDUP_METHOD = OPT / VCPU                        # like-for-like, same CPU
HWLINE = f"{HW['cpu']} · {HW['cores_avail']} cores · {HW['gpu']} · node {HW['node']}"

F = lambda *p: os.path.join(OUTPUTS, *p)

slides = [
    # ───────────────────────────────────────────────────────────── 1
    dict(kind="title",
         title="4DVarNet inference speed",
         subtitle=f"Is the query time a problem?   —   measured: {VGPU:.2f} ms per frame, "
                  f"{RT / VGPU:,.0f}x real time",
         author="Xinle Zhang"),

    # ───────────────────────────────────────────────────────────── 4
    dict(title="Inference cost per frame",
         images=[(F("eval", "speed_bars.png"), (0.55, 1.25, 12.2, 4.5))],
         bullets=[
             (0.9, 5.95, 11.6, 1.3, 15, [
                 (f"On an H200:  {VGPU:.2f} ms/frame  =  {1000 / VGPU:,.0f} frames per second  "
                  f"=  {RT / VGPU:,.0f}x real time", 0),
                 (f"On the SAME CPU:  {SPEEDUP_METHOD:.0f}x faster than the EnKF "
                  f"({VCPU:.2f} vs {OPT:.0f} ms) — this is not a GPU effect", 0),
                 (f"The EnKF's original implementation is slower than real time "
                  f"({ORIG / RT:.1f} s to process 1 s of data)", 0),
             ]),
         ],
         notes="Log axis — the numbers span four orders of magnitude. The red line is the "
               "real-time budget. Lead with the 52x same-CPU number, not the GPU one.\n\n"
               "If asked how it was measured: one job, one node, the same 200 frames for "
               "every method; the six measurements run serially and INTERLEAVED (A B C A B C "
               "A B C) rather than blocked, so a shift in the node's background load spreads "
               "over all methods instead of landing on one and reading as a method "
               "difference; one untimed warm-up each; 3 timed repeats; max run-to-run spread "
               "2.0%. The node was shared, not exclusive — no node in batch-csl (0 of 48 "
               "idle) or gpu-h200 was free, and interleaving plus repeats converts that "
               "contention from a hidden bias into a reported spread. Observation-matrix "
               "construction is outside the timed region: that is our driver's bookkeeping, "
               "not the filter's cost."),

    # ───────────────────────────────────────────────────────────── 5
    dict(title="The EnKF's cost is its implementation, not its algorithm",
         bullets=[
             (0.9, 1.6, 11.5, 2.0, 17, [
                 ("Profiling one assimilation step — where the time goes:", 0),
                 ("96%   building the localization matrix — a table of "
                  "distance weights", 1),
                 ("2%   the Kalman update itself, the actual filtering", 1),
             ]),
             (0.9, 4.1, 11.5, 2.4, 17, [
                 (f"We vectorised that one function: {SPEEDUP_IMPL:.0f}x faster "
                  f"({ORIG:,.0f} -> {OPT:.0f} ms/frame), output bit-identical.", 0),
                 (f"So the comparison uses {OPT:.0f} ms, not {ORIG:,.0f} ms. "
                  f"The original project was not modified.", 0),
             ]),
         ],
         notes="Raise this before they find it in the code. Framing: a useful finding for "
               "their project — a 20x speed-up with unchanged results — not a criticism. And "
               "it is why the headline comparison uses the 131 ms figure.\n\n"
               "Detail if asked: the localization matrix is built by a four-deep Python loop "
               "over 252 observations x 4 channels x 36 x 12 = 435,456 iterations, and "
               "update() calls it twice per step with the same argument. It ran at 3.7 of 16 "
               "cores — serial Python, so more CPUs would not have helped. Verification: same "
               "call on real observed cells through both copies, np.array_equal, plus a "
               "full-day run. We vendored a read-only reference copy (enkf_lab) and changed a "
               "separate experiment copy (enkf_opt) inside our own repository; "
               "Partial_observation itself is untouched."),

    # ───────────────────────────────────────────────────────────── 7  (placeholder)
    dict(title="Reconstruction accuracy — held-out test set",
         bullets=[
             (0.9, 1.3, 11.6, 1.0, 14, [
                 (f"{ACC[1]['days']} held-out days (2013-08-11 to 2013-09-29), "
                  f"{ACC[1]['frames']:,} frames the models never trained on. Same days, same "
                  f"frames and the same physical bounds for every row. RMSE over the whole "
                  f"state — the convention the EnKF project uses for its own filter.", 0)]),
             # a table, not a list: level -1 suppresses the bullet glyph
             (1.5, 2.7, 4.6, 0.4, 15, [("", -1)]),
             (6.5, 2.7, 2.6, 0.4, 15, [("4DVarNet", -1)]),
             (9.4, 2.7, 2.6, 0.4, 15, [("EnKF", -1)]),
             (1.5, 3.3, 4.6, 1.4, 15, [
                 ("every 4th frame observed", -1),
                 ("every frame observed", -1)]),
             (6.5, 3.3, 2.6, 1.4, 15, [
                 (f"{ACC[4]['rmse']:.3f}", -1),
                 (f"{ACC[1]['rmse']:.3f}", -1)]),
             (9.4, 3.3, 3.0, 1.4, 15, [
                 (f"{ENKF4['rmse_mean']:.3f}", -1),
                 ("still running", -1)]),
         ],
         notes="If asked whether the gaps are reliable: on every one of the 7 days the "
               "ordering is the same, with no exception — 4DVarNet beats the filter on all "
               "seven, and every-frame beats every-4th-frame on all seven. Day-to-day spread "
               f"is {ACC[4]['rmse_std']:.3f} / {ACC[1]['rmse_std']:.3f} / "
               f"{ENKF4['rmse_std']:.3f} RMSE respectively, several times smaller than the "
               "gaps themselves.\n\n"
               f"Reading the table out loud, if useful: at the same observation density "
               f"4DVarNet's error is {(1 - ACC[4]['mse'] / ENKF4['mse_mean']) * 100:.0f}% "
               f"lower than the filter's (MSE {ENKF4['mse_mean']:.4f} vs {ACC[4]['mse']:.4f}); "
               f"observing every frame lowers it a further "
               f"{(1 - ACC[1]['mse'] / ACC[4]['mse']) * 100:.0f}% at no inference cost.\n\n"
               "The EnKF's every-frame run is genuinely still going: its original "
               "implementation needs ~4.8 s per frame, so one test day is ~53 hours. Do not "
               "present the empty cell as a result either way.\n\n"
               f"If asked whether this is overfitting: the same k comparison on the TRAINING "
               f"days gives {(1 - TRAIN[1] / TRAIN[4]) * 100:.0f}% against "
               f"{(1 - ACC[1]['mse'] / ACC[4]['mse']) * 100:.0f}% here, so the gain carries "
               f"over to unseen days.\n\n"
               f"If asked how much gets observed: {COV[4]:.1%} of all 36x12 cells at every "
               f"4th frame, {COV[1]:.1%} at every frame; relative to the 290 walkable cells "
               "it is 17.5% and 69.8%. The robots run a fixed seeded A* route, so coverage is "
               "identical on every test day.\n\n"
               "If asked about the blind zone specifically: restricted to cells never "
               f"observed, 4DVarNet goes {ACC[4]['blind_rmse']:.3f} -> "
               f"{ACC[1]['blind_rmse']:.3f} — a smaller gain, since full-state RMSE also "
               "improves simply by moving cells into the observed set.\n\n"
               "Earlier per-epoch figures came from the training split and are not shown."),

    # ───────────────────────────────────────────────────────────── 8
    dict(title="Where this goes next",
         bullets=[
             (0.9, 1.5, 11.5, 5.0, 17, [
                 ("Done: denser observations", 0),
                 (f"moved from observing every 4th frame to every frame — RMSE "
                  f"{ACC[4]['rmse']:.3f} -> {ACC[1]['rmse']:.3f} on the held-out days, "
                  f"{(1 - ACC[1]['mse'] / ACC[4]['mse']) * 100:.0f}% lower error", 1),
                 ("costs nothing at inference — the solver's runtime is unchanged", 1),
                 ("", 0),
                 ("Next: more model capacity", 0),
                 ("the GENN prior currently has 13,900 parameters", 1),
                 ("the paper's Lorenz-96 setup uses roughly 50,000 — there is room", 1),
             ]),
         ],
         notes="Keep this short. The speed result is the message today; this slide is just "
               "to show the direction."),
]

if __name__ == "__main__":
    out = paths.SLIDES
    render_pptx(slides, os.path.join(out, "speed_deck.pptx"))
    render_pdf(slides, os.path.join(out, "speed_deck.pdf"))
    render_notes(slides, os.path.join(out, "speed_deck_notes.md"))
    print(f"\nheadline numbers:")
    print(f"  EnKF original    {ORIG:9.1f} ms/frame   {RT / ORIG:7.2f}x real time")
    print(f"  EnKF vectorised  {OPT:9.1f} ms/frame   {RT / OPT:7.2f}x real time")
    print(f"  4DVarNet CPU     {VCPU:9.2f} ms/frame   {RT / VCPU:7.0f}x real time")
    print(f"  4DVarNet GPU     {VGPU:9.2f} ms/frame   {RT / VGPU:7.0f}x real time")
    print(f"  implementation speed-up {SPEEDUP_IMPL:.1f}x   like-for-like {SPEEDUP_METHOD:.0f}x")
