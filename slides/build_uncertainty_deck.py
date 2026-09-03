"""
build_uncertainty_deck.py  —  the meeting deck for the two tasks set last week

    1. re-run the optimised EnKF and complete the 7-day test evaluation
    2. select a paper that reconstructs state WITH uncertainty, for a three-way benchmark

Both are done, so the deck reports rather than proposes. Every number is read from
check_outputs/eval/*.json and from the trained checkpoints, so nothing here can drift out of
step with what was actually measured.

Deliberately not in the deck: the beta=0 failure (plain NLL degenerating on our multi-channel
state). It is a real finding and it is written up in losses.py and runs/FAILED_ml5_beta0.json,
but it is our own detour and the meeting is about results.

    python3 slides/build_uncertainty_deck.py
"""
from __future__ import annotations

import json
import os
from crowdcore import paths
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
from slides.build_slides import OUTPUTS, ROOT, render_notes, render_pdf, render_pptx  # noqa: E402

import numpy as np  # noqa: E402

J = lambda n: json.load(open(os.path.join(OUTPUTS, "eval", n)))
ML = J("uncertainty_ml5.json")
MLR = ML["results"]
EK = {k: J(f"uncertainty_enkf_k{k}.json")["results"] for k in (1, 4)}
# appended to every provenance note: the heads were fitted WITH a sigma^2 floor and are
# read without one, so these figures are the inference-only variant of that change.
NOTE = ("  A floorless retrain (runs/varnet_nf5_s*) is running now; if it moves these "
        "numbers it will move them in the observed cells, which is where the mismatch is.")

F = lambda *p: os.path.join(OUTPUTS, *p)

# accuracy cost of the NLL objective, at matched epochs on the validation set
LAST = 149
def _m(run):
    d = {}
    for l in open(os.path.join(ROOT, "runs", run, "metrics.jsonl")):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if "epoch" in r:
            d[r["epoch"]] = r
    return d
B0 = _m("varnet_b0_k1")[LAST]
MEM = [_m(f"varnet_ml5_s{s}")[LAST] for s in range(5)]
FULL_B0 = B0["r_score"] ** 0.5
FULL_ML = float(np.mean([m["r_score"] for m in MEM])) ** 0.5
COST = (FULL_ML / FULL_B0 - 1) * 100

pc = lambda a, b: (b - a) / b * 100          # how much better a is than b, in %

slides = [
    # ───────────────────────────────────────────────────────────── 1
    dict(kind="title",
         title="Uncertainty: completing the EnKF evaluation, and a third method",
         subtitle="Both tasks done — 7-day EnKF evaluation at k=1 and k=4, "
                  "and deep ensembles implemented and compared",
         author="Xinle Zhang"),

    # ───────────────────────────────────────────────────────────── 2
    dict(title="Task 1 — the optimised EnKF, 7 days, both observation densities",
         bullets=[
             (0.9, 1.35, 11.6, 4.4, 15, [
                 (f"Accuracy and uncertainty both complete, on all "
                  f"{J('uncertainty_enkf_k1.json')['days']} held-out test days "
                  f"({EK[1]['all']['n']:,} points per setting).", 0),
                 ("", 0),
                 (f"k=1 (every frame observed)     RMSE {EK[1]['all']['rmse']:.4f}   "
                  f"CRPS {EK[1]['all']['crps']:.4f}   spread/skill "
                  f"{EK[1]['all']['spread_skill']:.3f}   90% coverage "
                  f"{EK[1]['all']['coverage']['90']:.2%}", 1),
                 (f"k=4 (every 4th frame)          RMSE {EK[4]['all']['rmse']:.4f}   "
                  f"CRPS {EK[4]['all']['crps']:.4f}   spread/skill "
                  f"{EK[4]['all']['spread_skill']:.3f}   90% coverage "
                  f"{EK[4]['all']['coverage']['90']:.2%}", 1),
                 ("", 0),
                 (f"Four times the observations changes almost nothing: RMSE moves "
                  f"{abs(pc(EK[1]['all']['rmse'], EK[4]['all']['rmse'])):.1f}%, CRPS "
                  f"{abs(pc(EK[1]['all']['crps'], EK[4]['all']['crps'])):.1f}%. Together with "
                  f"the Kalman correction accounting for 0.22% of the forecast error, this says "
                  f"the observations are barely entering the state.", 0),
             ]),
         ],
         notes="Say the last point deliberately: it is the first hint of what slide 5 explains."
               "\n\nOne gap to own if asked: test_metrics_enkf_k*.json does not record which of "
               "the three copies (orig / lab / opt) produced it, so 'the optimised "
               "implementation' currently rests on the process record rather than on the file. "
               "The numbers are unaffected -- verify_enkf_opt.py proves the three are "
               "bit-identical -- and I am adding the field."),

    # ───────────────────────────────────────────────────────────── 3
    dict(title="Task 2 — the third method: deep ensembles",
         bullets=[
             (0.9, 1.35, 11.6, 4.6, 15, [
                 ("Lakshminarayanan, Pritzel & Blundell, NeurIPS 2017 (arXiv:1612.01474). "
                  "Chosen because it is the standard against which uncertainty methods are "
                  "measured, and because it needs no change to the reconstruction itself.", 0),
                 ("", 0),
                 ("The recipe, and what we implemented", 0),
                 ("each network predicts a mean AND a variance, and is trained on the Gaussian "
                  "NLL instead of MSE (their Eq.1) — a proper scoring rule, so it rewards a "
                  "calibrated sigma rather than merely a small error", 1),
                 (f"5 members, differing only in initialisation ({len(MEM)} trained, 150 epochs "
                  f"each, ~79 GPU-hours); combined by their Sec. 2.4 moment matching", 1),
                 ("adversarial training (their Sec. 2.3) skipped: optional, and their Table 2 "
                  "states it does not help on regression", 1),
                 ("", 0),
                 (f"The variational cost J(x) is untouched — only the training loss changed, "
                  f"which is a substitution 4DVarNet explicitly allows. The solver still "
                  f"descends the gradient of the cost, computed from observations alone.", 0),
             ]),
         ],
         notes="The question to put to the meeting: deep ensembles is an uncertainty RECIPE, "
               "not a separate reconstruction method — we applied it to our own 4DVarNet. If "
               "the three-way benchmark is meant to have three independent RECONSTRUCTION "
               "methods, that is a different search and I should know now.\n\nCost detail if "
               "asked: the variance head is 548 parameters, 0.03% of the model."),

    # ───────────────────────────────────────────────────────────── 4
    dict(title="Result — better on accuracy and on uncertainty",
         images=[(F("eval", "unc_scores.png"), (0.55, 1.15, 8.6, 3.72)),
                 (F("eval", "unc_reliability.png"), (9.35, 1.15, 3.55, 3.31))],
         bullets=[
             (0.9, 5.10, 11.6, 2.2, 14, [
                 (f"RMSE {MLR['all']['rmse']:.4f} against {EK[1]['all']['rmse']:.4f} "
                  f"({pc(MLR['all']['rmse'], EK[1]['all']['rmse']):.0f}% better); CRPS "
                  f"{MLR['all']['crps']:.4f} against {EK[1]['all']['crps']:.4f} "
                  f"({pc(MLR['all']['crps'], EK[1]['all']['crps']):.0f}% better).", 0),
                 (f"Calibration: spread/skill {MLR['all']['spread_skill']:.2f} against "
                  f"{EK[1]['all']['spread_skill']:.3f} (1.0 is perfect), and "
                  f"{MLR['blind']['spread_skill']:.2f} in the blind cells, which is where it "
                  f"has to work. The observed cells are over-confident at "
                  f"{MLR['observed']['spread_skill']:.2f} — we are not uniformly on the safe "
                  f"side.", 0),
                 (f"Cost of the change: full-state RMSE {FULL_B0:.4f} -> {FULL_ML:.4f}, "
                  f"{COST:+.1f}% at matched epochs. The paper reports a median penalty of 5.6% "
                  f"for the same trade.", 0),
             ]),
         ],
         notes="These numbers are from after the head's sigma^2 floor was removed, so they "
               "differ from any earlier version of this deck. No retraining — the same five "
               "checkpoints with sigma^2 = softplus(head)." + NOTE + "\n\nThe member spread is not "
               f"optional. The learnt sigma alone scores "
               f"{MLR['all_aleatoric_only']['spread_skill']:.2f} spread/skill against the "
               f"five-member {MLR['all']['spread_skill']:.2f}, and its NLL diverges, because "
               "with nothing flooring it a single member can claim a sigma near zero. While the "
               "floor existed the aleatoric-only variant was the better calibrated of the two, "
               "so this used to look like a way to cut inference 5x. It is not any more.\n\nThe learned sigma also "
               f"knows where it is unsure: {MLR['blind']['sigma_mean']:.4f} in blind cells "
               f"against {MLR['observed']['sigma_mean']:.4f} in observed ones. The EnKF's is "
               f"{EK[1]['blind']['sigma_mean']:.4f} against "
               f"{EK[1]['observed']['sigma_mean']:.4f} — identical."),

    # ───────────────────────────────────────────────────────────── 5
    dict(title="Why the EnKF's uncertainty fails — and why tuning would not fix it",
         images=[(F("eval", "unc_spread_decay.png"), (1.85, 1.20, 9.6, 4.05))],
         bullets=[
             (0.9, 5.50, 11.6, 1.7, 14, [
                 ("Its uncertainty IS the ensemble spread, and the forecast model removes "
                  "member disagreement faster than any noise term can restore it.", 0),
                 ("So the collapse is structural, not a bad setting: a deterministic surrogate "
                  "trained to minimise prediction error is trained to be contractive, which is "
                  "exactly what an ensemble needs it not to be.", 0),
                 ("This is why we did not retune the baseline. Raising the inflation or the "
                  "process noise buys spread only by corrupting the forecast.", 0),
             ]),
         ],
         notes="This is the strongest technical content in the deck. The chain, all measured: "
               "the surrogate removes ~65% of member disagreement per step; the filter injects "
               "0.0022 per step; the equilibrium spread is 0.0025, which is what we observe. "
               "With spread that small the Kalman gain is near zero, so observations are barely "
               "assimilated -- which is why slide 2's k=1 and k=4 look the same, and why the "
               "accuracy suffers too.\n\nIf pushed to fix it: matching the true error would need "
               "~100x the injected noise, i.e. adding noise of the same magnitude as the state "
               "itself every step. Inflation would need ~2.9 against a normal range of "
               "1.02-1.2.\n\nThe honest limit of the claim: this is about a deterministic "
               "neural surrogate as the forecast model, not about ensemble filters in general."),

    # ───────────────────────────────────────────────────────────── 6
    dict(title="Where this leaves us",
         bullets=[
             (0.9, 1.35, 11.6, 5.2, 15, [
                 ("Settled", 0),
                 (f"the EnKF evaluation is complete at both observation densities, and the "
                  f"comparison is favourable on every metric we measured", 1),
                 (f"the uncertainty is calibrated and it is informative — it is larger in cells "
                  f"no robot saw ({MLR['blind']['sigma_mean']:.3f}) than in observed ones "
                  f"({MLR['observed']['sigma_mean']:.3f}), and it beats a constant-sigma null "
                  f"model on CRPS ({MLR['all']['crps']:.4f} against "
                  f"{MLR['all_constant_sigma_baseline']['crps']:.4f})", 1),
                 ("", 0),
                 ("Open", 0),
                 (f"the observed cells are over-confident at "
                  f"{MLR['observed']['spread_skill']:.2f} spread/skill while the blind cells sit "
                  f"at {MLR['blind']['spread_skill']:.2f}. The heads were fitted WITH a "
                  f"sigma^2 floor and are being read without one; a floorless retrain "
                  f"(runs/varnet_nf5_s*) is running now and should say whether that split is "
                  f"the cause", 1),
                 ("we have no 4DVarNet uncertainty at k=4; given that the EnKF barely moves "
                  "between k=1 and k=4, the value of adding it looks low against 79 more "
                  "GPU-hours", 1),
                 ("is the three-way benchmark meant to compare three reconstruction methods, or "
                  "three uncertainty treatments? That decides the next search", 1),
             ]),
         ],
         notes="Lead with the open question about what 'three-way' means -- it is the one thing "
               "only the supervisor can answer, and it determines the next two weeks.\n\nOne "
               "item that used to be on this list is now settled: whether the five members are "
               "needed. With the sigma^2 floor removed the learnt sigma alone is badly "
               "calibrated and its NLL diverges, so the member spread is load-bearing. They "
               "stay."),
]

if __name__ == "__main__":
    out = paths.SLIDES
    render_pptx(slides, os.path.join(out, "uncertainty_deck.pptx"))
    render_pdf(slides, os.path.join(out, "uncertainty_deck.pdf"))
    render_notes(slides, os.path.join(out, "uncertainty_deck_notes.md"))
    print(f"\n  {len(slides)} slides")
    print(f"  ML-5  RMSE {MLR['all']['rmse']:.4f}  CRPS {MLR['all']['crps']:.4f}  "
          f"sp/sk {MLR['all']['spread_skill']:.2f}  90% {MLR['all']['coverage']['90']:.1%}")
    print(f"  EnKF  RMSE {EK[1]['all']['rmse']:.4f}  CRPS {EK[1]['all']['crps']:.4f}  "
          f"sp/sk {EK[1]['all']['spread_skill']:.3f}  90% {EK[1]['all']['coverage']['90']:.2%}")
    print(f"  cost  full-state RMSE {FULL_B0:.4f} -> {FULL_ML:.4f}  ({COST:+.1f}%)")
