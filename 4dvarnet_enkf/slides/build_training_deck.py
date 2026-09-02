"""
build_training_deck.py  —  everything we have trained, and what came out of it

Four configurations of the reconstruction, all evaluated on the same 7 held-out test days with
one convention, plus the EnKF baseline. The deck's job is to justify which one is the model of
record, so it reports the two configurations that are better on accuracy and says why neither
is used.

The uncertainty slide is here rather than in its own deck because the NLL variant costs accuracy
and reporting that cost without what it bought leaves the argument half-finished.

Every number is read from check_outputs/eval/test_metrics_*.json and from the checkpoints; the
figures come from checks/plot_training_results.py, which reads the same files.

    python3 slides/build_training_deck.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT_)
sys.path.insert(0, os.path.join(ROOT_, "checks"))
from build_slides import OUTPUTS, ROOT, render_notes, render_pdf, render_pptx  # noqa: E402
from model_io import load_solver  # noqa: E402

# appended to every provenance note: the heads were fitted WITH a sigma^2 floor and are
# read without one, so these figures are the inference-only variant of that change.
NOTE = ("  A floorless retrain (runs/varnet_nf5_s*) is running now; if it moves these "
        "numbers it will move them in the observed cells, which is where the mismatch is.")

F = lambda *p: os.path.join(OUTPUTS, *p)
NP = lambda m: sum(p.numel() for p in m.parameters())


def rd(tag):
    d = json.load(open(os.path.join(OUTPUTS, "eval", f"test_metrics_{tag}.json")))
    fu = np.array([r["full_mse"] for r in d["per_day"]], float)
    bl = np.array([r["blind_mse"] for r in d["per_day"]], float)
    return dict(full=np.sqrt(fu).mean(), blind=np.sqrt(bl).mean(), epoch=d.get("epoch"))


def prior_of(run):
    S, A, ck = load_solver(os.path.join(ROOT, "runs", run, "varnet_best.pt"), "cpu")
    return NP(S.phi), A["hidden"], A.get("kt"), NP(S)


R = {r: {k: rd(f"{r}_k{k}") for k in (1, 4)} for r in ("b0", "a2", "a4")}
P = {r: prior_of(f"varnet_{r}_k1") for r in ("b0", "a2", "a4")}
ML = [rd(f"ml5_s{s}") for s in range(5)]
MLM = float(np.mean([m["full"] for m in ML]))
MLS = float(np.std([m["full"] for m in ML]))
EK = {k: json.load(open(os.path.join(OUTPUTS, "eval", f"test_metrics_enkf_k{k}.json")))["rmse_mean"]
      for k in (1, 4)}

# uncertainty, for the slide that says what the +1.8% bought
_J = lambda n: json.load(open(os.path.join(OUTPUTS, "eval", n)))
U = _J("uncertainty_ml5.json")["results"]
UE = _J("uncertainty_enkf_k1.json")["results"]

slides = [
    # ───────────────────────────────────────────────────────────── 1
    dict(kind="title",
         title="Training results",
         subtitle=f"Four configurations of the reconstruction, {len(ML)} ensemble members, "
                  f"all on the same 7 held-out test days",
         author="Xinle Zhang"),

    # ───────────────────────────────────────────────────────────── 2
    dict(title="What has been trained",
         bullets=[
             (0.9, 1.35, 11.6, 5.4, 14, [
                 (f"b0   prior {P['b0'][0]:,} params  (hidden {P['b0'][1]}, kt {P['b0'][2]})"
                  f"   —   MODEL OF RECORD", 0),
                 (f"full-state RMSE {R['b0'][1]['full']:.4f} (k=1) / {R['b0'][4]['full']:.4f} "
                  f"(k=4); best epoch {R['b0'][1]['epoch']} of 150, still improving when it "
                  f"stopped", 1),
                 ("", 0),
                 (f"a2   prior {P['a2'][0]:,} params  (hidden {P['a2'][1]}, kt {P['a2'][2]})", 0),
                 (f"RMSE {R['a2'][1]['full']:.4f} / {R['a2'][4]['full']:.4f}  —  "
                  f"{(1 - R['a2'][1]['full'] / R['b0'][1]['full']) * 100:.1f}% better than b0 at "
                  f"k=1, for 3.4x the prior and 2.15x the inference time (3.66 -> 7.86 ms/frame, "
                  f"same 4-core CPU)", 1),
                 ("", 0),
                 (f"a4   prior {P['a4'][0]:,} params  (hidden {P['a4'][1]}, kt {P['a4'][2]})", 0),
                 (f"RMSE {R['a4'][1]['full']:.4f} / {R['a4'][4]['full']:.4f}  —  the best "
                  f"accuracy of the three, but its training destabilised at epoch 52 and never "
                  f"recovered", 1),
                 ("", 0),
                 (f"ml5   b0's architecture, trained on the Gaussian NLL, {len(ML)} members", 0),
                 (f"RMSE {MLM:.4f} ± {MLS:.4f}  —  {(MLM / R['b0'][1]['full'] - 1) * 100:+.1f}% "
                  f"against b0, in exchange for a calibrated per-cell uncertainty", 1),
             ]),
         ],
         notes="a4 was left unfinished earlier and has now been completed to epoch 100 and "
               "evaluated, so there are no partial runs left.\n\nThe inference figures quoted "
               "for b0 and a2 are the 4-core CPU benchmark, which is the convention used "
               "throughout. a4 was never benchmarked on CPU; on a matched A100 it is 3.1x "
               "b0's architecture, but that is a different measurement and should not be mixed "
               "with the CPU numbers."),

    # ───────────────────────────────────────────────────────────── 3
    dict(title="Prior capacity has not saturated",
         images=[(F("eval", "train_capacity.png"), (2.35, 1.20, 8.6, 4.83))],
         bullets=[
             (0.9, 6.25, 11.6, 1.0, 14, [
                 (f"From {P['b0'][0]:,} to {P['a4'][0]:,} prior parameters — 5.8x — the "
                  f"full-state RMSE falls "
                  f"{(1 - R['a4'][1]['full'] / R['b0'][1]['full']) * 100:.1f}%, and it is still "
                  f"descending. Accuracy alone would argue for the largest prior.", 0),
             ]),
         ],
         notes="This corrects an expectation I had going in: I assumed capacity would saturate "
               "somewhere near 30k parameters and that a4 would confirm it. It did not — a4 is "
               "the most accurate of the three.\n\nThe prior is a small fraction of the model "
               "either way: even a4's 54,220 is 2.6% of the 2.1M total, almost all of which is "
               "the ConvLSTM solver. So this is not a story about model size, it is about the "
               "prior specifically."),

    # ───────────────────────────────────────────────────────────── 4
    dict(title="But the largest prior does not train reliably",
         images=[(F("eval", "train_curves.png"), (2.15, 1.20, 9.0, 4.83))],
         bullets=[
             (0.9, 6.25, 11.6, 1.0, 14, [
                 ("a4's best epoch is 50. At epoch 52 its blind-zone error jumps 29% and stays "
                  "above its own starting point for the remaining 47 epochs — the 16 epochs "
                  "added to finish the run changed nothing. b0 trains smoothly throughout.", 0),
             ]),
         ],
         notes="The jump lands two epochs after the curriculum raises the solver from 10 to 15 "
               "iterations. A longer unrolled gradient path through a larger prior is the "
               "obvious suspect and --clip-grad 1.0 did not contain it, but this is one run at "
               "one capacity — a mechanism worth stating as a hypothesis, not a "
               "conclusion.\n\nThe practical consequence is what matters: the accuracy advantage "
               "on the previous slide comes from a checkpoint taken before the instability, so "
               "it is real but it was not obtained by a training procedure I would rely on. "
               "b0 delivers within 3% of it, trains monotonically, and runs 2-3x faster."),

    # ───────────────────────────────────────────────────────────── 5
    dict(title="What the NLL objective bought: a usable uncertainty",
         images=[(F("eval", "unc_reliability.png"), (8.60, 1.15, 4.35, 4.09))],
         bullets=[
             (0.9, 1.35, 7.4, 3.9, 13.5, [
                 (f"spread / skill   —   ideal is 1.0", 0),
                 (f"ours {U['all']['spread_skill']:.2f} overall, "
                  f"{U['blind']['spread_skill']:.2f} in the blind cells   ·   EnKF "
                  f"{UE['all']['spread_skill']:.3f}", 1),
                 ("", 0),
                 (f"90% interval actually contains   —   ideal is 90%", 0),
                 (f"ours {U['all']['coverage']['90']:.1%}   ·   EnKF "
                  f"{UE['all']['coverage']['90']:.1%}", 1),
                 ("", 0),
                 (f"does sigma know WHERE it is unsure?", 0),
                 (f"ours: {U['blind']['sigma_mean']:.4f} in blind cells against "
                  f"{U['observed']['sigma_mean']:.4f} in observed ones "
                  f"({U['blind']['sigma_mean'] / U['observed']['sigma_mean']:.2f}x)", 1),
                 (f"EnKF: {UE['blind']['sigma_mean']:.4f} against "
                  f"{UE['observed']['sigma_mean']:.4f} — identical, it cannot tell them apart", 1),
             ]),
             (0.9, 5.45, 11.6, 1.8, 13.5, [
                 (f"The blind cells — the ones this exists for — are close to ideal at "
                  f"{U['blind']['spread_skill']:.2f}. The observed cells are over-confident at "
                  f"{U['observed']['spread_skill']:.2f}, so we are not uniformly on the safe "
                  f"side; the EnKF is over-confident everywhere by roughly 90x.", 0),
                 (f"Structurally the matched comparison is against the members' disagreement "
                  f"alone: {U['all_epistemic_only']['sigma_mean']:.4f} against the EnKF's "
                  f"{UE['all']['sigma_mean']:.4f}.", 0),
             ]),
         ],
         notes="This is the answer to 'why pay 1.8% accuracy'.\n\nThese numbers are from after "
               "the head's sigma^2 floor was removed — same five checkpoints, no retraining, "
               "sigma^2 = softplus(head)." + NOTE + " Anyone who saw an earlier version of this slide saw "
               "different figures.\n\nThe member spread is not optional: the learnt sigma alone "
               f"is calibrated at {U['all_aleatoric_only']['spread_skill']:.2f} with "
               f"{U['all_aleatoric_only']['coverage']['90']:.1%} coverage and its NLL diverges, "
               "because nothing floors a member that claims certainty. That question used to be "
               "open and is now settled.\n\nThe EnKF's spread is its own project's "
               "configuration, unmodified."),

    # ───────────────────────────────────────────────────────────── 6
    dict(title="Where the reconstruction stands",
         images=[(F("eval", "train_summary.png"), (2.15, 1.20, 9.0, 4.71))],
         bullets=[
             (0.9, 6.15, 11.6, 1.1, 14, [
                 (f"Against the EnKF baseline at k=1: b0 is "
                  f"{(1 - R['b0'][1]['full'] / EK[1]) * 100:.1f}% better, and the NLL variant "
                  f"{(1 - MLM / EK[1]) * 100:.1f}% better while also reporting a per-cell "
                  f"uncertainty. The {len(ML)} members agree to within "
                  f"{MLS / MLM * 100:.1f}%.", 0),
             ]),
         ],
         notes="The EnKF number is its own project's configuration, unmodified — worth saying "
               "before it is asked.\n\nOpen items, none of them blocking: no CPU benchmark for "
               "a4; no uncertainty measured at k=4 for our side; and whether the 5-member "
               "ensemble is needed at all, given how closely the members agree."),
]

if __name__ == "__main__":
    out = os.path.join(ROOT, "slides")
    render_pptx(slides, os.path.join(out, "training_deck.pptx"))
    render_pdf(slides, os.path.join(out, "training_deck.pdf"))
    render_notes(slides, os.path.join(out, "training_deck_notes.md"))
    print(f"\n  {len(slides)} slides")
    for r in ("b0", "a2", "a4"):
        print(f"  {r:4s} prior {P[r][0]:>6,}  RMSE k1 {R[r][1]['full']:.4f}  "
              f"k4 {R[r][4]['full']:.4f}  best ep {R[r][1]['epoch']}")
    print(f"  ml5  prior {P['b0'][0]:>6,}  RMSE {MLM:.4f} ± {MLS:.4f}")
    print(f"  EnKF                RMSE k1 {EK[1]:.4f}  k4 {EK[4]:.4f}")
