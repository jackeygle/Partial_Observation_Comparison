"""
build_meeting3_deck.py  —  joint-meeting deck, six slides.

Scope is deliberately narrow. The EnKF appears only for accuracy and for inference time; its
own uncertainty is not compared, because a fair comparison would need its inflation retuned and
this project does not change the EnKF's configuration.

Cut from the longer draft, and why: the slide explaining what sigma is and how the Gaussian
likelihood trains it, removed at the presenter's request -- that will be covered elsewhere, so
the deck now goes straight from accuracy to the measurement. Also cut: the per-cell ranking slide (sigma barely orders cells by
their true error -- 0.036 / 0.035 / 0.002 / 0.040 by channel) was removed at the presenter's
request as too involved to defend live. It is the one thing the deck no longer states about what
the uncertainty CANNOT do, so the finding and its figure are held as a prepared answer in slide
4's notes; put the slide back if the meeting turns on per-cell reliability. Also cut: the Sec. 2.4 aleatoric/epistemic split (the "2.5% but the
ensemble has not collapsed" point needs a paragraph to state and is easy to misread as a
failure); the read-out design history and the 24-bin lookup probe (a diagnostic we built, not a
method we propose -- it invites questions the deck should not have to defend); and AUSE, a
composite metric that has to be defined before it can be quoted. The measurements all still
exist in check_outputs/eval and can be pulled back in if the meeting goes that way.

What survives is what can be defended in one sentence each: accuracy is a tie, the uncertainty
is on a usable scale exactly where it matters, it cannot say WHICH cell is wrong, and it is an
order of magnitude cheaper. Every number is read from check_outputs/eval at build time.

    python3 slides/build_meeting3_deck.py
"""
from __future__ import annotations
import json
import os
from crowdcore import paths
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
from slides.build_slides import OUTPUTS, ROOT, render_notes, render_pdf, render_pptx  # noqa: E402

EV = os.path.join(OUTPUTS, "eval")
J = lambda n: json.load(open(os.path.join(EV, n)))
F = lambda *p: os.path.join(OUTPUTS, *p)

U = J("uncertainty_vsb0.json")["results"]
E = J("uncertainty_enkf_k1.json")["results"]
SP = J("sparsification.json")["per_channel_blind"]
SF = J("sigma_fair.json")["splits"]
B = J("bench_speed_vsb0.json")["runs"]

t = lambda k: B[k]["wall_mean"]
raw = lambda k, f: SF[k]["raw"][f]
sp = lambda c: SP[c]["spearman"]

slides = [
    dict(kind="title",
         title="Predictive uncertainty for crowd-field reconstruction",
         subtitle="what our $\\sigma$ measures, and what it does not",
         author="Xinle Zhang",
         notes="Four claims, one per slide after this: accuracy is a tie, the uncertainty is on "
               "a usable scale where it matters, it cannot say which individual cell is wrong, "
               "and it is an order of magnitude faster.\n\nThe EnKF appears only on the accuracy "
               "and timing slides. Its own uncertainty is not compared here — a fair comparison "
               "would need its inflation retuned, and this project does not change the EnKF's "
               "configuration."),

    dict(title="Accuracy — a tie where it counts",
         images=[(F("eval", "mt_rmse.png"), (1.55, 1.25, 10.2, 4.55))],
         bullets=[(0.9, 6.10, 11.6, 1.05, 13.5, [
             (f"On unobserved cells the two methods are "
              f"{abs(U['blind']['rmse'] / E['blind']['rmse'] - 1) * 100:.0f}% apart "
              f"({U['blind']['rmse']:.4f} vs {E['blind']['rmse']:.4f}). That is the region the "
              f"problem is about, and neither method is ahead there.", 0),
             ("Everything after this slide is about the uncertainty, not the reconstruction.", 0),
         ])],
         notes="Lead with the tie so the rest is not read as advocacy.\n\n7 held-out Sundays, "
               "k=1, identical observation realisation for both methods. The gap on OBSERVED "
               "cells is where the aggregate difference comes from."),

    dict(title="Is it the right size? It over-states by about 2.5x",
         images=[(F("eval", "mt_size.png"), (2.55, 1.30, 8.4, 3.9))],
         bullets=[(0.9, 5.50, 11.6, 1.75, 13.5, [
             (f"Split by whether a robot actually measured the cell. The model reports almost "
              f"the same uncertainty either way ({U['observed']['sigma_mean']:.2f} vs "
              f"{U['blind']['sigma_mean']:.2f}) while the real error differs by "
              f"{(U['blind']['rmse'] / U['observed']['rmse'] - 1) * 100:.0f}% "
              f"({U['observed']['rmse']:.2f} vs {U['blind']['rmse']:.2f}).", 0),
             ("So it over-states everywhere, and it does not register whether that cell was "
              "measured at all.", 0),
             ("Most of the gap comes from the unobserved cells that are truly empty; on the "
              "cells that do contain people the two sizes match closely.", 0),
         ])],
         notes="Three cases the audience already understands: the robots either measured that "
               "cell or they did not. Two magnitudes in each, no ratio and no coverage curve — "
               "both of those say the same thing and both need defining first.\n\nThe third "
               "bullet does not need expanding unless asked. If it is: sigma is 0.487 on empty "
               "unobserved cells against an actual error of 0.114, and 0.507 on occupied ones "
               "against an actual error of 0.519. Sigma barely moves while the real error "
               "changes 4.5x, so the aggregate is dragged by the empty majority, which is 87% "
               "of unobserved cells. The figure is check_outputs/eval/mt_split.png.\n\nIf "
               "asked 'why not scale sigma down' — the factor would be set by that 87% majority "
               "and it takes the occupied cells from about right to badly over-confident. One "
               "factor cannot serve two regimes that differ by 4.5x.\n\nPREPARED ANSWER, no "
               "longer on a slide: 'can you use sigma to flag which individual cells are "
               "unreliable?' No. Ranking cells by sigma and by their true error agree at 0.036 / "
               "0.035 / 0.002 / 0.040 across the four channels, where 1 would be perfect and 0 "
               "unrelated — so within a channel it is close to a random ordering. Being the "
               "right SIZE and knowing WHERE are independent, and we have the first only. The "
               "uncertainty is therefore usable at the level of regions and channels, not "
               "single cells. Figure: check_outputs/eval/mt_agreement.png."),

    dict(title="Inference time",
         images=[(F("eval", "mt_time.png"), (2.35, 1.20, 8.7, 4.7))],
         bullets=[(0.9, 6.10, 11.6, 1.05, 13.5, [
             ("1,000 frames, same node, 4 cores, interleaved order with a warm-up and 5 timed "
              "repeats. Largest run-to-run spread 3.6%.", 0),
             (f"The 5-network ensemble is {t('enkf_k1') / t('varnet_ens5'):.1f}x faster than the "
              f"EnKF; a single network is {t('enkf_k1') / t('varnet_single'):.0f}x faster.", 0),
         ])],
         notes="Interleaving matters: no node was free, so contention becomes a reported spread "
               "instead of a hidden bias.\n\nThe EnKF pays a Kalman update per observed frame, so "
               "k=4 costs less than k=1. Ours runs a fixed number of iterations regardless of how "
               "sparse the observations are.\n\nIf inference cost ever binds, the five networks "
               "are the first thing to reconsider — they cost 4.6x one network."),
]

if __name__ == "__main__":
    out = paths.SLIDES
    render_pdf(slides, os.path.join(out, "meeting3_deck.pdf"))
    render_pptx(slides, os.path.join(out, "meeting3_deck.pptx"))
    render_notes(slides, os.path.join(out, "meeting3_deck_notes.md"))
    print(f"[ok] {len(slides)} slides -> slides/meeting3_deck.{{pdf,pptx}} + _notes.md")
