"""
build_de_apply_deck.py  —  what of deep ensembles applies to this project

Not a summary of the paper and not a report of results. The question it answers is: given that
our reconstruction is a 4DVarNet trained on squared error, what can be taken from
Lakshminarayanan et al. (NeurIPS 2017) and what cannot.

The slide showing where the variance head attaches to our solver was removed on purpose --
that is an implementation question, and it is being deferred. checks/plot_de_application.py
still draws the figure if it is wanted back.

Numbers on these slides are the PAPER's only. Ours are deliberately absent — they belong to the
results discussion, not this one.

    python3 slides/build_de_apply_deck.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT_)
from build_slides import OUTPUTS, ROOT, render_notes, render_pdf, render_pptx  # noqa: E402

F = lambda *p: os.path.join(OUTPUTS, *p)

slides = [
    # ───────────────────────────────────────────────────────────── 1
    dict(kind="title",
         title="Deep Ensembles — what applies to our reconstruction",
         subtitle="Lakshminarayanan, Pritzel & Blundell, NeurIPS 2017  (arXiv:1612.01474)",
         author="Xinle Zhang"),

    # ───────────────────────────────────────────────────────────── 2
    dict(title="Where we start: the model has no uncertainty to report",
         bullets=[
             (0.9, 1.45, 11.6, 5.2, 16, [
                 ("Our 4DVarNet is trained on 4DVarNet's Eq. 14 — the plain squared error "
                  "between the reconstruction and the truth.", 0),
                 ("", 0),
                 ("Squared error constrains the MEAN and nothing else. There is no second "
                  "output, and no term in the objective that any second output could have "
                  "been fitted against.", 0),
                 ("", 0),
                 ("So the gap is not that we have not computed the uncertainty yet. It is that "
                  "a model trained this way has no quantity that represents it — which is why "
                  "it cannot be added as a post-processing step, and why the change has to "
                  "reach the training objective.", 0),
                 ("", 0),
                 ("What follows is what the paper offers for exactly this situation, and how "
                  "much of our existing model it would disturb.", 0),
             ]),
         ],
         notes="If the softmax question comes up: a classifier's softmax output is routinely "
               "read as confidence and the paper's Fig. 1 shows why that is wrong — a point far "
               "outside the training data can still be given probability 1. Confidence in a "
               "class is not confidence in the prediction. The regression case here is starker: "
               "there is not even a misleading number to point at."),

    # ───────────────────────────────────────────────────────────── 3
    dict(title="What the paper proposes instead  (their Eq. 1)",
         images=[(F("eval", "de_eq_nll.png"), (0.75, 1.85, 11.8, 3.39))],
         bullets=[
             (0.9, 1.30, 11.6, 0.6, 14.5, [
                 ("The network emits two values per output — a mean and a variance — and the "
                  "target is treated as a draw from N(μ, σ²).", 0),
             ]),
             (0.9, 5.45, 11.6, 1.8, 14.5, [
                 ("σ² is kept positive with a softplus and floored at a small minimum variance "
                  "for numerical stability.", 0),
                 ("It is heteroscedastic: σ² is a function of the input, so the model can be "
                  "confident in some cells and not in others, instead of quoting one error bar "
                  "for the whole field.", 0),
             ]),
         ],
         notes="The two terms are what makes this work. On its own the first drives sigma to "
               "zero and the model claims certainty everywhere; on its own the second inflates "
               "sigma and the model claims nothing. Their sum has one minimum, at sigma equal "
               "to the actual error.\n\nThe paper is explicit that this costs accuracy — it "
               "optimises NLL, not MSE. Their Table 1 shows RMSE slipping against squared-error "
               "methods by a median of about 5.6%."),

    # ───────────────────────────────────────────────────────────── 4
    dict(title="The second thing worth taking: what the uncertainty is made of",
         images=[(F("eval", "de_eq_combine.png"), (0.75, 1.55, 11.8, 3.58))],
         bullets=[
             (0.9, 5.35, 11.6, 1.9, 14.5, [
                 ("First term: the average of what each network itself reports it cannot "
                  "account for. Adding members does not reduce it. Second term: how far the "
                  "networks' means are from each other on the same input — it shrinks as they "
                  "come to agree.", 0),
             ]),
         ],
         notes="Be careful with the vocabulary if it is challenged. 'Aleatoric' strictly means "
               "irreducible given the measurement process — sensor noise qualifies, a cell no "
               "robot visited does not, since more robots would reduce it. Both nevertheless "
               "land in the first term, because a single network reports large sigma in either "
               "case. The split of the total variance is exact; the naming is a "
               "reading.\n\nThe second term is the cleaner of the two: disagreement between "
               "models fitted to the same data is unambiguous, and it is what the paper's "
               "out-of-distribution experiments measure — it grows on inputs unlike anything "
               "seen in training."),

    # ───────────────────────────────────────────────────────────── 5
    dict(title="What we would not take, and what it costs",
         bullets=[
             (0.9, 1.40, 11.6, 2.5, 15, [
                 ("Not taken — adversarial training (their Sec. 2.3)", 0),
                 ("it is step 2 of their recipe but optional, and their Table 2 states it does "
                  "not significantly help on the regression benchmarks", 1),
                 ("its input perturbation, a step along the sign of the loss gradient, also has "
                  "no clear reading on an observation field: our inputs are a measured state "
                  "and a mask, not free variables", 1),
             ]),
             (0.9, 4.10, 11.6, 1.6, 15, [
                 ("What it costs", 0),
                 ("M independent trainings and M forward passes at inference; the paper "
                  "recommends M = 5 and reports diminishing returns beyond it", 1),
                 ("accuracy: the objective optimises NLL rather than squared error", 1),
             ]),
         ],
         notes="Open question for the meeting, and the one thing only the supervisor can "
               "settle: deep ensembles is an uncertainty recipe that applies to any regressor, "
               "not a third reconstruction method. If the three-way benchmark is meant to "
               "compare three independent RECONSTRUCTION methods, that is a different "
               "literature search and it should start now.\n\nIf asked why not a Bayesian "
               "neural network: the paper's own argument is cost and intrusion — parameters "
               "double, the training procedure changes, and on their benchmarks it did not "
               "come out ahead."),
]

if __name__ == "__main__":
    out = os.path.join(ROOT, "slides")
    render_pptx(slides, os.path.join(out, "de_apply_deck.pptx"))
    render_pdf(slides, os.path.join(out, "de_apply_deck.pdf"))
    render_notes(slides, os.path.join(out, "de_apply_deck_notes.md"))
    print(f"\n  {len(slides)} slides — applicability, paper's numbers only")
