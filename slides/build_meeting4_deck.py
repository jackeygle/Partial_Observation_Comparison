"""
build_meeting4_deck.py — 2026-09-07 汇报，六页(标题 + 五页正文)

结构照导师的要求做:**一个具体假设 -> 验证 -> 结论**，不是图的堆砌。上一次被指出
"三方对比图没进幻灯片"以及"把 1000 帧总时间当成了推理时间"，这两条都在这一版里正面
回应 —— 而且速度那条不是当作待修的 bug，是当作主论点的第三个实例。

主线:

    假设      四个方法的排名，取决于一个通常不写进论文的口径选择。
    验证①    精度        有定义格 vs 所有格子  -> DINCAE 第 1 <-> 第 5
    验证②    不确定性    结构性 vs 事后附加    -> 跨两套口径都稳

另有一页回应上周的 action item:**逐帧推理时间**,四方 mean ± s.d.。只报吞吐 ——
latency 那一栏按要求撤下,但图上留了一行说明 4DVarNet 那个数是窗口摊销值,
否则单独引用它会重现上周被指出的那个问题。
    压轴      同一架构只换监督口径，allcells 误差掉 77%，垫底变第 2

刻意**停在消融**，不加结论页:那两个原本要问导师的问题已经当面答过
(三方 = 四个独立重建方法 + 两个我们自己的 4DVarNet 不确定性设计;窗口时延不必考虑)。

刻意不放的:
  * 逐通道拆解(四通道 x 两口径 x 三切法 = 24 个数，现场讲不完，放备注里当备答)
  * 消融表(a4_k1 加宽先验、16-ckpt 平均)，同上
  * k=4 与 a2 加宽先验那两档速度 —— 见速度页的备注。

用法(登录节点):
    source sbatch/_env.sh && python3 -m slides.build_meeting4_deck
"""
from __future__ import annotations

import os
import sys

from crowdcore import paths                                                   # noqa: E402
from slides.build_slides import OUTPUTS, render_notes, render_pdf, render_pptx  # noqa: E402


def F(*p):
    return os.path.join(OUTPUTS, *p)


TITLE = dict(
    kind="title",
    title="Four methods, two benchmarks — and a ranking that moves",
    subtitle="Crowd-field reconstruction from partial robot observations · ATC, 7 held-out days",
    author="2026-09-07",
    notes="One sentence to open with: every headline number in this deck is stable, and every "
          "RANKING built from those numbers is not. That is the finding, not a caveat.")

SLIDES = [
    TITLE,

    dict(title="What the four methods actually offer",
         images=[(F("eval", "m4_capability.png"), (1.05, 1.35, 11.2, 3.5))],
         bullets=[(0.9, 5.15, 11.6, 1.9, 13.5, [
             ("Accuracy compares four methods. Uncertainty compares three — Senseiver has no "
              "sigma-hat at all.", 0),
             ("For DINCAE and the EnKF the uncertainty IS the method: remove the information "
              "form or the ensemble and you no longer have that method.", 0),
             ("For 4DVarNet we added it. Two designs, neither in the original paper.", 0),
             ("So: four reconstruction methods, plus two 4DVarNet uncertainty designs that are "
              "ours, not the paper's.", 1),
         ])],
         notes="The bottom line settles what “three-way” means for this project: the comparison "
               "is over four independent RECONSTRUCTION methods, and the two 4DVarNet sigma-hat "
               "designs are ours rather than the paper's.\n\n"
               "If asked why Senseiver has no uncertainty: it is a deterministic sensor-to-field "
               "decoder; adding a variance head would be the same kind of bolt-on we are about to "
               "show does not work for 4DVarNet."),

    dict(title="Hypothesis",
         bullets=[(1.15, 1.75, 11.1, 4.4, 20, [
             ("The ranking of these four methods is decided by a scoring convention that most "
              "papers do not state.", 0),
             ("", 0),
             ("Two independent tests, both on the same 7 held-out days:", 0),
             ("accuracy — which cells count", 1),
             ("uncertainty — which cells count", 1),
             ("", 0),
             ("If the hypothesis is wrong, the ordering survives both. It survives neither.", 0),
         ])],
         notes="Say the last line slowly. The deck is falsifiable: two chances for the "
               "ordering to hold, and it holds in neither."),

    dict(title="Test 1 — accuracy: the same predictions, scored two ways",
         images=[(F("eval", "m4_accuracy.png"), (1.30, 1.02, 10.7, 4.72))],
         bullets=[(0.9, 5.86, 11.6, 1.35, 12.8, [
             ("DINCAE is first under one convention and last under the other — 4.5x worse than "
              "Senseiver. Nothing about the model changed.", 0),
             ("Cause: DINCAE only defines velocity where density > 0. 88.4% of blind velocity "
              "cells are empty, and on those the truth is exactly 0 (measured: 0.000% non-zero).", 0),
             ("Convention A is the defensible one — “velocity” on an empty cell is not a "
              "physical quantity.", 0),
             ("Left column scores the full field, right column the blind cells alone. Observed "
              "cells are handed to the model, so the right column is the reconstruction task.", 0),
         ])],
         notes="Numbers if pushed: convention A blind RMSE — DINCAE 0.329, Senseiver 0.343, "
               "4DVarNet 0.359, EnKF 0.368. Convention B — Senseiver 0.168, 4DVarNet 0.191, "
               "EnKF 0.215, DINCAE 0.757.\n\n"
               "All five rows are single models, default configuration, seed 0, and each one's "
               "checkpoint picked on ITS OWN validation set. No ensembling, no checkpoint "
               "averaging. Held ready: 16-checkpoint averaging is worth only 1.7% to DINCAE, "
               "less than picking the right single checkpoint.\n\n"
               "Four panels, not four bars in two: convention down the rows, cell scope across "
               "the columns. Each panel is one plain bar chart in the method colours, so nothing "
               "has to be decoded from a legend.\n\n"
               "One detail visible in the figure and worth having ready: the EnKF is the only "
               "method whose FULL-FIELD error (0.374) is worse than its blind error (0.368). "
               "Every other method is clearly better where it was given observations. That is "
               "the same ensemble collapse the uncertainty slide is about — the gain is so "
               "small the filter barely assimilates what it sees."),

    dict(title="Test 2 — uncertainty: what makes a sigma-hat useful",
         images=[(F("eval", "m4_uncertainty.png"), (0.55, 1.10, 12.2, 4.35))],
         bullets=[(0.9, 5.65, 11.6, 1.55, 13.0, [
             ("Judged against a null model: replace every cell's sigma-hat with one constant, "
              "that split's own RMSE. It knows how big the error is on average and nothing about "
              "WHERE — and it scores a perfect 1.00 spread/skill for free.", 0),
             ("Uncertainty built into the model beats it. Bolted on afterwards, or read off "
              "ensemble spread, does not.", 0),
             ("This ordering is the one thing in the deck that does NOT move with the convention.", 1),
         ])],
         notes="Why CRPS is the verdict column and not NLL: NLL has a (x-mu)^2/2sigma^2 term that "
               "is unbounded as sigma -> 0. The EnKF's ensemble collapses to sigma = 0.0025 while "
               "actually being wrong by 0.215, so its NLL is 1.7e18 — that cannot rank anything. "
               "CRPS is bounded, in data units, and reduces to MAE as sigma -> 0.\n\n"
               "Robustness, the important part: under the all-cells convention the same four come "
               "out -76.1% / -24.3% / +50.1% / +26.8%. The magnitudes move a lot — vsb0 looks "
               "catastrophic there and merely tied here — but the SIGN does not. Structural wins, "
               "bolt-on loses, both ways.\n\n"
               "Cost, if asked: the sigma-hat is not free. aug0 is 13.9% worse than plain MSE on "
               "accuracy, vsb0 10.3% worse; the three arms' 5-seed intervals do not overlap. So "
               "vsb0 pays and buys nothing, aug0 pays 3.6 points more and buys a usable sigma."),

    dict(title="Action item — inference time, per frame",
         images=[(F("eval", "m4_speed.png"), (1.55, 1.12, 10.2, 4.35))],
         bullets=[(0.9, 5.62, 11.6, 1.55, 13.0, [
             ("You were right that last week's figure was total run time, not inference time. "
              "This is per frame: mean ± s.d. over 5 interleaved repeats, all four methods on "
              "the same node over the same frames, with nothing printed inside the timed region.", 0),
             ("Senseiver is the fastest by a wide margin — and it is also the most accurate "
              "under Convention B.", 0),
             ("Caveat on the 4DVarNet bar: it solves a whole 200-frame window at once, so its "
              "2.74 ms is amortised over the window, not the wait for one frame.", 1),
         ])],
         notes="Own the correction plainly, then move on — it is one slide, not a defence.\n\n"
               "Interleaved, not blocked: round 1 runs every method, then round 2, and so on. "
               "Blocking would let a change in the shared node's background load land entirely "
               "on one method and be read as a difference between methods. Largest run-to-run "
               "spread here is 3.5% of the mean.\n\n"
               "The caveat matters and I would rather state it than be asked: for the three "
               "per-frame methods this number IS the delay before that frame's estimate exists. "
               "For 4DVarNet it is not — one window solves in about 0.55 s. I have the latency "
               "figures measured and can show them if you want, but the amortised cost is what "
               "is on this slide.\n\n"
               "k=4 is not shown: in our setting the robots observe every frame. The a2 widened-"
               "prior 4DVarNet is not shown either — the bar here is the same model as the "
               "accuracy slides."),

    dict(title="The ablation that settles it",
         bullets=[(1.0, 1.45, 11.4, 4.8, 15.5, [
             ("Same DINCAE architecture. Same 7 days. Same validation-picked checkpoint rule. "
              "The only change: supervise every cell instead of only the defined ones.", 0),
             ("", 0),
             ("all-cells blind RMSE:   0.757  ->  0.173      (-77%)", 1),
             ("defined blind RMSE:     0.329  ->  0.349      (+5.9%)", 1),
             ("", 0),
             ("Last place, 4.5x behind, becomes second — by changing the loss mask, not the "
              "model. The gap under convention B was never a capability gap.", 0),
             ("And the 5.9% it loses under convention A says the information form is genuinely "
              "part of why DINCAE wins there — not just its architecture.", 0),
         ])],
         notes="This is the strongest slide. Everything before it argues the hypothesis across "
               "methods; this proves it INSIDE one method, where architecture, data, and "
               "checkpoint rule are all held fixed.\n\n"
               "Cost: 150 epochs, ~21 GPU-hours, plus a separate 13 GB target cache. The change "
               "itself is two lines in encode_target.\n\n"
               "One trap worth mentioning if the question comes: the naive version — flip the "
               "loss mask to 1 and keep the existing targets — trains the network to predict the "
               "per-cell CLIMATOLOGY on empty cells, because a stored 0 in normalised space "
               "decodes to mean[c] (-0.067 for vy), not to physical zero.")
]


def main():
    out = paths.SLIDES
    render_pdf(SLIDES, os.path.join(out, "meeting4_deck.pdf"))
    render_pptx(SLIDES, os.path.join(out, "meeting4_deck.pptx"))
    render_notes(SLIDES, os.path.join(out, "meeting4_deck_notes.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
