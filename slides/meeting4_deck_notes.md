# 演讲备注 / Speaker notes

## Slide 1: Four methods, two benchmarks — and a ranking that moves

One sentence to open with: every headline number in this deck is stable, and every RANKING built from those numbers is not. That is the finding, not a caveat.

## Slide 2: What the four methods actually offer

The bottom line settles what “three-way” means for this project: the comparison is over four independent RECONSTRUCTION methods, and the two 4DVarNet sigma-hat designs are ours rather than the paper's.

If asked why Senseiver has no uncertainty: it is a deterministic sensor-to-field decoder; adding a variance head would be the same kind of bolt-on we are about to show does not work for 4DVarNet.

## Slide 3: Hypothesis

Say the last line slowly. The deck is falsifiable: two chances for the ordering to hold, and it holds in neither.

## Slide 4: Test 1 — accuracy: the same predictions, scored two ways

Numbers if pushed: convention A blind RMSE — DINCAE 0.329, Senseiver 0.343, 4DVarNet 0.359, EnKF 0.368. Convention B — Senseiver 0.168, 4DVarNet 0.191, EnKF 0.215, DINCAE 0.757.

All five rows are single models, default configuration, seed 0, and each one's checkpoint picked on ITS OWN validation set. No ensembling, no checkpoint averaging. Held ready: 16-checkpoint averaging is worth only 1.7% to DINCAE, less than picking the right single checkpoint.

Four panels, not four bars in two: convention down the rows, cell scope across the columns. Each panel is one plain bar chart in the method colours, so nothing has to be decoded from a legend.

One detail visible in the figure and worth having ready: the EnKF is the only method whose FULL-FIELD error (0.374) is worse than its blind error (0.368). Every other method is clearly better where it was given observations. That is the same ensemble collapse the uncertainty slide is about — the gain is so small the filter barely assimilates what it sees.

## Slide 5: Test 2 — uncertainty: what makes a sigma-hat useful

Why CRPS is the verdict column and not NLL: NLL has a (x-mu)^2/2sigma^2 term that is unbounded as sigma -> 0. The EnKF's ensemble collapses to sigma = 0.0025 while actually being wrong by 0.215, so its NLL is 1.7e18 — that cannot rank anything. CRPS is bounded, in data units, and reduces to MAE as sigma -> 0.

Robustness, the important part: under the all-cells convention the same four come out -76.1% / -24.3% / +50.1% / +26.8%. The magnitudes move a lot — vsb0 looks catastrophic there and merely tied here — but the SIGN does not. Structural wins, bolt-on loses, both ways.

Cost, if asked: the sigma-hat is not free. aug0 is 13.9% worse than plain MSE on accuracy, vsb0 10.3% worse; the three arms' 5-seed intervals do not overlap. So vsb0 pays and buys nothing, aug0 pays 3.6 points more and buys a usable sigma.

## Slide 6: Action item — inference time, per frame

Own the correction plainly, then move on — it is one slide, not a defence.

Interleaved, not blocked: round 1 runs every method, then round 2, and so on. Blocking would let a change in the shared node's background load land entirely on one method and be read as a difference between methods. Largest run-to-run spread here is 3.5% of the mean.

The caveat matters and I would rather state it than be asked: for the three per-frame methods this number IS the delay before that frame's estimate exists. For 4DVarNet it is not — one window solves in about 0.55 s. I have the latency figures measured and can show them if you want, but the amortised cost is what is on this slide.

k=4 is not shown: in our setting the robots observe every frame. The a2 widened-prior 4DVarNet is not shown either — the bar here is the same model as the accuracy slides.

## Slide 7: The ablation that settles it

This is the strongest slide. Everything before it argues the hypothesis across methods; this proves it INSIDE one method, where architecture, data, and checkpoint rule are all held fixed.

Cost: 150 epochs, ~21 GPU-hours, plus a separate 13 GB target cache. The change itself is two lines in encode_target.

One trap worth mentioning if the question comes: the naive version — flip the loss mask to 1 and keep the existing targets — trains the network to predict the per-cell CLIMATOLOGY on empty cells, because a stored 0 in normalised space decodes to mean[c] (-0.067 for vy), not to physical zero.
