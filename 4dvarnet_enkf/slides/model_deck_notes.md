# 演讲备注 / Speaker notes

## Slide 1: The dynamical prior  Φ

(no notes)

## Slide 2: What the prior is for

Say the last bullet out loud even if nobody asks: it is the reason the prior matters at all for our problem. Roughly 38% of walkable cells carry no observation in a given frame, and their reconstruction comes only from Φ.

α_obs and α_reg are learned alongside everything else, so the balance between the two terms is not hand-tuned either.

## Slide 3: Inside Φ, part 1 — ψ, the one convolution that sees neighbours

This is the single most important slide in the deck — everything else in the prior's design exists to protect this property.

If asked how far it reaches: kernel 3x3x3 means +-1 frames in time and +-1 cells in space. Because phi is pointwise, that is the ENTIRE receptive field of one branch.

## Slide 4: Inside Φ, part 2 — φ, pointwise on purpose

The order in the code is psi -> ReLU -> Conv(1x1x1) -> ReLU -> Conv(1x1x1); the 1,188 parameters are the two convolutions..

## Slide 5: Inside Φ, part 3 — two scales  (Eq. 10)

The mistake we made first and then fixed: blurring x, feeding blur and high-pass to two FULL-resolution branches. Both branches then had the same receptive field in real terms, which defeats the purpose. Now the coarse branch genuinely runs on the smaller grid.

Caveat if pressed: with two scales the zero-centre property is no longer exact. Pooling folds x(s) into a coarse cell and Up spreads it back, so d|Phi(x)(s)|/dx(s) is 2e-2 on this model — 2% of what the identity would give, against exactly 0 for a single branch. Still far from reachable, but say 'effectively excluded', not 'excluded'. It is a property of Eq.10 itself; the paper does not quantify it. Measured by checks/check_paper_conformance.py.

## Slide 6: Where it stands — accuracy

More observations help us far more than they help the filter (0.188 -> 0.167 against 0.243 -> 0.239).

If asked why the filter barely improves: its ensemble spread is about 90x smaller than its actual error, so the Kalman gain is near zero and observations are largely ignored — measured, not inferred. Mention only if asked, and add that it is their configuration and we did not change it.

## Slide 7: Where it stands — inference cost

If asked how solid the absolute numbers are, say it plainly: the EnKF's per-frame cost has measured anywhere from 130 to 290 ms across sessions on identical code, because every batch-csl node is shared and its k=1 pass runs for minutes, so it absorbs whatever else is on the node. 4DVarNet's is stable (3.3-3.8 ms) because each pass is seconds. The ratio between the methods is what survives that, and it is 40-80x in every session. No exclusive node was available to do better.

The structural point the figure makes: the filter's cost tracks observation density because it pays a Kalman analysis per observed frame, while its forecast runs every frame either way (33 vs 32 ms). The solver's cost does not move at all (3.66 vs 3.63 ms) — it runs a fixed 20 iterations over the whole dense window.

So denser observations widen the gap in speed as well as in accuracy.
