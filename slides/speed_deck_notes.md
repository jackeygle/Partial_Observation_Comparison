# 演讲备注 / Speaker notes

## Slide 1: 4DVarNet inference speed

(no notes)

## Slide 2: Inference cost per frame

Log axis — the numbers span four orders of magnitude. The red line is the real-time budget. Lead with the 52x same-CPU number, not the GPU one.

If asked how it was measured: one job, one node, the same 200 frames for every method; the six measurements run serially and INTERLEAVED (A B C A B C A B C) rather than blocked, so a shift in the node's background load spreads over all methods instead of landing on one and reading as a method difference; one untimed warm-up each; 3 timed repeats; max run-to-run spread 2.0%. The node was shared, not exclusive — no node in batch-csl (0 of 48 idle) or gpu-h200 was free, and interleaving plus repeats converts that contention from a hidden bias into a reported spread. Observation-matrix construction is outside the timed region: that is our driver's bookkeeping, not the filter's cost.

## Slide 3: The EnKF's cost is its implementation, not its algorithm

Raise this before they find it in the code. Framing: a useful finding for their project — a 20x speed-up with unchanged results — not a criticism. And it is why the headline comparison uses the 131 ms figure.

Detail if asked: the localization matrix is built by a four-deep Python loop over 252 observations x 4 channels x 36 x 12 = 435,456 iterations, and update() calls it twice per step with the same argument. It ran at 3.7 of 16 cores — serial Python, so more CPUs would not have helped. Verification: same call on real observed cells through both copies, np.array_equal, plus a full-day run. We vendored a read-only reference copy (enkf_lab) and changed a separate experiment copy (enkf_opt) inside our own repository; Partial_observation itself is untouched.

## Slide 4: Reconstruction accuracy — held-out test set

If asked whether the gaps are reliable: on every one of the 7 days the ordering is the same, with no exception — 4DVarNet beats the filter on all seven, and every-frame beats every-4th-frame on all seven. Day-to-day spread is 0.011 / 0.007 / 0.011 RMSE respectively, several times smaller than the gaps themselves.

Reading the table out loud, if useful: at the same observation density 4DVarNet's error is 39% lower than the filter's (MSE 0.0591 vs 0.0359); observing every frame lowers it a further 19% at no inference cost.

The EnKF's every-frame run is genuinely still going: its original implementation needs ~4.8 s per frame, so one test day is ~53 hours. Do not present the empty cell as a result either way.

If asked whether this is overfitting: the same k comparison on the TRAINING days gives 24% against 19% here, so the gain carries over to unseen days.

If asked how much gets observed: 11.7% of all 36x12 cells at every 4th frame, 46.9% at every frame; relative to the 290 walkable cells it is 17.5% and 69.8%. The robots run a fixed seeded A* route, so coverage is identical on every test day.

If asked about the blind zone specifically: restricted to cells never observed, 4DVarNet goes 0.195 -> 0.184 — a smaller gain, since full-state RMSE also improves simply by moving cells into the observed set.

Earlier per-epoch figures came from the training split and are not shown.

## Slide 5: Where this goes next

Keep this short. The speed result is the message today; this slide is just to show the direction.
