# 演讲备注 / Speaker notes

## Slide 1: Uncertainty: completing the EnKF evaluation, and a third method

(no notes)

## Slide 2: Task 1 — the optimised EnKF, 7 days, both observation densities

Say the last point deliberately: it is the first hint of what slide 5 explains.

One gap to own if asked: test_metrics_enkf_k*.json does not record which of the three copies (orig / lab / opt) produced it, so 'the optimised implementation' currently rests on the process record rather than on the file. The numbers are unaffected -- verify_enkf_opt.py proves the three are bit-identical -- and I am adding the field.

## Slide 3: Task 2 — the third method: deep ensembles

The question to put to the meeting: deep ensembles is an uncertainty RECIPE, not a separate reconstruction method — we applied it to our own 4DVarNet. If the three-way benchmark is meant to have three independent RECONSTRUCTION methods, that is a different search and I should know now.

Cost detail if asked: the variance head is 548 parameters, 0.03% of the model.

## Slide 4: Result — better on accuracy and on uncertainty

These numbers are from after the head's sigma^2 floor was removed, so they differ from any earlier version of this deck. No retraining — the same five checkpoints with sigma^2 = softplus(head).  A floorless retrain (runs/varnet_nf5_s*) is running now; if it moves these numbers it will move them in the observed cells, which is where the mismatch is.

The member spread is not optional. The learnt sigma alone scores 0.65 spread/skill against the five-member 0.87, and its NLL diverges, because with nothing flooring it a single member can claim a sigma near zero. While the floor existed the aleatoric-only variant was the better calibrated of the two, so this used to look like a way to cut inference 5x. It is not any more.

The learned sigma also knows where it is unsure: 0.1960 in blind cells against 0.1177 in observed ones. The EnKF's is 0.0025 against 0.0025 — identical.

## Slide 5: Why the EnKF's uncertainty fails — and why tuning would not fix it

This is the strongest technical content in the deck. The chain, all measured: the surrogate removes ~65% of member disagreement per step; the filter injects 0.0022 per step; the equilibrium spread is 0.0025, which is what we observe. With spread that small the Kalman gain is near zero, so observations are barely assimilated -- which is why slide 2's k=1 and k=4 look the same, and why the accuracy suffers too.

If pushed to fix it: matching the true error would need ~100x the injected noise, i.e. adding noise of the same magnitude as the state itself every step. Inflation would need ~2.9 against a normal range of 1.02-1.2.

The honest limit of the claim: this is about a deterministic neural surrogate as the forecast model, not about ensemble filters in general.

## Slide 6: Where this leaves us

Lead with the open question about what 'three-way' means -- it is the one thing only the supervisor can answer, and it determines the next two weeks.

One item that used to be on this list is now settled: whether the five members are needed. With the sigma^2 floor removed the learnt sigma alone is badly calibrated and its NLL diverges, so the member spread is load-bearing. They stay.
