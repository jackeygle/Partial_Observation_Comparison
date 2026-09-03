# 演讲备注 / Speaker notes

## Slide 1: Training results

(no notes)

## Slide 2: What has been trained

a4 was left unfinished earlier and has now been completed to epoch 100 and evaluated, so there are no partial runs left.

The inference figures quoted for b0 and a2 are the 4-core CPU benchmark, which is the convention used throughout. a4 was never benchmarked on CPU; on a matched A100 it is 3.1x b0's architecture, but that is a different measurement and should not be mixed with the CPU numbers.

## Slide 3: Prior capacity has not saturated

This corrects an expectation I had going in: I assumed capacity would saturate somewhere near 30k parameters and that a4 would confirm it. It did not — a4 is the most accurate of the three.

The prior is a small fraction of the model either way: even a4's 54,220 is 2.6% of the 2.1M total, almost all of which is the ConvLSTM solver. So this is not a story about model size, it is about the prior specifically.

## Slide 4: But the largest prior does not train reliably

The jump lands two epochs after the curriculum raises the solver from 10 to 15 iterations. A longer unrolled gradient path through a larger prior is the obvious suspect and --clip-grad 1.0 did not contain it, but this is one run at one capacity — a mechanism worth stating as a hypothesis, not a conclusion.

The practical consequence is what matters: the accuracy advantage on the previous slide comes from a checkpoint taken before the instability, so it is real but it was not obtained by a training procedure I would rely on. b0 delivers within 3% of it, trains monotonically, and runs 2-3x faster.

## Slide 5: What the NLL objective bought: a usable uncertainty

This is the answer to 'why pay 1.8% accuracy'.

These numbers are from after the head's sigma^2 floor was removed — same five checkpoints, no retraining, sigma^2 = softplus(head).  A floorless retrain (runs/varnet_nf5_s*) is running now; if it moves these numbers it will move them in the observed cells, which is where the mismatch is. Anyone who saw an earlier version of this slide saw different figures.

The member spread is not optional: the learnt sigma alone is calibrated at 0.65 with 76.9% coverage and its NLL diverges, because nothing floors a member that claims certainty. That question used to be open and is now settled.

The EnKF's spread is its own project's configuration, unmodified.

## Slide 6: Where the reconstruction stands

The EnKF number is its own project's configuration, unmodified — worth saying before it is asked.

Open items, none of them blocking: no CPU benchmark for a4; no uncertainty measured at k=4 for our side; and whether the 5-member ensemble is needed at all, given how closely the members agree.
