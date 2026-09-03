# 演讲备注 / Speaker notes

## Slide 1: Deep Ensembles — what applies to our reconstruction

(no notes)

## Slide 2: Where we start: the model has no uncertainty to report

If the softmax question comes up: a classifier's softmax output is routinely read as confidence and the paper's Fig. 1 shows why that is wrong — a point far outside the training data can still be given probability 1. Confidence in a class is not confidence in the prediction. The regression case here is starker: there is not even a misleading number to point at.

## Slide 3: What the paper proposes instead  (their Eq. 1)

The two terms are what makes this work. On its own the first drives sigma to zero and the model claims certainty everywhere; on its own the second inflates sigma and the model claims nothing. Their sum has one minimum, at sigma equal to the actual error.

The paper is explicit that this costs accuracy — it optimises NLL, not MSE. Their Table 1 shows RMSE slipping against squared-error methods by a median of about 5.6%.

## Slide 4: The second thing worth taking: what the uncertainty is made of

Be careful with the vocabulary if it is challenged. 'Aleatoric' strictly means irreducible given the measurement process — sensor noise qualifies, a cell no robot visited does not, since more robots would reduce it. Both nevertheless land in the first term, because a single network reports large sigma in either case. The split of the total variance is exact; the naming is a reading.

The second term is the cleaner of the two: disagreement between models fitted to the same data is unambiguous, and it is what the paper's out-of-distribution experiments measure — it grows on inputs unlike anything seen in training.

## Slide 5: What we would not take, and what it costs

Open question for the meeting, and the one thing only the supervisor can settle: deep ensembles is an uncertainty recipe that applies to any regressor, not a third reconstruction method. If the three-way benchmark is meant to compare three independent RECONSTRUCTION methods, that is a different literature search and it should start now.

If asked why not a Bayesian neural network: the paper's own argument is cost and intrusion — parameters double, the training procedure changes, and on their benchmarks it did not come out ahead.
