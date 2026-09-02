# 演讲备注 / Speaker notes

## Slide 1: Predictive uncertainty for crowd-field reconstruction

Four claims, one per slide after this: accuracy is a tie, the uncertainty is on a usable scale where it matters, it cannot say which individual cell is wrong, and it is an order of magnitude faster.

The EnKF appears only on the accuracy and timing slides. Its own uncertainty is not compared here — a fair comparison would need its inflation retuned, and this project does not change the EnKF's configuration.

## Slide 2: Accuracy — a tie where it counts

Lead with the tie so the rest is not read as advocacy.

7 held-out Sundays, k=1, identical observation realisation for both methods. The gap on OBSERVED cells is where the aggregate difference comes from.

## Slide 3: Is it the right size? It over-states by about 2.5x

Three cases the audience already understands: the robots either measured that cell or they did not. Two magnitudes in each, no ratio and no coverage curve — both of those say the same thing and both need defining first.

The third bullet does not need expanding unless asked. If it is: sigma is 0.487 on empty unobserved cells against an actual error of 0.114, and 0.507 on occupied ones against an actual error of 0.519. Sigma barely moves while the real error changes 4.5x, so the aggregate is dragged by the empty majority, which is 87% of unobserved cells. The figure is check_outputs/eval/mt_split.png.

If asked 'why not scale sigma down' — the factor would be set by that 87% majority and it takes the occupied cells from about right to badly over-confident. One factor cannot serve two regimes that differ by 4.5x.

PREPARED ANSWER, no longer on a slide: 'can you use sigma to flag which individual cells are unreliable?' No. Ranking cells by sigma and by their true error agree at 0.036 / 0.035 / 0.002 / 0.040 across the four channels, where 1 would be perfect and 0 unrelated — so within a channel it is close to a random ordering. Being the right SIZE and knowing WHERE are independent, and we have the first only. The uncertainty is therefore usable at the level of regions and channels, not single cells. Figure: check_outputs/eval/mt_agreement.png.

## Slide 4: Inference time

Interleaving matters: no node was free, so contention becomes a reported spread instead of a hidden bias.

The EnKF pays a Kalman update per observed frame, so k=4 costs less than k=1. Ours runs a fixed number of iterations regardless of how sparse the observations are.

If inference cost ever binds, the five networks are the first thing to reconsider — they cost 4.6x one network.
