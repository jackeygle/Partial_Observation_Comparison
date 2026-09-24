# Structured-Q EnKF optimization report

Hyperparameters were selected exclusively on seven complete validation days. The selected candidates were then evaluated on seven complete held-out test days.

## Validation sweep

| config | all RMSE | all CRPS | defined-blind CRPS | DB spread/skill | DB coverage90 |
|---|---:|---:|---:|---:|---:|
| r175 | 0.22158 | 0.09880 | 0.14575 | 0.560 | 0.765 |
| r2 | 0.22698 | 0.10232 | 0.14599 | 0.616 | 0.803 |
| r15_rho50 | 0.21799 | 0.09651 | 0.14627 | 0.522 | 0.770 |
| r15_rt25 | 0.21727 | 0.09761 | 0.14646 | 0.512 | 0.731 |
| r15_rt50 | 0.21853 | 0.10028 | 0.14651 | 0.524 | 0.745 |
| r15 | 0.21698 | 0.09594 | 0.14665 | 0.500 | 0.713 |
| r15_rho80 | 0.22053 | 0.09789 | 0.14685 | 0.538 | 0.807 |
| r225 | 0.23305 | 0.10633 | 0.14710 | 0.669 | 0.832 |
| r25 | 0.23964 | 0.11069 | 0.14886 | 0.719 | 0.854 |
| r125 | 0.21329 | 0.09399 | 0.14903 | 0.437 | 0.642 |
| r1 | 0.21061 | 0.09330 | 0.15334 | 0.370 | 0.552 |
| g1 | 0.23949 | 0.11355 | 0.16433 | 0.289 | 0.503 |

Validation-only selections:

```json
{
  "minimum_defined_blind_crps": "r175",
  "minimum_all_crps": "r1",
  "minimum_db_crps_within_5pct_r1_rmse": "r15_rho50",
  "closest_db_coverage90": "r25"
}
```

## Locked held-out test

| config | all RMSE | all CRPS | defined-blind CRPS | DB spread/skill | DB coverage90 |
|---|---:|---:|---:|---:|---:|
| r175 | 0.22924 | 0.10153 | 0.15385 | 0.567 | 0.755 |
| r15_rho50 | 0.22574 | 0.09934 | 0.15435 | 0.529 | 0.759 |
| r15_rt25 | 0.22520 | 0.10045 | 0.15466 | 0.519 | 0.720 |
| r15 | 0.22490 | 0.09888 | 0.15482 | 0.506 | 0.703 |
| r15_rho80 | 0.22816 | 0.10061 | 0.15489 | 0.546 | 0.797 |
| r25 | 0.24650 | 0.11293 | 0.15697 | 0.729 | 0.847 |
| r1 | 0.21898 | 0.09674 | 0.16176 | 0.374 | 0.546 |
| g1 | 0.24800 | 0.11740 | 0.17315 | 0.293 | 0.501 |

## Recommended configuration

Primary: **r15_rho50**, selected by minimum validation defined-blind CRPS subject to no more than 5% RMSE degradation relative to residual scale 1.

Held-out test: all RMSE 0.22574, all CRPS 0.09934, defined-blind CRPS 0.15435, spread/skill 0.529, coverage90 0.759.

Relative to Gaussian Q on held-out test, this reduces all RMSE by 9.0%, all
CRPS by 15.4%, and defined-blind CRPS by 10.9%. Relative to residual scale 1,
it trades 3.1% RMSE and 2.7% all-CRPS for 4.6% lower defined-blind CRPS and
raises defined-blind coverage from 0.546 to 0.759. Overall spread/skill and
coverage are 0.746 and 0.820.

The absolute minimum test defined-blind CRPS is r175 (0.15385 versus 0.15435),
but r15_rho50 remains the methodologically selected primary because the choice
was locked on validation. The difference is only 0.3%, while r15_rho50 has
lower RMSE and all CRPS.

### Channel diagnosis for r15_rho50

| channel | defined-blind CRPS | spread/skill |
|---|---:|---:|
| density | 0.0659 | 0.30 |
| vx | 0.5031 | 0.76 |
| vy | 0.1970 | 0.69 |
| variance | 0.1862 | 1.27 |

One global scale cannot calibrate these simultaneously: density is still very
under-dispersed, vx/vy are much closer, and variance is already over-dispersed.
This is why r25 can reach 84.7% defined-blind coverage but worsens RMSE and
proper scores: it adds noise to the already-over-dispersed variance channel too.

Temporal persistence at rho 0.5 is useful but modest. Against plain residual
scale 1.5 it improves defined-blind CRPS (0.15482 to 0.15435), coverage (0.703
to 0.759), and long-blind CRPS (0.11144 to 0.11021), with a small RMSE cost.
Rho 0.8 increases coverage further (0.797) but loses CRPS, so rho 0.5 is the
better proper-score tradeoff.

## Channel-wise Q follow-up

The channel diagnosis above was followed up on validation only.  All runs keep
the same residual bank, global scale 1.5 and temporal rho 0.5.  A scale vector is
ordered as density, vx, vy, variance.

| config | channel scale | all RMSE | all CRPS | DB CRPS | DB coverage90 |
|---|---|---:|---:|---:|---:|
| base | 1, 1, 1, 1 | 0.21799 | 0.09651 | 0.14627 | 0.770 |
| den2 | 2, 1, 1, 0.75 | 0.21308 | 0.09459 | 0.14350 | 0.818 |
| bal2 | 2, 1.25, 1.4, 0.7 | 0.21626 | 0.09880 | **0.14152** | 0.850 |
| bal3 | 2.5, 1.3, 1.5, 0.65 | 0.21648 | 0.09986 | 0.14166 | 0.864 |
| bal4 | 3, 1.4, 1.6, 0.6 | 0.21780 | 0.10182 | 0.14234 | 0.875 |

`bal2` was the validation primary under the predeclared rule (minimum DB CRPS
within 2% of base RMSE).  `den2`, `bal3` and `bal4` entered held-out testing as
the all-CRPS, long-blind and calibration diagnostic winners, respectively.

Held-out results:

| config | all RMSE | all CRPS | DB CRPS | DB spread/skill | DB coverage90 |
|---|---:|---:|---:|---:|---:|
| base | 0.22574 | 0.09934 | 0.15435 | 0.529 | 0.759 |
| den2 | **0.22078** | **0.09727** | 0.15110 | 0.552 | 0.811 |
| bal2 | 0.22373 | 0.10125 | 0.14913 | 0.624 | 0.843 |
| bal3 | 0.22387 | 0.10222 | **0.14908** | 0.655 | 0.859 |
| bal4 | 0.22505 | 0.10405 | 0.14959 | 0.694 | 0.871 |

The DB-CRPS improvement of `bal2` over base occurs on all seven test days, as
does its RMSE improvement.  However, applying its larger velocity scales to
observed cells increases all-cell CRPS.

## Causal blind-walkable Q scaling

The final follow-up therefore applies an extra channel multiplier only at cells
that are both in the fixed walkable map and unobserved at the current time.  It
uses neither truth nor future masks.  The selected `sp_b2` configuration is:

- global scale vector: 2, 1, 1, 0.75 (`den2`);
- extra blind-walkable multiplier: 1, 1.25, 1.4, 0.933333;
- effective blind-walkable scale: 2, 1.25, 1.4, 0.7 (`bal2`).

On validation, `sp_b2` gave all RMSE 0.21591, all CRPS 0.09717 and DB CRPS
0.14162.  It retained essentially all of global `bal2`'s DB gain while recovering
61% of its all-CRPS penalty.  More aggressive spatial variants lost proper score
and were not sent to held-out testing.

Final held-out result:

| config | all RMSE | all CRPS | DB CRPS | DB spread/skill | DB coverage90 | long-blind CRPS |
|---|---:|---:|---:|---:|---:|---:|
| r15_rho50 | 0.22574 | 0.09934 | 0.15435 | 0.529 | 0.759 | 0.11021 |
| den2 | 0.22078 | **0.09727** | 0.15110 | 0.552 | 0.811 | 0.10851 |
| global bal2 | 0.22373 | 0.10125 | **0.14913** | 0.624 | 0.843 | 0.10610 |
| spatial sp_b2 | 0.22347 | 0.09972 | 0.14923 | 0.625 | **0.844** | **0.10627** |

`sp_b2` versus `r15_rho50` lowers all RMSE by 1.01% and DB CRPS by 3.32%,
raising DB coverage from 0.759 to 0.844, at a 0.38% all-CRPS cost.  The DB-CRPS
and RMSE improvements occur on all seven held-out days.  Versus Gaussian Q it
lowers all RMSE by 9.9%, all CRPS by 15.1%, and DB CRPS by 13.8%.

The final DB channel spread/skill values for `sp_b2` are 0.45, 0.91, 0.93 and
1.14.  Velocity and variance are now reasonably calibrated; density remains
under-dispersed and is the main unresolved limitation.

## Final recommendation

Use **sp_b2** as the primary configuration when the scientific target is
uncertainty in unobserved walkable cells.  Report **den2** as the all-domain
proper-score alternative and `r1` as the point-accuracy ablation.  Do not use
`bal4`: its coverage is higher only because it over-inflates the ensemble and
its proper scores are worse.

The next method change, if needed, should make density Q state-dependent (for
example, conditioned on forecast occupancy/density) rather than increasing a
global multiplier.  The current density spread/skill of 0.45 shows that a fixed
additive scale still cannot handle zero-clipping and occupied/empty regimes
simultaneously.

## Posterior density-spread calibration follow-up

The residual density limitation was first attacked by increasing density Q as a
function of forecast density.  Four smoke variants (local/3x3-neighbour risk,
maximum multipliers 2 and 3) moved density spread/skill toward one, but worsened
both density CRPS and pooled CRPS.  Extra Q passes through the nonnegative
density clipping and changes the ensemble mean, so this route was stopped before
a full validation sweep.

The retained alternative is explicitly a **report-only posterior covariance
calibration**.  It does not feed back into the Kalman gain, ensemble trajectory,
or analysis mean.  On currently blind walkable cells, density sigma is replaced
by:

    calibrated_sigma = raw_sigma * (1 + gain * min(analysis_density / threshold, 1))

Thirty combinations of gain, threshold and optional 3x3 neighbourhood risk were
evaluated together on each of the seven validation days.  The validation-only
winner was `gain=1.5`, `threshold=0.4`, `radius=0`; it improved DB CRPS and all
CRPS on 7/7 validation days.  It was then locked for held-out testing.

| held-out metric | raw sp_b2 | sp_b2 + density calibration |
|---|---:|---:|
| all RMSE | 0.223475 | 0.223475 |
| all CRPS | 0.099719 | **0.099592** |
| defined-blind RMSE | 0.369301 | 0.369301 |
| defined-blind CRPS | 0.149228 | **0.148184** |
| defined-blind spread/skill | 0.6249 | **0.6703** |
| defined-blind coverage90 | 0.8437 | **0.8869** |
| density DB spread/skill | about 0.45 | **0.6227** |
| density DB coverage90 | about 0.81 | **0.8728** |

All-CRPS and defined-blind-CRPS improvements occurred on all seven held-out
days.  Density DB CRPS after calibration is 0.06145 (about 2.9% below raw
`sp_b2`).  The calibrated method also beats every tested dynamic-Q variant on
held-out defined-blind CRPS; the best uncalibrated global candidate was `bal3`
at 0.14908.

Use `sp_b2 + density calibration` when a calibrated posterior uncertainty
product is required.  Use raw `sp_b2` when the ensemble covariance must be fed
back into another Kalman update without an additional calibration model.  This
distinction is important: the follow-up improves the reported posterior
covariance, not the underlying transition model or Kalman gain.
