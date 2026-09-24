# Structured-Q EnKF experiment

This experiment keeps the original 100-member localized EnKF.  The Kalman
background covariance is still the sample covariance of those members.  The
only methodological change is the distribution used for additive process
perturbations:

- `gaussian`: independent, per-cell Gaussian draws using the existing
  `PROC_STD` values.
- `residual`: complete one-step PedPred3 residual fields sampled from training
  days.  A draw therefore retains learned spatial and cross-channel error
  structure.

`build_residual_q_bank.py` constructs 8192 zero-mean residual fields from all
32 training days.  No validation or test day is used.  During evaluation, the
four residual-channel standard deviations are matched to the same `PROC_STD`
as the Gaussian arm.  This makes the comparison primarily about covariance
structure rather than a larger noise magnitude.

Both arms use 100 members, localization radius 7, corrected four-channel
localization, bias EMA, inflation 1.02, and the original `pinv` gain path.  The
formal run uses 2000 frames and discards the first 500.

## Formal result (test day atc-20130811)

| Q configuration | all RMSE | all CRPS | all spread/skill | defined-blind CRPS | defined-blind coverage 90% |
|---|---:|---:|---:|---:|---:|
| Gaussian, scale 1 | 0.20422 | 0.09520 | 0.739 | 0.10659 | 0.541 |
| Residual, scale 0.50 | 0.16954 | 0.07838 | 0.378 | 0.11578 | 0.354 |
| Residual, scale 0.75 | 0.16958 | **0.07561** | 0.535 | 0.10602 | 0.493 |
| Residual, scale 1.00 | 0.17137 | 0.07565 | 0.673 | 0.09912 | 0.619 |
| Residual, scale 1.25 | 0.17497 | 0.07767 | 0.796 | 0.09489 | 0.710 |
| Residual, scale 1.50 | 0.17957 | 0.08075 | 0.900 | **0.09249** | **0.773** |
| Residual, scale 1.00 + RTPS 0.5 | 0.17091 | 0.07774 | 0.843 | 0.09714 | 0.674 |

Interpretation: structured residual perturbations clearly outperform
independent Gaussian perturbations.  Scale 0.75 is the best pooled CRPS/point
accuracy compromise; scale 1.50 is better calibrated in unobserved defined
cells.  RTPS improves calibration but does not dominate direct scale tuning.
The velocity-x spread ranking changes from harmful (Spearman -0.197) to useful
(about +0.30 to +0.33).  Velocity-y remains weakly negative, so the covariance
structure is improved but not solved.

The `pinv` analysis failed on 2--21 of 2000 frames depending on the arm; those
frames use forecast-only fallback.  This numerical issue must be considered
before selecting scale 1.25 or 1.50 as a final configuration.

## Seven-day confirmation (2000 frames per test day)

| Q configuration | all RMSE | all CRPS | defined-blind CRPS | defined-blind coverage 90% | `pinv` fallbacks / 14000 |
|---|---:|---:|---:|---:|---:|
| Gaussian, scale 1 | 0.20558 +/- 0.02521 | 0.09565 +/- 0.00983 | 0.10727 +/- 0.03629 | 0.544 | 48 |
| Residual, scale 1 | **0.17301 +/- 0.02607** | **0.07603 +/- 0.00893** | 0.09946 +/- 0.03471 | 0.636 | 58 |
| Residual, scale 1.5 | 0.18151 +/- 0.02436 | 0.08112 +/- 0.00740 | **0.09350 +/- 0.03263** | **0.779** | 101 |

Residual scale 1 beats Gaussian on all RMSE, all CRPS, defined CRPS, and
defined-blind CRPS on every one of the seven days.  Its mean changes are
-16.1%, -20.6%, -20.3%, and -7.6%, respectively.  Scale 1.5 also wins all four
metrics on all seven days, with a larger defined-blind CRPS improvement
(-13.2%) but more `pinv` failures.  Mean velocity-x spread/error Spearman rises
from -0.200 (Gaussian) to +0.311 / +0.334 (residual 1 / 1.5).  Velocity-y remains
weak (-0.078 / -0.059), confirming the unresolved channel-specific limitation.

The machine-readable aggregation is `outputs/full7_summary.json`.  A second
stage evaluates all frames of all seven days with the mathematically equivalent
ensemble-space gain; it is kept separate from this `pinv` table.

## Commands

```bash
sbatch methods/enkf/enkf_opt/experiments/submit_build_residual_q.sbatch
sbatch methods/enkf/enkf_opt/experiments/submit_eval_structured_q.sbatch \
  --noise-kind residual --scale 1 --frames 2000 --warmup 500 --gain-mode pinv
```

## CUDA backend

`eval_structured_q_gpu.py` keeps the forecast model, all 100 members, structured-Q
sampling, localization, ensemble-space Kalman solve, bias EMA, observation
perturbations and RTPS on CUDA.  The posterior mean and spread alone are copied
back for scoring.  `enkf_opt/pedpred/grid.py` was fixed so `GridData` preserves
the producing tensor's device/dtype instead of silently allocating network output
on CPU.

On a V100, the 2000-frame verification took 13.6--13.9 seconds, versus 506--728
seconds for the original CPU `pinv` jobs.  Results were nearly identical:

| arm | CPU RMSE / CRPS | GPU RMSE / CRPS |
|---|---:|---:|
| Gaussian scale 1 | 0.20422 / 0.09520 | 0.20418 / 0.09518 |
| Residual scale 1 | 0.17137 / 0.07565 | 0.17142 / 0.07562 |

```bash
sbatch methods/enkf/enkf_opt/experiments/submit_eval_structured_q_gpu.sbatch \
  --noise-kind residual --scale 1 --frames 100000 --warmup 500
```

## Seven complete test days (CUDA backend)

All available frames (about 35k--43k per day) were evaluated.  CPU and CUDA
aggregates agree to the displayed precision and both had zero analysis failures.

| Q configuration | all RMSE | all CRPS | defined-blind CRPS | defined-blind spread/skill | defined-blind coverage 90% |
|---|---:|---:|---:|---:|---:|
| Gaussian, scale 1 | 0.24800 +/- 0.01061 | 0.11740 +/- 0.00477 | 0.17315 +/- 0.01167 | 0.293 | 0.501 |
| Residual, scale 1 | **0.21898 +/- 0.01050** | **0.09674 +/- 0.00442** | 0.16176 +/- 0.01126 | 0.374 | 0.546 |
| Residual, scale 1.5 | 0.22490 +/- 0.01002 | 0.09888 +/- 0.00384 | **0.15482 +/- 0.01072** | **0.506** | **0.703** |

Both residual configurations beat Gaussian on all RMSE, all CRPS, defined CRPS
and defined-blind CRPS on all seven days.  Scale 1 improves all RMSE/CRPS by
11.7%/17.6%; scale 1.5 improves defined-blind CRPS by 10.6%.  Unlike the early
2000-frame window, full-day mean spread/error ranking is positive on every
channel for residual scale 1.5: density 0.799, vx 0.438, vy 0.096 and variance
0.728.  Calibration is nevertheless still under-dispersed in defined blind
cells (spread/skill 0.506 and 70.3% rather than nominal 90% coverage), which is
the next defect to target.
