# Blind-region covariance calibration

Checkpoint: `covonly_pedpred5_raw_floor05_r32_s0/best.pt`

Base covariance calibration:

```text
factor covariance scale   = 1.10
diagonal covariance scale = 0.75
```

## One-step diagnosis (4,096 held-out pairs)

A uniform blind-region inflation was not supported by the data.  The baseline
posterior had 97.13% coverage for a nominal 90% interval in walkable blind
cells.  Lowering the blind diagonal standard deviation multiplier improved
CRPS, while increasing it made CRPS and coverage worse.

The preferred multiplier varied with activity: approximately 0.5 for low,
0.65 for medium, and 0.8 for high activity under marginal NLL/coverage
criteria.  This indicates state-dependent miscalibration rather than a single
blind-region scale error.

## Sequential predicted-empty gate

The tested gate uses only information available at inference time:

```text
if cell is walkable, blind, and predicted density < EMPTY_DENSITY:
    diagonal standard deviation *= 0.4
else:
    retain the baseline covariance
```

Weighted results over seven validation days, 5,000 frames per day (100-frame
warm-up):

| Scope | Method | RMSE | CRPS | Gaussian NLL | spread/skill | coverage 90% |
|---|---:|---:|---:|---:|---:|---:|
| walkable-blind | baseline | 0.583798 | 0.164465 | 0.535135 | 0.561507 | 0.932830 |
| walkable-blind | empty gate 0.4 | 0.583723 | 0.144079 | 2.737587 | 0.470957 | 0.927793 |
| defined-blind | baseline | 0.972379 | 0.333120 | 2.510837 | 0.399474 | 0.830814 |
| defined-blind | empty gate 0.4 | 0.972169 | 0.321818 | 10.778004 | 0.363838 | 0.816801 |

The gate improved walkable-blind CRPS by 12.4% and improved CRPS on every one
of the seven days.  It also improved defined-blind CRPS by 3.4%, so the gain is
not solely an obstacle/undefined-cell artifact.  Analysis RMSE was effectively
unchanged, as expected for a change that only rescales diagonal covariance at
unobserved cells.

The gate is not a complete replacement for the baseline uncertainty: Gaussian
NLL becomes much worse, revealing rare large errors that the sharper central
distribution cannot cover.  Keep the baseline as the primary Gaussian
covariance and treat the gate as a CRPS-optimal reporting calibration until a
heavy-tail-aware model is validated.

## Decision

The original hypothesis, "blind cells need uniform uncertainty inflation," is
rejected for the current five-frame model.  The remaining issue is a
distribution-shape mismatch: typical predicted-empty cases prefer a sharp
distribution, while rare/active cases require heavy tails or larger variance.
The next training experiment should target that mismatch rather than apply a
uniform blind loss.
