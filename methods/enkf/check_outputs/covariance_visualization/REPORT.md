# Learned covariance visualization report

Model: `covonly_pedpred5_raw_floor05_r32_s0/best.pt`

Mean model: `pedpred3_5to5_s0/best.pt`

The report samples 2,048 held-out validation cases and visualizes the cases at
the 10th, 50th, and 90th percentiles of walkable-region crowd activity.  For
each case it explicitly reconstructs the full 1,728 by 1,728 covariance
matrix

```text
B = U U^T + diag(d)
```

and saves it in `covariance_low.npz`, `covariance_medium.npz`, or
`covariance_high.npz`.

## Main observations

- The predicted covariance is state-dependent.  The covariance trace rises
  from 29.80 to 36.51 to 43.95 across the selected low-, medium-, and
  high-activity states.
- The pairwise relative Frobenius differences are 1.58 (low/medium), 1.60
  (low/high), and 1.49 (medium/high).  Thus the network is not returning a
  nearly constant covariance template.
- The low-rank term contributes about 10--12% of the total trace.  Most
  marginal variance comes from the diagonal term, while the low-rank term is
  responsible for all learned off-diagonal spatial and cross-channel
  covariance.
- In the high-activity case, the leading low-rank eigenvalues are 2.02, 0.56,
  0.39, and 0.30.  The corresponding modes form coherent patterns along the
  walkable regions rather than isolated pixel noise.
- A density variable at one anchor has signed correlations with distant
  density, velocity, and variance variables.  The single-observation Kalman
  gain therefore spreads a local density innovation spatially and across
  channels.

These visualizations support the earlier shuffle test: the covariance model
has learned conditional off-diagonal structure.  They do not by themselves
prove perfect calibration; coverage, CRPS, and sequential-analysis scores
remain the appropriate calibration and downstream tests.

The sign of an individual eigenvector is arbitrary.  Interpret its spatial
pattern and subspace, not whether one plotted mode happens to be red or blue.

## Figures

- `06_state_comparison.png`: low/medium/high activity comparison with shared
  color scales.
- `01_state_variance_*.png`: forecast mean and marginal prior standard
  deviation.
- `02_anchor_correlations_*.png`: density-anchor correlations at three
  spatial locations.
- `03_cross_channel_*.png`: all 16 source/target channel combinations at the
  primary anchor.
- `04_kalman_influence_*.png`: update induced by one density observation.
- `05_lowrank_modes_*.png`: the first four learned covariance modes.
- `summary.json`: frame identities and numerical summaries.
