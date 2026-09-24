# methods/enkf — localised ensemble Kalman filter on the ATC crowd field

> Part of [Partial Observation Comparison](../../PROJECT_OVERVIEW.md) — see the project overview for the problem statement, the scoring conventions, and where each final model comes from.

The one method in the final comparison that is a **filter** rather than a
reconstruction network: a 100-member ensemble is propagated by a neural forecast
model and corrected by a localised Kalman analysis on every frame. Its
uncertainty is the ensemble spread.

## The final EnKF (row "EnKF" in the results)

| Part | What | Where |
|---|---|---|
| Forecast model | PedPred3, 5 frames in → 5 frames out, trained on the 32 training days; epoch 38 chosen on the full validation split | `runs/pedpred3_5to5_clip_s0/dyn150_best.pt` (trained by `lcskf/dynamics/train.py`) |
| Process noise | **structured**: whole one-step forecast-residual fields sampled from a bank of 8192 built on training days only, scaled per channel, AR(1) in time | `enkf_opt/experiments/build_residual_q_bank.py` |
| Filter | 100 members, localisation radius 7, inflation 1.02, cross-channel coupling; runs on GPU | `enkf_opt/experiments/eval_structured_q_gpu.py` |
| Settings | noise scale 1.5, AR(1) ρ 0.5, per-channel and blind-cell noise scales — selected on the validation days | `FINAL_CONFIG["enkf"]` in `supervisor_evaluation/evaluate.py`; selection record in `enkf_opt/experiments/outputs/optimization_report.md` |

`supervisor_evaluation/evaluate.py` runs this filter on the seven test days (about
7 ms per frame on one V100, a few minutes per day), discards a 500-frame warm-up,
and scores its analysis mean and ensemble spread next to the other methods.
Both packaged files (`pedpred3_5to5_epoch38.pt`, `enkf_residual_q_bank.npz`) are
in `supervisor_evaluation/models/`. Retraining commands are in the project
overview.

The forecast model's first training run diverged at epoch 30 (every iteration
non-finite from epoch 31); the reported run restarts from that run's epoch-29
weights with a fresh optimiser, gradient clipping at 1.0 and lr 5e-4, and was
stopped at epoch 131 of 150 once the validation loss had been flat for ~20
epochs (`runs/pedpred3_5to5_clip_s0/dyn150_pick.json`).

## Two copies of the original code, on purpose

| Directory | Role |
|---|---|
| `enkf_lab/` | **pristine, read-only** byte-for-byte copy of the EnKF baseline from the original `Partial_observation` project, trained surrogate weights included. Files are chmod 444 deliberately. This is the reference. |
| `enkf_opt/` | the copy we are allowed to modify. Changes to the original filter path must be **bit-identical** to `enkf_lab` on real data — `np.array_equal`, not `np.isclose`. The structured-noise GPU filter above lives in `enkf_opt/experiments/`. |

Verify before trusting any change to the original path in `enkf_opt/`:

```bash
python3 -m methods.enkf.checks.verify_enkf_opt --frames 12
# [verdict] PASS -- strict path bit-identical: True, unit checks: True
```

## History: the original filter's ensemble collapse

The original configuration — the vendored surrogate as forecast model and
independent Gaussian process noise, run on CPU (`checks/run_enkf_baseline.py`,
~53 CPU-hours per day, outputs in `check_outputs/enkf_k1_full/`) — has an
ensemble spread of only about 1% of its actual error, and a nominal 90% interval
that contains the truth 1.6% of the time. `checks/diag_enkf_spread_growth.py`
propagates a perturbed ensemble with no injected noise and no analysis: the spread
decays ~65% per step, verdict `"contractive"`. That forecast model damps
disagreement between members, so no amount of extra noise or inflation sustains a
spread.

Two lines answered it:

- **The final EnKF** (above) changes the forecast model and draws the process noise
  from real forecast residuals, so perturbations carry the spatial and
  cross-channel structure of actual forecast errors. Its spread no longer collapses;
  on average it is now slightly too large (spread/RMSE 1.21), mostly on the velocity
  channels.
- **[`lcskf/`](lcskf/README.md)**, the learned-covariance sequential Kalman filter,
  replaces the ensemble covariance with a U-Net-predicted `B = U Uᵀ + diag(d)`. A
  research line with its own README; **not in the final comparison**.

Numbers in `check_outputs/` and in the `lcskf` README come from earlier scoring
scripts, not the final evaluation protocol; the final numbers are in the repository
README.

## What the checks answer

| Question | Script |
|---|---|
| Is `enkf_opt/` still bit-identical to `enkf_lab/`? | `checks/verify_enkf_opt.py` |
| Same, for the gain-mode variants | `checks/verify_enkf_gain_mode.py` |
| Does an ensemble sustain spread, or collapse? | `checks/diag_enkf_spread_growth.py` |
| Where does the original filter's wall time go? | `checks/bench_enkf_split.py`, `checks/bench_enkf_opt.py` |
| Score an exported estimate of the original filter | `checks/score_enkf.py`, `checks/eval_uncertainty_enkf.py` |
| Export the robots' observations in the filter's format | `checks/export_obs_for_enkf.py` (also used by the final evaluation) |
