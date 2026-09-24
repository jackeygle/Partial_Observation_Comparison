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

## Directory layout

| path | role |
|---|---|
| `enkf_opt/pedpred/` | the filter and PedPred3 code, adapted from the original `Partial_observation` EnKF baseline (see [`enkf_opt/README.md`](enkf_opt/README.md)) |
| `enkf_opt/experiments/eval_structured_q_gpu.py` | **the final filter**: 100 members, structured residual noise, localisation, ensemble-space Kalman update, all on GPU |
| `enkf_opt/experiments/build_residual_q_bank.py` | builds the residual noise bank from training days |
| `enkf_opt/experiments/outputs/` | the tuning record: `optimization_report.md` and the per-stage `*_summary.json` |
| `enkf_opt/apt-ibex_train_model_28D.pth` | the original baseline's forecast model (14 MB), used by the collapse diagnosis below |
| `lcskf/dynamics/` | trains the final forecast model (PedPred3 5→5) |
| `checks/export_obs_for_enkf.py` | writes the robots' observations in the filter's format (used by the final evaluation) |
| `checks/diag_proc_scale_sweep.py` | the collapse diagnosis: the original filter with only its noise multiplier changed |
| `check_outputs/eval/proc_scale_*.json` | its results |
| `runs/pedpred3_5to5_clip_s0/`, `runs/pedpred3_5to5_s0/` | training logs of the forecast model (checkpoints gitignored) |

The original read-only baseline copy (`enkf_lab/`), the bit-identity checks and
benchmarks of the engineering speed-ups, the original CPU driver, and every other
experiment are in the git tag `archive-full-2026-09-24`.

## History: the original filter's ensemble collapse, and what fixed it

The original configuration — the vendored surrogate as forecast model and
independent Gaussian process noise, run on CPU (~53 CPU-hours per day) — has an
ensemble spread of only about 1% of its actual error, and a nominal 90% interval
that contains the truth 1.6% of the time.

**The main cause is a constant.** The vendored `forecast()` injects
`0.01 * proc_noise_vec`, i.e. 1% of the process noise that `PROC_STD` (the
surrogate's own measured one-step error) calls for. The forecast damps
perturbations (~65% per step with no noise at all, verdict `"contractive"`), so the
spread settles at a level set by the injection.
Changing only that multiplier (`checks/diag_proc_scale_sweep.py`, `atc-20130811`,
2000 frames):

| noise multiplier | spread/RMSE | 90% coverage | CRPS |
|---|---:|---:|---:|
| 0.01 (original) | 0.014 | 2% | 0.122 |
| 0.1 | 0.13 | 12% | — |
| 1 | 0.96 | 93% | 0.097 |

An earlier version of this README concluded that no amount of noise could fix the
collapse; that diagnostic only showed that the forecast damps perturbations when
none are added. It was wrong.

**Full-strength independent noise is not enough, though.** It fixes the size of
σ̂ but not where it is large: the injected noise is spatially uniform, and on the
velocity channels σ̂ is uncorrelated with the actual error (Spearman ≈ 0). The
final EnKF therefore draws whole residual fields from real one-step forecast
errors on training days (`enkf_opt/experiments/`), which carry the spatial and
cross-channel structure of real errors: on seven full test days this lowers CRPS
by 17.6% against full-strength Gaussian noise, on every day, and turns the vx
spread–error correlation from −0.20 to +0.31. The noise scales were then tuned on
validation days (`optimization_report.md`).

The learned-covariance sequential Kalman filter (LCSKF), a separate answer to the
same problem with a U-Net-predicted covariance, is **not in the final comparison**;
it is in the archive tag.
