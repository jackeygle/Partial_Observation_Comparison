# lcskf — the learned-covariance sequential Kalman filter

> Part of [methods/enkf](../README.md). The EnKF next door is a filter with no
> learned parameters; this is the answer to its one fatal property.

## Why this exists

The EnKF's uncertainty *is* its ensemble spread, and that spread collapses to
about 0.011 of the actual error. `../checks/diag_enkf_spread_growth.py` shows why:
propagated through PedPred3 with no injected noise and no analysis, the spread
decays ~65% per step — verdict `"contractive"`. A deterministic forecast model can
only sustain an ensemble if its dynamics *amplify* differences. This one damps
them, so no amount of inflation or process noise fixes it.

So this method throws the ensemble away. A U-Net predicts the background
covariance directly, as a low-rank-plus-diagonal factorisation, and a
differentiable Kalman analysis turns it into a posterior:

```text
x_{t-4:t} ──► PedPred3 (ConvLSTM) ──► mean_{t+1}
                                        │
[x_t, mean_{t+1}, mean_{t+1}−x_t, static_mask] ──► covariance U-Net ──► U, d
                                        │
              B = U Uᵀ + diag(d) ──► differentiable Kalman update ──► analysis_{t+1}
```

Result on the 7-day test split, walkable blind cells, against a constant-σ null:
the EnKF is **26.0% worse** than the null; this is **28.3% better**. Spread/skill
goes 0.010 → 0.500, and coverage of a nominal 90% interval goes 1.6% → 92.9%.

## The one idea you need

**There is no ground truth for a covariance.** You have `x_true`; nobody can label
a 1728×1728 matrix. So nothing here is fitted to B directly. The loss is the
error *after* the Kalman update:

```
loss = mean( ((analysis − x_true) / STATE_SCALE)² )
```

and the gradient reaches `(U, d)` only by flowing backwards through the analysis.
That is the whole reason `kalman.py` exists: the update has to be differentiable,
and it has to be cheap enough to sit in a training loop. Woodbury's identity
reduces every solve from the number of observed state entries (~1,000) to the
rank (8–32), so the dense covariance is never formed.

A covariance *can* be trained standalone — `kalman.gaussian_nll` fits B to the
forecast error's likelihood with no Kalman update at all. It was swept and it
lost; see the ablation table below.

## Layout

| Path | What |
|---|---|
| `dynamics/` | PedPred3 (vendored from `../enkf_lab`, unchanged), its grid_cache pair loader, and its pretraining script |
| `covariance.py` | the two-head U-Net: `factor` (B,R,C,H,W) and positive `diagonal` |
| `kalman.py` | `analysis_update`, `posterior_diag`, `gaussian_nll` — exact, differentiable, Woodbury |
| `train.py` | trains `covariance.py`, optionally fine-tuning `dynamics/` through the same loss |
| `filter.py` | runs a trained pair as a sequential filter over a whole day, exporting estimate + σ̂ |
| `checks/` | diagnostics, ablations, calibration and the sequential evaluation driver |

Artifacts stay with the rest of the method: checkpoints in `../runs/`, result JSON
and exported fields in `../check_outputs/`. The uncertainty scorer is shared with
the EnKF baseline and lives at `methods.enkf.checks.eval_uncertainty_enkf` — it
scores any directory of `est_<day>.npz`, whoever wrote it.

## The three training stages

| Stage | PedPred3 | covariance U-Net | Loss |
|---|---|---|---|
| ① pretrain dynamics | **learning** (alone) | — | vendored "mean total weighted NLLL" |
| ② pretrain covariance | **frozen** (`no_grad`) | random init → learning | analysis error MSE |
| ③ joint | **unfrozen**, lr 1e-5 | warm-started from ② , lr 1e-4 | analysis error MSE (unchanged) |

```bash
# ① the mean forecast (1 frame in, 1 frame out -- what runs/surrogate_mean_s0 is)
sbatch methods/enkf/lcskf/sbatch/submit_dynamics.sbatch --seed 0

# ② covariance only, mean frozen
sbatch methods/enkf/lcskf/sbatch/submit_lcskf.sbatch train \
  --rank 32 --width 32 --batch 128 --lr 1e-3 --epochs 20 --seed 0 \
  --out runs/lowrank_a_r32_s0

# ③ joint -- this produces the published checkpoint
sbatch methods/enkf/lcskf/sbatch/submit_lcskf.sbatch train \
  --cov-ckpt runs/lowrank_a_r32_s0/best.pt \
  --mean-ckpt runs/surrogate_mean_s0/best.pt \
  --mean-train full --mean-lr 1e-5 --input-frames 5 \
  --rank 32 --width 32 --batch 32 --lr 1e-4 --epochs 20 --eval-pairs 2048 --seed 0 \
  --out runs/joint_j4_5f_r32_nofcst_s0
```

Paths passed to these batch scripts are relative to `methods/enkf/`, not to the
repository root: the script `cd`s there before running Python. Submit the
`sbatch` command itself from the repository root, as always.

In ③ PedPred3 stops being optimised to forecast well and starts being optimised
to be a good *component of a filter*. The training log shows it plainly: over 20
epochs `forecast_mse_norm` moves 0.170 → 0.149 while `analysis_mse_norm`, the
thing actually in the loss, moves 0.0588 → 0.0548.

Jobs request three hours. Five minutes before the limit the batch script
resubmits itself with `--resume`; `train.py` checkpoints after every epoch, so at
most the in-progress epoch is repeated.

## Calibration: α is convention-dependent, and this is the trap

Inference applies one scalar: `factor *= √α`, `diagonal *= α`, i.e. `B → αB`. It
is pure post-processing — no retraining.

**Fit it under the convention you will report in.** The seven-model one-step grid
over 4096 held-out pairs picks **α = 0.5**. A full-day sequential run picks
**α = 2**, and at α = 2 every column improves at once (CRPS −4.8%, spread/skill
0.249 → 0.500, coverage 83.3% → 92.9%). Sequential error accumulates and one-step
scoring cannot see it. Reusing the grid's α in a sequential table is not a small
approximation; it is the difference between a badly overconfident σ̂ and a
calibrated one.

```bash
# one-step grid over models x alpha
sbatch methods/enkf/lcskf/sbatch/submit_lcskf.sbatch compare --model a32=<ckpt> --alphas 0.02,0.05,0.1,0.25,0.5,1,2,4 ...
# full-day sequential sweep, and the export at the chosen alpha
sbatch methods/enkf/lcskf/sbatch/submit_sequential.sbatch \
  --calibration check_outputs/eval/controlled_grid_calibration_4096.json \
  --new-checkpoint runs/joint_j4_5f_r32_nofcst_s0/best.pt \
  --obs check_outputs/enkf_k1_full/obs_atc-20130811.npz \
  --new-alphas 2 --warmup 100 --skip-old --skip-mean \
  --export-npz check_outputs/recon_j4nofcst_a2 --out <json>
# score any exported directory
python3 -m methods.enkf.checks.eval_uncertainty_enkf --k 1 ...
```

## What was ablated, and what it cost

Seven models, every pair differing by exactly one factor, scored on 4096 held-out
one-step pairs (`../check_outputs/eval/controlled_grid_calibration_4096.json`),
posterior / walkable_blind:

| Model | differs by | RMSE @α=2 | CRPS @α=2 |
|---|---|---|---|
| a32 | baseline: frozen mean, 1 history frame | 0.35081 | 0.09753 |
| nll0 | a32 at batch 32 (the NLL arm's control) | 0.35126 | 0.09997 |
| nll001 | `nll_weight` 0.01 | 0.35217 | 0.10358 |
| nll003 | `nll_weight` 0.03 | 0.35148 | 0.10649 |
| j4 | joint, 5 frames, `forecast_weight` 0.1 | 0.34059 | 0.10055 |
| j4frozen | j4 with the mean frozen | 0.35825 | 0.10169 |
| **j4nofcst** | **j4 with `forecast_weight` 0** | **0.33983** | **0.09638** |

- **A forecast-MSE auxiliary term hurts.** j4 → j4nofcst wins on both columns.
- **The mean must be unfrozen.** j4frozen is the worst row in the table.
- **A forecast-NLL term hurts, monotonically.** Training B against its own
  likelihood instead of the analysis error is strictly worse.
- **Rank barely matters.** r8/r16/r32 score 0.06638/0.06656/0.06675 blind
  normalized MSE. The diagnostics say why: `effective_rank_mean` is 1.0–1.6 and
  `lowrank_trace_fraction` is 0.96 — 32 directions are offered and about one is
  used. `covariance.py` scales the factor by `1/√rank`, so the initial total
  variance is rank-independent by construction.
- **Bias correction hurts** everywhere it was tried.
- **Splitting α into separate factor/diagonal scales is not worth it**: the best
  split beats the tied scale by 0.03%.

`--nll-weight`, `--forecast-weight` and `--mean-train head` were removed in
9825c81 once no surviving checkpoint used them, so the losing arms above are
reproducible only from before that commit.
