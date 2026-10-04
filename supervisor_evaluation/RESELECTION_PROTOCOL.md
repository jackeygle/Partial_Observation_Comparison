# Re-selection of the design choices on the validation split

Written and committed **before** any of the runs below was trained or scored.

## Why

Several design choices of the final comparison were made, in the exploratory phase,
by comparing candidates on the seven **test** days: Senseiver-G's latent array and time
window, 4DVarNet's prior width, DINCAE's full-field loss, the 4DVarNet uncertainty
design, and parts of the EnKF's process noise. Each final model's *epoch* was already
chosen on the validation split; the *designs* were not. This protocol re-makes every
one of those choices on the validation split, with the decision rules fixed here.

## Rules common to all methods

1. Candidates are trained on the 32 training days only, with exactly the recipe of the
   existing model they are compared with (data, robot routes, seed, epochs,
   hyperparameters). Routes: `trajectory_mode=fixed` for every Senseiver, 4DVarNet and
   DINCAE candidate, as for the existing models of those methods.
2. One seed per candidate.
3. A candidate's score is its blind walkable pooled RMSE (four channels, predictions
   clipped to the physical bounds) on all seven validation days, every frame, at its
   best epoch on that same score -- the scope used for every final model's epoch.
4. Candidates are compared on the validation split only. The seven test days are not
   touched until all six final models are fixed; they are then evaluated once.
5. If the validation winner is the current design, the current final model is kept.
   Otherwise the final model of that method becomes the winner, and the thesis results
   change accordingly.

## Senseiver-G

Candidates, 40 epochs, seed 123, every epoch saved:

| candidate | run |
|---|---|
| A40: abstract latent, decoder, k = 1 | epochs 0-39 of `runs/senseiver_A_full` (constant learning rate, so identical to a 40-epoch run) |
| G-direct, k = 1 | new: `runs/reselect/gdirect_k1_s123` |
| G-direct, k = 4 | new: `runs/reselect/gdirect_k4_s123` |
| G-direct, k = 8 | new: `runs/reselect/gdirect_k8_s123` |
| G-direct, k = 16 | epochs 0-39 of `runs/capacity/base32_k16_s123_full` |

Step 1 (latent): A40 vs G-direct k = 1; the lower validation RMSE wins.
Step 2 (window), on the step-1 winner: k in {1, 4, 8, 16}, fixed now and not extended
afterwards. The winner is the lowest validation RMSE among the windows whose every
channel RMSE is at most 5% above that of k = 1. (If step 1 picks A40, step 2 is
re-planned before anything is trained.) If the result is G-direct k = 16, the current
final model (60 epochs, epoch 57) is kept.

## 4DVarNet

Candidates: prior width/temporal kernel 32/3, 64/5, 96/5; init seed 0, data seed 0,
recipe of `runs/varnet_mse5_h96_s0` (150 epochs, iteration schedule 5/10/15/20 at
epochs 0/25/50/75, batch 32, clip 1.0, AMP). 96/5 is the existing
`runs/varnet_mse5_h96_s0`; 32/3 and 64/5 are new (`runs/varnet_reselect_h32k3_s0`,
`runs/varnet_reselect_h64k5_s0`). Each candidate's epoch is chosen among the 20-iteration
snapshots (epoch >= 80), as for the existing runs. The lowest validation RMSE wins. If
96/5 does not win, the 4DVarNet final model and the backbone of the uncertainty model
are re-planned before anything else is trained.

## 4DVarNet with an augmented variance

Candidates (existing, no training): `runs/varnet_aug0_h96_s0` (log-variance in the prior
term only) and `runs/varnet_aughead_obs_h96_s0` (the current model). vsb0 is no longer
in the code base and is not a candidate. Each at its validation-selected epoch. The
winner is the lower validation CRPS (pooled over the four channels, blind walkable cells,
physical units), scored on the same validation observations for both.

## DINCAE

Candidates: full-field loss (existing `runs/dincae_ff`) and the reference loss on
withheld observed values (new `runs/dincae_reselect_obsloss`, `--no-full-field-loss`,
otherwise the recipe of `dincae_ff`). The lower validation RMSE wins.

## EnKF (no training)

On the seven validation days, with the current forecast model and residual bank at its
measured size:

| candidate | process noise |
|---|---|
| E1 (current) | residual bank, AR(1) 0.5, x1.5, factors (1, 1.25, 1.4, 0.93) on blind cells only |
| E2 | residual bank, AR(1) 0.5, x1.5, no extra factors |
| E3 | residual bank, AR(1) 0.5, x1.5, factors (1, 1.25, 1.4, 0.93) on every cell |
| E4 | independent Gaussian noise with the original filter's per-channel standard deviations (`--noise-kind gaussian`) instead of the residual bank; otherwise as E2 |

The winner is the lowest validation CRPS among the candidates whose validation RMSE is
at most 5% above that of E2.

## Early stopping (4DVarNet, DINCAE)

Checkpoints are written every 10 epochs. From epoch 80 (4DVarNet) or epoch 30 (DINCAE)
on, every new checkpoint is scored on the validation split as in rule 3; training stops
once the best checkpoint is 20 (4DVarNet) or 30 (DINCAE) epochs old. Applied to the seven
existing runs of these methods, this rule selects the same epoch as their full 150-epoch
selection in every case.

## Outcome (validation split only; recorded before the test days were touched again)

| method | candidates (validation blind walkable RMSE / CRPS) | winner | change |
|---|---|---|---|
| Senseiver-G, latent | A40 0.2130; G-direct k=1 0.2099 | G-direct | none |
| Senseiver-G, window | k=1 0.2099; k=4 0.1974; k=8 0.1920; k=16 0.1886 (no channel worse than k=1) | k = 16 | none |
| 4DVarNet | 32/3 0.2488; 64/5 0.2389; 96/5 0.2268 (both new runs early-stopped at epoch 100) | 96/5 | none |
| 4DVarNet aug. var. | aug0 CRPS 0.1174; aughead CRPS 0.0999 | aughead | none |
| DINCAE | observed-value loss 0.2948 (stopping rule: epoch 50); full-field 0.2165 | full-field | none |
| EnKF | E1 0.2542 / 0.1125; E2 0.2451 / 0.1042; E3 0.2522 / 0.1117; E4 0.3017 / 0.1505 | E2 | blind-cell factors dropped |

Records: `outputs/reselect/{aug_validation,enkf_validation,senseiver_window_channels}.json`,
each candidate run's `select_valid.json` / `early_stop.json`.
