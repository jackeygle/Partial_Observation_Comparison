# Superseded arms and overturned conclusions (4DVarNet)

Facts that are easy to re-derive wrongly, or to waste time re-discovering.
Rescued from `slides/STALE.md` when the decks it annotated were deleted
(all of `slides/` is gone as of 2026-09-09, archived at tag
`archive-2026-09-09`); the decks are gone, these findings are not.

## `ml5` is NOT a learned-sigma arm any more

`runs/varnet_ml5_s{0..4}` is a 5-member ensemble whose uncertainty is
**member disagreement only** — Lakshminarayanan et al.'s "Ensemble-M (MSE)".

It used to have a learned variance head (a single pointwise read-out, 548
parameters). That head was **removed on purpose**: it keyed off
distance-to-observation, a feature that predicts the true error with an R^2 of
roughly 0.01-0.03, while ignoring |x_hat|, which predicts it with 0.21-0.45
(measured by `checks/diag_sigma_drivers.py`, archived at tag `archive-2026-09-09`). Today's checkpoints have 23 keys and no
variance parameters — `var_cost.*` are variational-cost weights, unrelated.

The surviving variance head is a different design, `grad_net.out_var`
(800x864x1x1, ~692k parameters), and only the `vsb0` family has it.

Re-measured 2026-09-05: blind RMSE 0.1975, CRPS 0.0712, spread/skill 0.435,
90% coverage 80.8%. Older numbers quoting spread/skill 0.87 and 93.8% coverage
came from an abandoned `sigma^2 = obs_var + softplus(head)` convention, where
obs_var was a hand-set observation-noise constant (four-channel mean 0.1161).
**ml5 is not in the main tables.**

## The EnKF's ensemble collapse

The surrogate forecast removes ~65% of member disagreement per step while the
filter injects 0.0022 per step, so the spread settles at ~0.0025 — matching
what is measured. Confirmed independently in
`check_outputs/eval/enkf_spread_growth.json` (verdict: `contractive`).

Corroborating oddity: the EnKF is the only method whose **observed** cells
score worse than its **blind** cells (0.3788 vs 0.3677); every other method is
clearly better where it was given data. The gain is so small the filter barely
assimilates what it sees.

## `FAILED_ml5_beta0` — a valid negative result

With beta=0, sigma^2 collapses to the per-channel floor and the loss degenerates
into a weighted MSE with weights [154, 5, 67, 11905]. Kept in
`runs/FAILED_ml5_beta0.json`.

## Convention change

Anything predating 2026-09-05 reports **MSE**, usually under a single cell
scope. The main tables now use **RMSE**, split three ways (full field /
observed / blind) x two conventions (defined-and-walkable / all cells). Old
numbers are not directly comparable — see `README.md`'s scoring-convention
section.
