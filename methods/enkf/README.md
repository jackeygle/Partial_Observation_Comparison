# methods/enkf — localised ensemble Kalman filter on the ATC crowd field

> Part of [Partial Observation Comparison](../../README.md) — see the root README for the problem statement, the scoring-convention pitfall that decides the ranking, and which checkpoint backs which published number.

The only method here that is **not learned**. It is a filter: a 100-member
ensemble propagated by the PedPred3 neural surrogate, corrected by a localised
Kalman analysis on every frame that carries an observation. There is no
`train.py` and no checkpoint — its "model" is the vendored baseline code plus a
configuration.

## Two copies of the same code, on purpose

| Directory | Role |
|---|---|
| `enkf_lab/` | **pristine, read-only** byte-for-byte copy of the EnKF baseline from the original `Partial_observation` project, trained surrogate weights included. Files are chmod 444 deliberately. This is the reference. |
| `enkf_opt/` | the copy we are allowed to modify. Every change must be **bit-identical** to `enkf_lab` on real data — `np.array_equal`, not `np.isclose`. |

Verify before trusting any change to `enkf_opt/`:

```bash
python3 -m methods.enkf.checks.verify_enkf_opt --frames 12
# [verdict] PASS -- strict path bit-identical: True, unit checks: True
```

The optimised copy is ~24x faster than the reference at identical output (and
~114x with the fast path), which is why it exists at all. Each directory has its
own README with the details.

## Producing the EnKF's results

Two stages, both CPU-only — no GPU accelerates a Kalman analysis. This is
**slow**: one real day took ~53 CPU-hours in the run that produced
`check_outputs/enkf_k1_full/`, so submit one job per day and let the seven run in
parallel. Commands are in the root `README.md` under "Training".

Outputs land in `check_outputs/enkf_k1_full/` as `obs_<day>.npz` (the simulated
robot observations) and `est_<day>.npz` (the filter's estimate and ensemble
spread). Those `.npz` are gitignored; the `.out` logs and timing JSON beside them
are tracked.

## What the checks answer

| Question | Script |
|---|---|
| Is `enkf_opt/` still bit-identical to `enkf_lab/`? | `checks/verify_enkf_opt.py` |
| Same, for the gain-mode variants | `checks/verify_enkf_gain_mode.py` |
| Score the exported estimate against ground truth | `checks/score_enkf.py` |
| Is the ensemble spread a useful uncertainty? (CRPS/NLL/coverage vs a constant-sigma null model) | `checks/eval_uncertainty_enkf.py` |
| **Why is the ensemble so overconfident?** | `checks/diag_enkf_spread_growth.py` |
| Where does the wall time go (forecast vs analysis)? | `checks/bench_enkf_split.py`, `checks/bench_enkf_opt.py` |
| Reconstructed velocity field as a figure | `checks/plot_velocity_enkf.py` |

## Experimental differentiable low-rank UNetKF

`lowrank/` is the isolated next experiment.  It keeps the best retrained PedPred3 mean
forecast frozen and trains a separate covariance U-Net with two heads:

```text
x_t -> frozen PedPred3 -> mean_{t+1}
[x_t, mean_{t+1}, mean_{t+1}-x_t, static_mask] -> covariance U-Net -> U, d
B = U U^T + diag(d) -> differentiable Kalman update -> analysis_{t+1}
```

The Kalman layer uses Woodbury algebra and solves only a `rank x rank` system; it never
constructs the full 1728-square covariance.  The initial experiment freezes
`runs/surrogate_mean_s0/best.pt` and uses rank 16. Training samples legal robot positions
and unions their exact radius-plus-line-of-sight footprints from the real map; final
evaluation uses the exported moving-robot trajectories and observations.

```bash
# Algebra, gradient, and shape checks (run in the project PyTorch environment)
python3 -m methods.enkf.checks.check_lowrank_kalman

# Small smoke train, then the full rank-16 run
python3 -m methods.enkf.lowrank.train --allow-cpu --days 1 --max-frames 128 --epochs 1 --batch 8
sbatch methods/enkf/sbatch/submit_lowrank_smoke.sbatch
sbatch methods/enkf/sbatch/submit_lowrank_unetkf.sbatch train --rank 16 --seed 0

# Sequential validation and the existing uncertainty scorer
sbatch methods/enkf/sbatch/submit_lowrank_unetkf.sbatch eval \
  --checkpoint methods/enkf/runs/lowrank_r16_s0/best.pt --only atc-20130616
python3 -m methods.enkf.checks.eval_uncertainty_enkf \
  --dir-fmt check_outputs/lowrank_unetkf_valid \
  --out-fmt check_outputs/eval/uncertainty_lowrank_unetkf.json --k 1
```

Formal jobs request three hours. Five minutes before the limit, the batch script
automatically submits the same command with `--resume`; the last completed epoch is
restored from `last.pt`, so long runs continue across allocations without manual work.

## Paper-style local covariance experiment

`local_unetkf/` is a separate implementation of Lu's covariance-supervision idea.  Its
teacher is PedPred3's actual one-step forecast error rather than the collapsed legacy
ensemble.  The first gate audits systematic mean bias, then compares raw second-moment
and bias-centered climatological local covariance before any U-Net is trained:

```text
error_raw      = truth[t+1] - PedPred3(x_t)
error_centered = error_raw - train_bias[channel,y,x]
target[p,a,q,b] = error[p,a] * error[q,b]
```

The local window is `15x15` (radius 7, with a unique centre) and retains all 16 directed
channel pairs.  Boundary padding has an explicit validity mask.  Run the exact target
checks and the two formal H200 stages with:

```bash
python3 -m methods.enkf.checks.check_local_covariance_targets
job=$(sbatch --parsable methods/enkf/sbatch/submit_local_unetkf.sbatch audit)
sbatch --dependency=afterok:$job methods/enkf/sbatch/submit_local_unetkf.sbatch static
```

## The ensemble-collapse finding

The EnKF's uncertainty **is** its ensemble spread, and that spread collapses to
about 0.011 of the actual error. `diag_enkf_spread_growth.py` separates the two
possible causes by propagating a perturbed ensemble through PedPred3 with no
injected noise and no analysis step: the spread decays ~65% per step, verdict
`"contractive"`.

So it is not a tuning problem. A deterministic forecast model can sustain an
ensemble only if its dynamics *amplify* differences (that is how operational
weather ensembles work); this one damps them, so no amount of extra process
noise or inflation fixes it. Corroborating oddity: the EnKF is the only method
whose **observed** cells score worse than its **blind** cells — the gain is so
small the filter barely assimilates what it sees.

This is reported as a property of the forward model, not as a bug to fix.
