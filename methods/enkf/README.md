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
| `enkf_lab/` | **pristine, read-only** byte-for-byte copy of the EnKF baseline from `/scratch/work/zhangx29/Partial_observation`. Files are chmod 444 deliberately. This is the reference. |
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
