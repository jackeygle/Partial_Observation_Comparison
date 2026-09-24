# Partial Observation Comparison — project overview

> **To run the final evaluation, or to see how the data is split, read
> [README.md](README.md) first.** This file is the background: what each method
> is, where its final model came from, how to retrain it, and which older
> results in the repository are superseded.

**The problem.** Three robots move through a real shopping-centre corridor (the
ATC pedestrian dataset, Osaka). Each senses only the cells near itself — the
"partial observation". The task is to reconstruct the **full** crowd state over
the whole corridor — density, velocity $(v_x, v_y)$ and in-cell velocity
variance on a 36 × 12 grid of 1 m cells, one frame per second — from those
sparse sightings, and, where a method can, to say how uncertain the
reconstruction is.

## Contents

| Section | For |
|---|---|
| [The final comparison](#the-final-comparison) | which six models are compared, and why these |
| [Scoring conventions](#scoring-conventions) | what exactly every number measures |
| [Where each final model comes from](#where-each-final-model-comes-from) | provenance, retraining commands |
| [Superseded results, and the archive tag](#superseded-results-and-the-archive-tag) | what was removed, and how to get it back |
| [Running things](#running-things) | environment, layout, diagnostics |
| [Data](#data) | where the data lives, what is and is not in git |

---

## The final comparison

The supervisor stopped new experiments on 2026-09-21 and fixed the final model
list. The comparison is evaluated end to end by
`supervisor_evaluation/evaluate.py`; its results and the method for reading them
are in [README.md](README.md). Numbers are kept there only, so two copies cannot
drift apart.

| Row in the results | Directory | What it is | Uncertainty output |
|---|---|---|---|
| Senseiver-A | `methods/senseiver/` | faithful reproduction of Senseiver (Santos et al. 2023) | none |
| **Senseiver-G (ours)** | `methods/senseiver/` | our extension: one latent token per grid cell ("G-direct") and a 16-frame causal window of past observations | none |
| DINCAE | `methods/dincae/` | DINCAE 2.0 (Barth et al. 2022), trained with full-field supervision | σ̂ from the network's variance output |
| 4DVarNet | `methods/varnet/` | 4DVarNet (Fablet et al. 2020), plain MSE loss (Eq. 14), 200-frame window | none |
| 4DVarNet (aug. head) | `methods/varnet/` | 4DVarNet with σ inside the prior (`aug0`), a Gaussian-NLL observation term and a dedicated variance head (**ours**) | σ̂ from the variance head |
| EnKF | `methods/enkf/` | localised ensemble Kalman filter, 100 members, PedPred3 5→5 forecast model, structured process noise sampled from training-day forecast residuals | ensemble spread |

Three rules hold for every row:

- **One model per row.** No seed averaging, no checkpoint averaging, no deep
  ensembles — none of the papers reports one, and a five-seed ensemble next to
  single models would give one method five times the training.
- **Every choice is made on the validation split**: the checkpoint epoch of
  every learned model, which 4DVarNet seed represents it, and the EnKF's noise
  configuration. The test days are touched only by the final evaluation.
- **Identical observations.** Every method sees the same simulated robot
  observations of a day (routes seeded by the date, see
  [README.md](README.md#observation-protocol)).

Not in the final comparison, by decision: the two 4DVarNet deep-ensemble
uncertainty designs (`aug0`, `vsb0`), the learned-covariance Kalman filter
(LCSKF), and the Senseiver temporal-mixer variants. Their code, runs and results
were removed from the current tree and are kept in the git tag
`archive-full-2026-09-24`; see [Superseded results](#superseded-results-and-the-archive-tag).

---

## Scoring conventions

**Walkable cells only.** 290 of the 432 grid cells are walkable; the other 142
(32.9%) are walls and the stalls lining the corridor, derived from the real ATC
map (`crowdcore/assets/atc_map/`, a 1 m cell is a wall iff ≥ 50% of it is
obstacle). No method is asked to reconstruct an obstacle cell and none is scored
there.

**Unobserved ("blind") walkable cells are the headline.** At each frame, about
44% of walkable cells are not seen by any robot; reconstructing them is the task
proper. The accuracy table also reports all walkable cells (observed ones
included). Empty walkable cells are scored in both: getting "nobody is here"
right is part of the task.

**Pooled RMSE.** Squared errors are pooled over all scored cells and all four
channels before the square root; per-channel RMSE is reported next to it. Every
method's point prediction is held to the same physical bounds (density [0, 5],
velocity [−5, 5], variance [0, 2]) — clipped by the accuracy evaluators, and
inside the filter for the EnKF.

**Uncertainty is scored in physical units, on one protocol for all three
probabilistic methods**: same frames (the frames every method can predict —
DINCAE needs t±1, 4DVarNet complete 200-frame windows, and the EnKF discards a
500-frame warm-up), same cells, no clipping. DINCAE's velocity-variance channel
is log1p-transformed inside the network, so its physical predictive distribution
is a shifted log-normal and is scored with the closed-form log-normal CRPS
(`compare/score_uncertainty.py`); everything else is Gaussian. The null model
for CRPS skill is each method's own N(point, RMSE²) on the same cells.

**Every frame of every test day** is scored; nothing is subsampled. (The first
400 frames of a day are an almost empty field — about 1/7 of the day's mean
density — so a subset starting at frame 0 flatters every method.)

**4DVarNet carries a ~4e-4 reproducibility floor**: its inference pass
backpropagates through autograd, and conv backward accumulates with atomicAdd,
so re-running an identical configuration differs in the fourth decimal.
Senseiver and DINCAE are bit-reproducible.

---

## Where each final model comes from

The nine files in `supervisor_evaluation/models/` are the only weights the
evaluation loads; `models/manifest.json` maps each to its training-run source
and SHA-256, and `evaluate.py verify` checks them before every run
(`evaluate.py prepare` rebuilds the package from the sources below).

| Packaged file | Source | Chosen how |
|---|---|---|
| `senseiver_A.pt` | `methods/senseiver/runs/senseiver_A/best.pt` | best validation epoch |
| `senseiver_G_k16.pt` | `methods/senseiver/runs/capacity/base32_k16_s123/best.pt` | best validation epoch (56 of 60) |
| `dincae_epoch60.pt` + `dincae_state_stats.npz` | `methods/dincae/runs/dincae_ff/ckpt_00060.pt`, `methods/dincae/artifacts/state_stats.npz` | `methods/dincae/checks/select_checkpoint.py` on validation |
| `varnet_mse_s3_epoch80.pt` | `methods/varnet/runs/varnet_mse5_h96_s3/ckpt_00080.pt` | epoch per run by `checks/select_checkpoint.py`; seed 3 has the lowest validation error of the five |
| `varnet_aughead_obs_s0.pt` | `methods/varnet/runs/varnet_aughead_obs_h96_s0/ckpt_00090.pt` | `select_valid.json` in that run (epoch 90) |
| `pedpred3_5to5_epoch38.pt` | `methods/enkf/runs/pedpred3_5to5_clip_s0/dyn150_best.pt` | best epoch on the full validation split (`dyn150_pick.json`) |
| `enkf_residual_q_bank.npz` | `methods/enkf/enkf_opt/experiments/outputs/pedpred3_5to5_train_residual_q_8192.npz` | 8192 one-step forecast residual fields from the 32 training days |

The EnKF's scalar settings (process-noise scale 1.5, AR(1) persistence 0.5,
per-channel and blind-cell noise scales, cross-channel matrix, localisation
radius 7, inflation 1.02) are frozen in `FINAL_CONFIG` at the top of
`supervisor_evaluation/evaluate.py`; they were selected on the validation days
(`methods/enkf/enkf_opt/experiments/outputs/optimization_report.md`).

### Retraining

All commands from the repository root, after `source sbatch/_env.sh`. The
4DVarNet and EnKF-dynamics jobs self-resubmit until their epoch budget is done.

```bash
# 4DVarNet, plain MSE: five seeds, then pick every run's epoch on validation
for s in 0 1 2 3 4; do
  sbatch --job-name=varnet_mse5_h96_s$s methods/varnet/sbatch/submit_mse5_chain.sbatch $s 0
done
sbatch --export=ALL,RUNS="mse5_h96_s0 mse5_h96_s1 mse5_h96_s2 mse5_h96_s3 mse5_h96_s4" \
  methods/varnet/sbatch/submit_select.sbatch

# 4DVarNet (aug. head): one seed, then its epoch on validation
sbatch --job-name=varnet_aughead_s0 methods/varnet/sbatch/submit_aughead_chain.sbatch 0 0
sbatch --export=ALL,RUNS="aughead_obs_h96_s0" methods/varnet/sbatch/submit_select.sbatch

# DINCAE: normalisation stats (once), training (~16-27 GPU-hours), checkpoint choice
python3 -m methods.dincae.state
sbatch methods/dincae/sbatch/submit_train.sbatch --out runs/dincae_ff
python3 -m methods.dincae.checks.select_checkpoint

# Senseiver-A (up to a day) and Senseiver-G (grid latent, direct read-out, 16-frame window)
sbatch methods/senseiver/sbatch/submit_train.sbatch --out runs/senseiver_A
EPOCHS=60 OUT=runs/capacity/base32_k16_s123 sbatch methods/senseiver/sbatch/submit_train.sbatch \
  --latent-mode grid --readout direct --time-window 16 --seed 123

# EnKF forecast model (PedPred3, 5 frames in -> 5 out). The first run diverged at
# epoch 30; the reported one restarts from its epoch-29 weights with a fresh
# optimiser and gradient clipping:
sbatch methods/enkf/lcskf/sbatch/submit_dynamics_5to5.sbatch --epochs 150 --seed 0 \
  --out runs/pedpred3_5to5_s0
sbatch methods/enkf/lcskf/sbatch/submit_dynamics_5to5.sbatch \
  --init-weights runs/pedpred3_5to5_s0/epoch_29.pt --clip-grad 1.0 --lr 5e-4 \
  --epochs 150 --batch 64 --eval-batch 1024 --seed 0 --out runs/pedpred3_5to5_clip_s0
# ... then its residual noise bank (training days only)
python3 -m methods.enkf.enkf_opt.experiments.build_residual_q_bank
```

After retraining, `python3 supervisor_evaluation/evaluate.py prepare` copies the
new files into `supervisor_evaluation/models/` and rewrites the manifest.

Two traps that silently produce a *different number* rather than an error:

1. **Do not average DINCAE's checkpoints.** The reference implementation does;
   `methods/dincae/checks/evaluate.py` refuses unless given
   `--average-checkpoints`, because the result is not the reported configuration.
2. **Do not score a 4DVarNet run at `varnet_best.pt`.** Despite the name it is
   chosen on the *training* split, on the emptiest windows of a day. Reported
   numbers use the epoch in each run's `select_valid.json`, which
   `methods/varnet/checks/model_io.py:reported_ckpt` resolves; a run without it
   prints a `[ckpt] ... falling back` line.

---

## Superseded results, and the archive tag

On 2026-09-24 the repository was reduced to the six final models: their code,
training and selection scripts, the final evaluation, and the records needed to
explain them. Everything else — all other experiments, their runs, logs, figures and
result files, the earlier cross-method scorer outputs, the original EnKF code copy and
its diagnostics — is preserved unchanged in the git tag **`archive-full-2026-09-24`**:

```bash
git checkout archive-full-2026-09-24          # the whole earlier tree
git checkout archive-full-2026-09-24 -- <path>   # one file or directory into the current tree
```

What those older results were, so that numbers quoted elsewhere are not mistaken for
the final ones:

- **`compare/results/compare5_final.json`** — the accuracy table used until
  2026-09-17, when robot routes changed from one fixed seed for every day to a seed
  per day. Different observations, so different numbers.
- **4DVarNet `aug0` / `vsb0` five-seed deep ensembles** — replaced by the single
  `aughead_obs` model.
- **Uncertainty scored in DINCAE's normalised space** — coverage is unaffected by
  that space, but CRPS skill and spread/RMSE are not (log1p compresses the large
  errors, and a constant σ there is a different null model). The final scores are all
  in physical units.
- **The EnKF "ensemble collapse"** (spread ≈ 1% of the error, coverage 1.6% of a
  90% interval) in the original filter. Its main cause is configuration: the vendored
  forecast step injects only `0.01 ×` the process noise it was designed for. At full
  strength the same filter reaches spread/RMSE 0.96 and 93% coverage
  (`methods/enkf/checks/diag_proc_scale_sweep.py`, still in the tree, one day, 2000
  frames), so an earlier conclusion that no amount of noise can fix it was wrong.
  Full-strength independent noise still gives a spatially uninformative σ̂; the final
  EnKF therefore samples the noise from real forecast residuals
  (`methods/enkf/README.md`).
- **The learned-covariance Kalman filter (LCSKF)** — a research line that replaces
  the EnKF's ensemble covariance with a learned one; not in the final comparison.
- **Senseiver extensions after G-direct k=16** (temporal mixers, history loss, motion
  tokens, geodesic bias) — none beat k=16.
- Anything quoting MSE over all 432 cells, `obs_every_k=4`, or 400-frame subsets —
  scopes the project stopped reporting in early September.

---

## Running things

### Environment

```bash
cd path/to/this/repo
source sbatch/_env.sh          # module load + PYTHONPATH + PYTHONSAFEPATH=1
```

`_env.sh` loads Aalto Triton's `scicomp-pytorch-env/2026.1` (Python 3.12); on
another machine, `pip install -r requirements.txt`. Invoke code as
`python3 -m <package.module>` and submit `sbatch` from the repository root.

**Never use a bare script path.** `python3 methods/varnet/train.py` puts
`methods/varnet/` on `sys.path[0]`, and the methods each ship their own,
different `losses.py`, `dataset.py` and `model.py` — whichever is found first
wins, silently. `PYTHONSAFEPATH=1` in `_env.sh` blocks that.

The login node has no GPU. Anything that runs `torch` on real data goes through
`sbatch`, or an interactive node: `srun -p gpu-debug --gres=gpu:1 -t 00:15:00 --pty bash`.

### Repository layout

```
README.md                 how to run the final evaluation; the data split
supervisor_evaluation/    the final evaluation: evaluate.py, packaged weights,
                          job scripts, split lists, results (outputs/full/)
crowdcore/                shared by every method, method-agnostic
  config.yaml               single source of truth for every parameter
  observation_model.py      splits, robot simulation, observations
  navigation.py             real ATC map -> walkable cells, line of sight
  assets/atc_map/           the map (ROS occupancy grid)
  data/                     raw CSV -> gridded H5 pipeline (DOC_data_pipeline.md)
methods/                  one directory per method; none imports another
  varnet/                   4DVarNet (MSE and aug. head)
  dincae/                   DINCAE
  senseiver/                Senseiver (A, and the G / temporal extensions)
  enkf/                     EnKF: enkf_opt/ (filter code + the final structured-noise
                            GPU filter and its tuning record), lcskf/dynamics/
                            (PedPred3 forecast-model training)
  each has train.py (not enkf/), checks/ (checkpoint selection, evaluation),
  sbatch/ (training jobs) and runs/ (training logs; checkpoints gitignored)
compare/                  the only code that imports more than one method:
                          cross-method scoring, shared metrics, plot style
sbatch/_env.sh            the single environment entry point
```

### Per-method scripts that are not training

| Question | Script |
|---|---|
| DINCAE accuracy, σ̂ calibration, variance retention on its own | `methods.dincae.checks.evaluate` |
| Which epoch of a run to report (validation split) | `methods.varnet.checks.select_checkpoint`, `methods.dincae.checks.select_checkpoint` |
| Why did the original EnKF collapse? | `methods.enkf.checks.diag_proc_scale_sweep` |
| A reconstructed field as a picture, all methods | `supervisor_evaluation/evaluate.py images` (100 matched frames) |

All take `--help`. The earlier diagnostics (solver checks, pipeline traces, EnKF
bit-identity checks and benchmarks) are in the archive tag.

---

## Data

Three pipeline stages; **only the last is read at run time**.

| Stage | Location (Triton) | Size | Read at run time? |
|---|---|---|---|
| (1) raw ATC CSVs, 92 recording days | `/scratch/work/zhangx29/ATC/` | 225 GB | no |
| (2) trajectory H5, the 46 Sundays | `/scratch/work/zhangx29/data/ATC/Sundays/` | 36 GB | no — only its file names |
| (3) `grid_cache`, 4-channel 36 × 12 fields, 46 days | `/scratch/work/zhangx29/data/grid_cache/` | 3.2 GB | **yes** |

The split lists name stage-(2) files, but `observation_model.split_files()` uses
only the date and opens the stage-(3) file. `data.root` in
`crowdcore/config.yaml` (or `--data-root` for the evaluation) is the one path to
change on another machine. Rebuild the stages only if the raw data or the grid
resolution changes; see `crowdcore/data/DOC_data_pipeline.md`.

**In git:** code, configuration, the real ATC map, the split lists (copies in
`supervisor_evaluation/data_split/`), the nine packaged final weights (126 MB,
`supervisor_evaluation/models/`), the original EnKF forecast model (`methods/enkf/enkf_opt/apt-ibex_train_model_28D.pth`, 14 MB), DINCAE's
normalisation statistics, result JSON/CSV and figures.

**Not in git:** the ATC data at every stage (it carries its own redistribution
terms), the intermediate training checkpoints under `methods/*/runs/` (~11 GB),
DINCAE's encoding cache (~13 GB), and exported EnKF fields (`*.npz`, several GB).
To run the evaluation on another machine, the repository plus the seven test
days of `grid_cache` (483 MB) and the split lists are enough.
