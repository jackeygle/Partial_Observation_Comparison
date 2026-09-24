# Partial Observation Comparison — evaluation

Three robots move through a shopping-centre corridor (the ATC pedestrian
dataset, Osaka) and each senses only the cells near itself. Six methods
reconstruct the **full** crowd state — density, velocity $(v_x, v_y)$ and
velocity variance on a 36 × 12 grid, one frame per second — from those partial
observations; three of them also predict their own uncertainty.

This README covers **how the final comparison is evaluated and how the data is
split**. Background on the methods, training, and the experiment history is in
[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md).

## Quick start

On Aalto Triton, from the repository root:

```bash
sbatch supervisor_evaluation/sbatch/smoke.sbatch   # ~2 min on gpu-debug: end-to-end check, not results
sbatch supervisor_evaluation/sbatch/full.sbatch    # complete evaluation -> supervisor_evaluation/outputs/full/
```

Everything goes through one script, `supervisor_evaluation/evaluate.py`
(`bench_inference.py` next to it is called by it, never run on its own). It
imports code from the rest of the repository (`crowdcore/`, `compare/`,
`methods/`), so it must be run inside this repository, from its root.

- **Environment:** `source sbatch/_env.sh` loads Triton's
  `scicomp-pytorch-env/2026.1` (Python 3.12) and sets `PYTHONPATH`; elsewhere,
  `pip install -r requirements.txt`. A GPU is required.
- **Weights:** the nine files in `supervisor_evaluation/models/` are the frozen
  final models and are part of the repository. `models/manifest.json` records
  their SHA-256; every run checks them first and refuses to start on a mismatch.
- **Data:** not in the repository. Point `--data-root` (or `data.root` in
  `crowdcore/config.yaml`) at a directory laid out as described
  [below](#data-layout). The two job scripts use `/scratch/work/zhangx29/data`;
  change that path to run on another copy.

What `full` does, in order: verify packages, checkpoint hashes, data schema and
test dates → score the six methods on the seven test days → run the EnKF →
score uncertainty for the three probabilistic methods on one shared protocol →
benchmark inference time on one GPU → write CSVs and figures. The command
reference (single steps, reruns, image boards) is in
[supervisor_evaluation/README.md](supervisor_evaluation/README.md).

## Dataset split

The ATC dataset records the corridor on Sundays and Wednesdays for about a
year. **Only the 46 available Sundays are used** (one weekday pattern, so crowd
behaviour is comparable between days). The split is **chronological, never
shuffled** — training is the earliest stretch, validation the middle, test the
latest — so no afternoon is seen from both sides of a boundary.

| Split | Days | Frames | Date range | Used for |
|---|---|---|---|---|
| train | 32 | 1,262,518 | 2012-10-28 → 2013-06-09 | fitting weights and all training-derived statistics |
| valid | 7 | 269,743 | 2013-06-16 → 2013-07-28 | choosing checkpoints and calibration constants |
| test | 7 | 277,543 | 2013-08-11 → 2013-09-29 | every reported number |

Test days: `20130811 20130818 20130825 20130901 20130915 20130922 20130929`.
There is no recording for the Sundays 2013-08-04 and 2013-09-08 in the source
data, which is why they are missing. Every frame of every test day is scored
(about 38,000–43,000 frames per day, ~11 hours); nothing is subsampled.

**Where the split is defined.** The code reads three list files from the data
root, `sunday_atc_{train,valid,test}.lst` (one `ATC/Sundays/atc-YYYYMMDD.h5`
per line; `crowdcore/observation_model.split_files`). Copies of the exact lists
used are in [`supervisor_evaluation/data_split/`](supervisor_evaluation/data_split/).
`evaluate.py` also pins the test dates in `TEST_DATES`; the `verify` step that
starts every run fails unless the test list and `TEST_DATES` name exactly the
same days.

**Using a different split.**

- *Different test days only:* put the new days in `sunday_atc_test.lst`, update
  `TEST_DATES` in `evaluate.py` to match, and make sure their gridded files
  exist (see below). The new test days must not appear in the train or valid
  lists. No retraining is needed.
- *Different train or valid days:* the packaged weights are no longer valid.
  Besides the model weights, these are fitted on the training days and must be
  rebuilt with the new split: the walkable-cell mask (a cell is walkable if its
  density exceeds 0.5 on some training day; cached as
  `grid_cache/visited_union_train_tau0.5_corridor.npy` — delete it and it is
  recomputed from the training list), DINCAE's normalisation statistics
  (`methods/dincae/artifacts/state_stats.npz`), and the EnKF's residual noise
  bank. Retraining is described in [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md).

## Data layout

```
<data-root>/
  sunday_atc_train.lst  sunday_atc_valid.lst  sunday_atc_test.lst
  grid_cache/atc-YYYYMMDD_corridor_1.0s.h5        # one file per day
  grid_cache/visited_union_train_tau0.5_corridor.npy   # training-day walkable evidence
```

Each gridded day file holds `grid` — float32 `(N, 4, 36, 12)`, channels
`[density, vx, vy, vel_var]`, 1 m cells, one frame per second — and `time`
`(N,)`. They are produced from the raw ATC CSV files by
`crowdcore/data/csv_to_h5.py` and `crowdcore/data/h5_to_grid.py`; the pipeline
and its validation are documented in
[crowdcore/data/DOC_data_pipeline.md](crowdcore/data/DOC_data_pipeline.md).

## Observation protocol

Set in `crowdcore/config.yaml` (`observation:`), identical for every method:
three robots follow A* routes between random walkable goals, one cell per
second; a walkable cell is observed at a frame if it is within 7 cells of a
robot and in line of sight (obstacles from the real ATC map block sight).
On average about 56% of walkable cells are observed at a frame; the other 44%
are the blind cells the headline scores are computed on. Observations carry
per-channel Gaussian noise. Robot routes are seeded by the day's date, so every
day has its own routes and every script sees the same observations for a day.

## What is scored

All scores are on **unobserved walkable cells**: cells inside the corridor that
no robot sees at that frame. Obstacle cells are never scored.

- **Accuracy:** RMSE of the reconstruction, pooled over the four channels and
  per channel; six methods.
- **Uncertainty** (4DVarNet with augmented head, DINCAE, EnKF), all in
  physical units on the same frames and cells:
  - *CRPS skill* — improvement of the continuous ranked probability score over
    a null model that predicts the same constant σ (the method's own RMSE) in
    every cell. It is the main uncertainty metric: it rewards σ̂ for being
    large where the error is large, which an average-scale check cannot see.
    Below 0, the predicted uncertainty is worse than a constant.
  - *Spread / RMSE* — mean predicted σ̂ over RMSE; 1 means the right average
    size, below 1 too small.
  - Pooled numbers carry 95% bootstrap intervals over the seven test days.
- **Inference time:** median GPU time per reconstructed frame, same GPU and
  frames for every method.

## Results (seven test days)

| Method | RMSE (all) | Density | $v_x$ | $v_y$ | Vel. var. |
|---|---|---|---|---|---|
| Senseiver-A | 0.222 | 0.156 | 0.372 | 0.154 | 0.107 |
| **Senseiver-G (ours)** | **0.197** | **0.133** | **0.326** | **0.145** | **0.103** |
| DINCAE | 0.227 | 0.153 | 0.381 | 0.160 | 0.108 |
| 4DVarNet | 0.231 | 0.176 | 0.361 | 0.178 | 0.140 |
| 4DVarNet (aug. head) | 0.295 | 0.195 | 0.504 | 0.189 | 0.142 |
| EnKF | 0.283 | 0.154 | 0.419 | 0.183 | 0.295 |

| Method | CRPS skill (95% CI) | Spread / RMSE |
|---|---|---|
| 4DVarNet (aug. head) | 0.232 (0.221–0.244) | 0.66 |
| **DINCAE** | **0.275 (0.266–0.283)** | 0.58 |
| EnKF | 0.075 (0.068–0.081) | 1.21 |

![Reconstruction accuracy](supervisor_evaluation/outputs/full/figures/accuracy_rmse.png)
![Uncertainty](supervisor_evaluation/outputs/full/figures/uncertainty_summary.png)
![Inference time](supervisor_evaluation/outputs/full/figures/inference_latency.png)

The CSVs behind these (`accuracy.csv`, `uncertainty.csv`,
`uncertainty_by_channel.csv`, `inference_time_controlled.csv`), a LaTeX table
(`accuracy_table.tex`) and generated figure captions (`figures/captions.md`)
are in `supervisor_evaluation/outputs/full/`.
