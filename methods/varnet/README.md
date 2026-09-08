# Crowd-field reconstruction on ATC — 4DVarNet reproduction

Reproducing the **4DVarNet** learned variational data-assimilation framework
(Fablet et al. 2020, [arXiv:2007.12941](https://arxiv.org/abs/2007.12941)) on the
**ATC pedestrian dataset**: reconstruct the full crowd field (density + velocity)
over a time window from a few robots' partial, noisy observations, and compare
fairly against a **Localized EnKF** baseline.

This README is written so someone else can reproduce the whole thing from the raw
data. **All parameters live in `config.yaml` (single source of truth)**; every
command reads its defaults from there, and command-line flags override for one run.

---

## TL;DR — reproduce end to end

```bash
# 0. environment (Aalto Triton)
module load scicomp-pytorch-env/2026.1
cd /scratch/work/zhangx29/Thesis_Project/methods/varnet

# 1. (data is already gridded — see "Data" below if you must rebuild from CSV)

# 2. train 4DVarNet to convergence (full 32-day train split, paper window dT=200)
sbatch --job-name=varnet_b0_k1 sbatch/submit_b0_chain.sbatch 1   # self-chains on an H200
#   -> runs/varnet_b0_k1/varnet_last.pt + metrics.jsonl

# 3. evaluate on the 7 held-out TEST days
sbatch sbatch/submit_eval.sbatch checks/eval_test_days.py --tag _matched_clip   # 4DVarNet
python3 -m methods.enkf.checks.export_obs_for_enkf --frames 400        # export identical obs for the EnKF
python3 -m methods.enkf.checks.run_enkf_baseline  --frames 400         # EnKF (apt-ibex) on the SAME obs
python3 -m methods.enkf.checks.score_enkf                              # score EnKF vs truth

# 4. figures + comparison + slide deck
python3 -m compare.plot_reconstruction_enkf --day atc-20130811
python3 -m methods.enkf.checks.plot_velocity_enkf       --day atc-20130811
python3 -m compare.compare_channels
python3 -m compare.plot_comparison
python3 -m slides.build_meeting4_deck      # -> slides/meeting4_deck.{pptx,pdf} + _notes.md
```

> **GPU, not the login node.** Anything that touches `torch` (training, evaluation,
> reconstruction/velocity figures, the deck's table) must run on a GPU node —
> `sbatch`, or `srun --partition=gpu-debug --gres=gpu:1 --time=00:15:00 bash -c '...'`.
> The pure-map / pure-numpy figures are fine on a compute node too.

---

## Repository layout

| path | role |
|---|---|
| `config.yaml` | **all parameters** (data, grid, navigation/map, observation, prior, solver, training) |
| `crowdcore/data/csv_to_h5.py` | Stage 1: raw ATC CSV → trajectory H5 |
| `crowdcore/data/h5_to_grid.py` | Stage 2: trajectory H5 → grid_cache `(T,4,36,12)` |
| `observation_model.py` | Component 1: multi-robot partial observation; `split_files`, `load_state`, `generate_observations` |
| `navigation.py` | walkable map / obstacles / A*; `build_valid_mask_from_config` |
| `prior_model.py` | Component 2: GENN dynamical prior Φ |
| `variational_solver.py` | Component 3: variational cost + learned-gradient-descent solver (`GradSolver`) |
| `train.py` | end-to-end training (Φ + solver + cost weights, one loss) |
| `checks/` | verification scripts + all figure/evaluation scripts (see below) |
| `sbatch/` | SLURM submit scripts |
| `slides/build_meeting4_deck.py` | builds the current meeting deck (reads numbers from config/checkpoint/JSON) |
| `runs/varnet_b0_k1/` | the model of record: checkpoint + `metrics.jsonl` (see the repo README's method table) |
| `check_outputs/` | all generated figures & metric JSONs (organised by module; `eval/` = comparison) |

---

## Data

### Flow at a glance
```
raw ATC CSV                                    (per day, ~1-4 GB)
   │  Stage 1  crowdcore/data/csv_to_h5.py       mm->m, group by timestamp
   ▼
trajectory H5   position/velocity/index         (~650 MB/day)
   │  Stage 2  crowdcore/data/h5_to_grid.py       1 s frames, triangular kernel, rotated corridor
   ▼
grid_cache      (T, 4, 36, 12)  [density, vx, vy, var], 1 s per frame   <- day-to-day work starts HERE
   │  Component 1  observation_model.generate_observations
   ▼
X, Y, Ω, X0     full state / partial obs / mask / init      (in memory)
   │  Components 2-3  prior_model.py + variational_solver.py
   ▼
train.py          ->  reconstructed field  (Φ + solver trained jointly)
```

The `grid_cache/atc-*_corridor_1.0s.h5` files are already the Stage-2 output;
`observation_model.load_state` reads them directly. Rebuild from CSV only if needed:

```bash
python3 crowdcore/data/csv_to_h5.py  --csv <atc-YYYYMMDD.csv> --out /tmp/day.h5
python3 crowdcore/data/h5_to_grid.py --traj-h5 /tmp/day.h5 --out /tmp/day_corridor_1.0s.h5 --subset corridor
#   add --validate <ref.h5> to either stage to check a rebuild against an existing file
```
- **State** = 4 channels `[density, vx, vy, var]` on a 36×12 corridor grid, 1 s/frame.
  `vx` = velocity along grid rows (down the corridor), `vy` = along columns (across).

### Splits — temporal, disjoint, no leakage
Defined by `data/sunday_atc_{train,valid,test}.lst` (ATC Sundays):

| split | days | date range | used for |
|---|---|---|---|
| **train** | 32 | 2012-10-28 → 2013-06-09 | training 4DVarNet |
| valid | 7 | 2013-06-16 → 2013-07-28 | validation (not in the final comparison) |
| **test** | 7 | 2013-08-11 → 2013-09-29 | evaluation (4DVarNet **and** EnKF, identical days) |

Train is strictly earlier than test → past-trains-future, no overlap. Both methods
are scored on the same held-out test days they never saw in training.

---

## The three components

**Component 1 — partial observation** (`observation_model.generate_observations`).
Turns a full-state sequence `X` into `(Y, Ω, X0)`:
- walkable region = the **real ATC map** at 1 m (state) resolution. Supervisor's rule:
  a 1 m cell is WALL iff **≥ 50 %** of it is obstacle (`obstacle_rule=per_cell`,
  `occupancy_thresh=0.5`). Obstacles follow ROS semantics (OCCUPIED ∪ enclosed-UNKNOWN);
  hybrid with "walked in training" and the largest connected component
  (`navigation.build_valid_mask_from_config`).
- `num_agents=3` robots sample walkable start/goal, follow A* paths, and observe
  `disk ∩ line-of-sight` (range `sensing_range=7`) each second — **not** intersected
  with the walkable mask (a robot can see a pillar it cannot drive into).
- Gaussian sensor noise per channel, `obs_std` (see table). `X0` = fill the unobserved
  cells (`init_method=prev`).

**Component 2 — prior Φ = GENN** (`prior_model.py`). A *regulariser*, not a forecaster:
`Φ(x)(t)` says what frame `t` should look like judged from its neighbours. Key property
is **zero-centre** (`ψ`'s centre conv tap is masked to 0), so `‖x − Φ(x)‖²` cannot
collapse to the identity. Two-scale (paper Eq.10): coarse + fine branch, each
`ψ (3×3×3 conv) → ReLU → φ (two 1×1×1 pointwise convs)`.

**Component 3 — solver** (`variational_solver.GradSolver`). Minimises
`J(x) = α_obs²‖(x−y)⊙Ω‖² + α_reg²‖x−Φ(x)‖²` (α and per-channel weights learnable) by a
**learned gradient descent**: for `n_iter=20` steps, `g=∂J/∂x` (autodiff, no
hand-derived gradient) → `ConvLSTM2d` → `x ← x − u/n_iter`. Φ and the solver train
**jointly, end-to-end** under one loss (paper Eq.14).

---

## Train

```bash
sbatch --job-name=varnet_b0_k1 sbatch/submit_b0_chain.sbatch 1   # self-chaining --resume
#   runs: train.py --days 32 --dT 200 --n-iter 20 --epochs 100 --batch 32 --amp --loss supervised
#   out : runs/varnet_b0_k1/varnet_last.pt (+ optimizer/epoch for resume) + metrics.jsonl

# quick smoke test
srun --partition=gpu-debug --gres=gpu:1 --time=00:15:00 \
     bash -c 'module load scicomp-pytorch-env/2026.1; python3 -u -m methods.varnet.train --days 1 --steps 50'
```
- Loss: `--loss supervised` = `‖x_rec − X‖²` (paper Eq.14, unweighted MSE). `--no-noise`
  observes without sensor noise.
- `metrics.jsonl` logs train loss / blind-zone MSE / R-score / the X0 baseline per epoch.

---

## Evaluate (fair 4DVarNet vs EnKF)

Both methods use the **same test days, observations, noise, frames, metric, and clip
bounds**, and **neither is initialised from truth** (4DVarNet: obs-based `X0`; EnKF:
its original random ensemble — the `Partial_observation` project is left unmodified).

```bash
# 4DVarNet — blind-zone & full-state MSE on the 7 test days (clip on to match the EnKF)
sbatch sbatch/submit_eval.sbatch checks/eval_test_days.py --tag _matched_clip
#   -> check_outputs/eval/test_metrics_matched_clip.json

# EnKF — export identical observations, run, score
python3 -m methods.enkf.checks.export_obs_for_enkf --frames 400   # -> check_outputs/enkf/obs_<day>.npz
python3 -m methods.enkf.checks.run_enkf_baseline  --frames 400    # -> check_outputs/enkf/est_<day>.npz  (needs the apt-ibex model)
python3 -m methods.enkf.checks.score_enkf                         # -> check_outputs/eval/enkf_metrics.json

# per-channel × per-region (observed Ω / blind ¬Ω) breakdown
python3 -m compare.compare_channels                  # -> check_outputs/eval/channel_metrics.json + compare_channels.png
```
The EnKF driver lives **in this project** (`methods/enkf/checks/run_enkf_baseline.py`) and imports
`pedpred.*` from the `Partial_observation` project via `sys.path`; it does not modify it.

---

## Slides

```bash
python3 -m slides.build_meeting4_deck    # login node is fine (reads JSON, no checkpoint)
#   -> slides/meeting4_deck.pptx / .pdf / meeting4_deck_notes.md   (6 slides)
```
The deck **reads every number** from `config.yaml`, the checkpoint's saved `args`, and
the `check_outputs/eval/*.json` files — nothing is hard-coded in the slide script, so it
can never drift from what was actually trained/measured.

---

## config.yaml — key parameters & provenance

| key | value | meaning / provenance |
|---|---|---|
| `navigation.obstacle_rule` / `occupancy_thresh` | `per_cell` / `0.5` | supervisor's rule: 1 m cell is wall iff ≥50 % obstacle |
| `navigation.visited_tau` | 0.5 | "walked in training" data criterion (rejects cross-wall kernel spill) |
| `navigation.keep_largest_component` | true | drop unreachable islands |
| `observation.sensing_range` | 7 | grid cells; >50 % crowd-mass coverage |
| `observation.num_agents` | 3 | robots |
| `observation.line_of_sight` | true | walls block sight (ray-cast on the real map) |
| `observation.obs_std` | `[0.0569, 0.3147, 0.0862, 0.0064]` | per-channel sensor noise = **0.25 × 1.4826×MAD** on active cells (rederive with `checks/rederive_obs_std.py`); the 0.25 is an assumption, not a spec |
| `prior.hidden / kt,kh,kw / n_phi_layers / scale` | 32 / 3,3,3 / 2 / 2 | GENN Φ (two-scale) |
| `solver` / `training` | — | `dT=200`, `n_iter=20`, ConvLSTM hidden 64, Adam 1e-3, batch 32 (see checkpoint) |

---

## `checks/` — what each script is for

**Evaluation / comparison pipeline** (the results):
`eval_test_days.py`, `export_obs_for_enkf.py`, `run_enkf_baseline.py`, `score_enkf.py`,
`compare_channels.py`, `plot_comparison.py`,
`plot_reconstruction_enkf.py` (density + heading arrows), `plot_velocity_enkf.py`
(speed magnitude + heading arrows),
`plot_reconstruction_sequence.py` (N consecutive frames as PNGs, optional EnKF panels),
`rederive_obs_std.py`.

**Module verification** (the test suite): `check_data_pipeline.py`,
`check_map_orientation.py`, `check_navigation.py` (A* vs Dijkstra/BFS),
`check_observation_model.py`, `check_prior_model.py`, `check_variational_solver.py`
(shape / zero-centre / differentiability).

**Map figures for the deck**: `plot_nav_mask.py` (walkable grid) and
`plot_obstacle_map.py` (obstacle occupancy). The convergence curve, training table
and architecture diagrams come from `checks/plot_architecture.py` and
`checks/plot_training_results.py`. The eight pre-2026-09-07 decks and their build
scripts were deleted on 2026-09-08; `slides/build_meeting4_deck.py` is the only
deck builder left, and `methods/varnet/SUPERSEDED.md` keeps the findings that
were only recorded in those decks' notes.

---

## Gotchas

- **`torch` on GPU only** — never on the login node (see TL;DR note).
- **`/tmp` is node-local** — a compute node cannot read the login node's `/tmp`
  (incl. the session scratchpad). Write compute-node outputs to the shared project
  filesystem (`check_outputs/`, `runs/`, …), not `/tmp`.
- **Two projects, two `config.py`** — the EnKF driver imports only `pedpred.*` from
  `Partial_observation`, never that project's `config`, to avoid a module-name clash.
- Reconstruction/velocity figures need the EnKF outputs (`check_outputs/enkf/est_<day>.npz`)
  to exist first (run the EnKF step before them).
