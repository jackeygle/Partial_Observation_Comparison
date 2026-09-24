# Crowd-field reconstruction on ATC — 4DVarNet reproduction

> Part of [Partial Observation Comparison](../../PROJECT_OVERVIEW.md) — see the project overview for the problem statement, the scoring-convention pitfall that decides the ranking, and which checkpoint backs which published number.

Reproducing the **4DVarNet** learned variational data-assimilation framework
(Fablet et al. 2020, [arXiv:2007.12941](https://arxiv.org/abs/2007.12941)) on the
**ATC pedestrian dataset**: reconstruct the full crowd field (density + velocity)
over a time window from a few robots' partial, noisy observations.

Two models from this directory are in the final six-method comparison, evaluated
by `supervisor_evaluation/evaluate.py` (see the repository [README](../../README.md)):

| Row | Run | What it is |
|---|---|---|
| 4DVarNet | `runs/varnet_mse5_h96_s3/ckpt_00080.pt` | plain MSE (Eq. 14); of five seeds, the one with the lowest validation error |
| 4DVarNet (aug. head) | `runs/varnet_aughead_obs_h96_s0/ckpt_00090.pt` | ours: σ inside the prior (`aug0`), a Gaussian-NLL observation term, and a dedicated variance head |

This README is written so someone else can reproduce the whole thing from the raw
data. **All parameters live in `crowdcore/config.yaml` (single source of truth)**; every
command reads its defaults from there, and command-line flags override for one run.

---

## TL;DR — reproduce end to end

```bash
# 0. environment (Aalto Triton), from the repository root
source sbatch/_env.sh           # module load + PYTHONPATH

# 1. (data is already gridded — see "Data" below if you must rebuild from CSV)

# 2. train: plain MSE, 5 seeds (full 32-day train split, paper window dT=200), self-chaining
for s in 0 1 2 3 4; do
  sbatch --job-name=varnet_mse5_h96_s$s methods/varnet/sbatch/submit_mse5_chain.sbatch $s 0
done
#    and the uncertainty model (aug. head), one seed
sbatch --job-name=varnet_aughead_s0 methods/varnet/sbatch/submit_aughead_chain.sbatch 0 0

# 3. choose each run's epoch on the validation split (-> runs/<run>/select_valid.json)
sbatch --export=ALL,RUNS="mse5_h96_s0 mse5_h96_s1 mse5_h96_s2 mse5_h96_s3 mse5_h96_s4 aughead_obs_h96_s0" \
  methods/varnet/sbatch/submit_select.sbatch

# 4. evaluate against the other methods on the 7 test days
python3 supervisor_evaluation/evaluate.py prepare     # package the selected checkpoints
sbatch supervisor_evaluation/sbatch/full.sbatch
```

> **GPU, not the login node.** Anything that touches `torch` (training, evaluation,
> reconstruction figures) must run on a GPU node — `sbatch`, or
> `srun --partition=gpu-debug --gres=gpu:1 --time=00:15:00 bash -c '...'`.

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
| `runs/varnet_mse5_h96_s<seed>/`, `runs/varnet_aughead_obs_h96_s0/` | the runs behind the two final models; the reported epoch is in each run's `select_valid.json` (`checks/model_io.py:reported_ckpt`) |
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
Defined by `sunday_atc_{train,valid,test}.lst` in the data root (copies in
`supervisor_evaluation/data_split/`); the full description, and how to use a different
split, is in the repository [README](../../README.md#dataset-split).

| split | days | date range | used for |
|---|---|---|---|
| **train** | 32 | 2012-10-28 → 2013-06-09 | training |
| valid | 7 | 2013-06-16 → 2013-07-28 | choosing each run's epoch and the reported seed |
| **test** | 7 | 2013-08-11 → 2013-09-29 | the final evaluation, identical days for every method |

Train is strictly earlier than validation, validation strictly earlier than test.

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
(cd ../.. && sbatch --job-name=varnet_mse5_h96_s0 methods/varnet/sbatch/submit_mse5_chain.sbatch 0 0) # one seed, self-chaining --resume
#   runs: train.py --hidden 96 --kt 5 --days 32 --dT 200 --epochs 150 --batch 32 --amp --loss supervised
#         --iter-schedule 0:5:1e-3,25:10:7e-4,50:15:5e-4,75:20:3e-4   (§3.4 curriculum)
#   out : runs/varnet_mse5_h96_s0/ckpt_<epoch>.pt every 10 epochs, varnet_last.pt (resume) + metrics.jsonl

# quick smoke test
srun --partition=gpu-debug --gres=gpu:1 --time=00:15:00 \
     bash -c 'module load scicomp-pytorch-env/2026.1; python3 -u -m methods.varnet.train --days 1 --steps 50'
```
- Loss: `--loss supervised` = `‖x_rec − X‖²` (paper Eq.14, unweighted MSE). `--no-noise`
  observes without sensor noise.
- **Aug. head** (`sbatch/submit_aughead_chain.sbatch`): the same network and schedule,
  with log σ² iterated inside the prior operator G(x) (`aug0`), the observation term of
  the variational cost replaced by a Gaussian NLL sharing that σ² (`--obs-nll`), and σ²'s
  update produced by a head of its own that sees `[h, x]` (`--var-head`). Trained with an
  NLL loss; the reported epoch (90) is chosen on validation like the MSE runs.
- `metrics.jsonl` logs train loss / blind-zone MSE / R-score / the X0 baseline per epoch.

---

## Evaluate

The reported evaluation is not in this directory: `supervisor_evaluation/evaluate.py`
scores both 4DVarNet models next to the other methods, with the same test days,
observations, cell set and clip bounds, and scores the aug. head's σ̂ in physical units
on the same frames as DINCAE and the EnKF. Neither 4DVarNet model is initialised from
truth: the solver starts from the observation-filled `X0`.

For this method alone, `checks/eval_uncertainty.py` (single checkpoint or a seed
ensemble) and `checks/eval_comprehensive.py` remain for diagnostics; their outputs under
`check_outputs/eval/` predate the final protocol and are not the reported numbers.

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

**Checkpoint selection and evaluation**: `select_checkpoint.py` (epoch on the
validation split -> `select_valid.json`), `model_io.py` (`reported_ckpt`: resolves the
selected checkpoint, used by every evaluator), `eval_uncertainty.py`,
`eval_comprehensive.py`, `eval_sigma_fair.py`, `plot_uncertainty.py`. The reported
cross-method evaluation is `supervisor_evaluation/evaluate.py`;
`compare/compare5.py` is the earlier cross-method scorer it reuses for accuracy.
`plot_reconstruction_sequence.py` draws N consecutive frames as PNGs;
`rederive_obs_std.py` re-derives the observation-noise std in `config.yaml`.

**Module verification** (the test suite): `check_data_pipeline.py`,
`check_map_orientation.py`, `check_navigation.py` (A* vs Dijkstra/BFS),
`check_observation_model.py`, `check_prior_model.py`, `check_variational_solver.py`
(shape / zero-centre / differentiability).

**Map and architecture figures**: `plot_nav_mask.py` (walkable grid) and
`plot_obstacle_map.py` (obstacle occupancy). The architecture diagrams come from
`checks/plot_architecture.py`, drawn from the reported model
(`checks/model_io.py:baseline_ckpt`). `plot_training_results.py` and
`plot_results.py`, which plotted the hidden=32 capacity sweep (b0/a2/a4) and b0's
speed, were deleted with those runs on 2026-09-13. The eight pre-2026-09-07 decks and their build
scripts, and the whole of `slides/`, were deleted on 2026-09-08/09; the project
no longer builds presentation decks from this repository.
`methods/varnet/SUPERSEDED.md` keeps the findings that were only recorded in
those decks' notes.

---

## Gotchas

- **`torch` on GPU only** — never on the login node (see TL;DR note).
- **`/tmp` is node-local** — a compute node cannot read the login node's `/tmp`
  (incl. the session scratchpad). Write compute-node outputs to the shared project
  filesystem (`check_outputs/`, `runs/`, …), not `/tmp`.
- **Two projects, two `config.py`** — the EnKF driver imports only `pedpred.*` from
  `methods/enkf/enkf_lab/`, never that copy's `config`, to avoid a module-name clash.
