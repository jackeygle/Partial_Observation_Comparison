# methods/dincae — reproducing DINCAE on the ATC crowd field

> Part of [Partial Observation Comparison](../../README.md) — see the root README for the problem statement, the scoring-convention pitfall that decides the ranking, and which checkpoint backs which published number.

The second technical route. The first (4DVarNet reproduction + EnKF comparison) is
already complete, in [`../varnet/`](../varnet/); this directory
**does not depend on its conclusions**, only reuses its data pipeline
(`observation_model.py` / `navigation.py` / `data_pipeline/`) and the **exact same
observation configuration**.

Papers: Barth et al. 2020 (GMD 13, 1609) = **DINCAE 1.0**; Barth et al. 2022 (GMD 15,
2183) = **DINCAE 2.0**. Reference implementation:
[`../../reference/DINCAE.jl`](../../reference/DINCAE.jl) (Julia v2.0.6).

**The goal is to strictly reproduce the paper's method**, not to improve on it
first. Every place that deviates from the paper is listed below with its reason.

---

## Terminology

The paper comes from an ocean remote-sensing background, and some of its terms
(climatology, cloud) read as foreign transplants in a crowd field. **This directory
uses the right-hand column throughout**; the paper's original terms appear only in
this table, for cross-referencing against the paper/reference implementation.

| Paper / reference implementation | This directory | What it is |
|---|---|---|
| climatology, `remove_mean`, `meandata` | **per-cell mean field** | each cell's average over the 32 training days, a `(4,36,12)` table. Stored in `mean` inside `artifacts/state_stats.npz` |
| anomaly | **residual** | what's left after subtracting the per-cell mean field, i.e. "more or less than usual". The network works in this space |
| cloud / non-cloud (SST's cloud occlusion) | **missing / observed** | here it's outside the robots' field of view, not cloud |
| `obs_err_std`, sigma^2_obs | observation error variance | taken as the constant 1, see "Encoding" below |

Identifiers in the code follow this table: `residual_mse()`, `resid_std`,
`dev_resid_mse`. The array names in the artifact file `state_stats.npz`
(`mean` / `std` / `count` / `valid_mask`) are neutral and left unchanged.

---

## Directory layout

| path | role |
|---|---|
| `state.py` | **state definition**: the four channels, per-channel validity rules, per-channel transforms (`var` goes through log1p), and the **per-cell mean field** (each cell's average over the 32 training days) + residual std |
| `encoding.py` | information-form encoding: `y/sigma^2` + `1/sigma^2`, missing = both pieces 0 |
| `dataset.py` | one day -> training samples; observation config read from `crowdcore/config.yaml`; disk cache |
| `model.py` | U-Net + SumSkip + refinement step + sigma-hat parameterisation (Eq.6-7) |
| `losses.py` | Gaussian NLL (Eq.3), summed after independently normalising each variable |
| `train.py` | training loop (Adam / grad clip 5 / periodic checkpoints) |
| `checks/` | self-checks, evaluation, and scripts measuring the data itself |
| `sbatch/` | SLURM submission scripts |
| `artifacts/` | **artifacts**: `state_stats.npz` (read by training), `decay_tables.*` (diagnostic only) |
| `cache/` | **artifacts**: the encoding cache (~23 GB, safe to delete and rebuild at any time) |
| `check_outputs/` | **artifacts**: metric JSONs produced by `checks/` scripts |
| `runs/` | **artifacts**: checkpoints, `metrics.jsonl`, SLURM logs |

Inside `checks/`:

- `check_encoding.py` — self-checks `encoding.py`'s invariants (are both pieces 0 at
  missing cells, is the target mask correct, ...)
- `evaluate.py` — evaluation: sigma-hat calibration, variance retention, both MSE
  conventions, and optional multi-epoch output averaging (`--average-checkpoints`)
- `measure_coverage_revisit.py` / `measure_obs_age.py` / `measure_decorrelation.py`
  — measurements about **the data itself** (coverage, observation age, temporal
  autocorrelation), not on the training path
- `measure_decay_tables.py` — an early calibration for an age-dependent-sigma^2
  variant. **Not part of the paper's method**, kept because the autocorrelation
  table it produces is a valid measurement. Artifact in `artifacts/decay_tables.*`.

---

## End to end

```bash
module load scicomp-pytorch-env/2026.1
cd /scratch/work/zhangx29/Thesis_Project/methods/dincae

# 1. Per-cell statistics (once; pure numpy/scipy, login node is fine, ~8 minutes)
python3 -m methods.dincae.state         # -> artifacts/state_stats.npz

# 2. Encoding self-check (~2 minutes, login node is fine)
python3 -m methods.dincae.checks.check_encoding   # expect PASS

# 3. Training (GPU node)
sbatch sbatch/submit_train.sbatch            # 200 epochs, self-chaining + --resume
#   -> runs/dincae_full/{last.pt, ckpt_*.pt, metrics.jsonl}
#   the first epoch builds cache/ (~13 GB); every epoch after that only reads it

# 4. Evaluation (GPU node)
sbatch sbatch/submit_eval.sbatch --split test
#   -> check_outputs/eval_single_00070/dincae_metrics_test.json
#   (the PUBLISHED configuration: the single epoch-70 checkpoint. The
#    16-checkpoint average lives in check_outputs/eval/ and now needs an
#    explicit --average-checkpoints, see "Checkpoint policy" below.)
```

**torch only runs on GPU nodes** (`sbatch`, or
`srun --partition=gpu-debug --gres=gpu:1 --time=00:15:00`), never on the login
node. `state.py` and `checks/measure_*.py` are pure numpy/scipy and can run on the
login node.

---

## Method

### Encoding (`encoding.py`)

The paper does not "fill in" missing values. The input is two slices, `y/sigma^2`
and `1/sigma^2`; missing = both are 0 = zero precision (1.0 sec.3; 2.0 sec.2.3;
`data.jl:311-325`). **sigma^2_obs is taken as the constant 1** (the paper/code
default `obs_err_std = 1`; 1.0 sec.3 says the exact value of this constant does not
matter, it gets absorbed by the first layer's weights). So:

```
observed  ->  scaled = (fwd(y) - mean)/std ,  invvar = 1
missing   ->  scaled = 0                   ,  invvar = 0
```

Only the 3 neighbouring frames are used in time (`ntime_win = 3`, 1.0 sec.3's
"previous day / today / next day"). If a cell is not observed in any of
{t-1, t, t+1}, it is missing -- **there is no "fall back to an older observation"
in the paper**.

**30 input channels**: row, col (2) + cos/sin of the diurnal and weekly cycle (4) +
`dt in {-1,0,+1}` x (residual[4], mask[4]) (24). **8 target channels**: one
(residual*mask, mask) pair per channel.

### Per-channel validity (`state.channel_valid`)

Where each of the four channels "is defined" differs, and every statistic and loss
term only operates where it is defined. This is not a deviation from the paper --
it is the paper's own principle, "missing = zero precision," applied to our data:

```
density : defined everywhere (density=0 is a genuine measurement: "nobody is here")
vx, vy  : density > 0    -- velocity on an empty cell is a placeholder
var     : vel_var > 0 <=> at least 2 people in the cell (variance of 1 point is undefined)
```

The `var` channel (the **spread of pedestrian velocities within a cell**, a real
physical field) is **both the 4th channel being reconstructed** and the source of
velocity uncertainty (`vel_var / density` = the standard error of the cell mean) --
the two roles do not conflict.

### Network (`model.py`)

Follows the 2.0 architecture, not the 1.0 paper's Table 1:

- **no fully-connected bottleneck, no dropout** (2.0 sec.4: an FC layer requires
  identical input size at train and inference time)
- **SumSkip (addition) rather than concat** (2.0 sec.2.1 Eq.2; same benchmark
  0.3835 -> **0.3604**)
- **MeanPool** (hardcoded at `model.jl:268`). Warning: 2.0's Table 1 text says max
  pooling -- **the paper and code disagree here**, and this was never re-verified
  in 2.0; `--pool max` is available for an A/B test
- **refinement step** (2.0 sec.2.2 Eq.4, alpha=0.3/alpha'=0.7; Table 2: 0.60 ->
  **0.55**)
- 36x12 -> 18x6 -> 9x3 -> 5x2, filters 32/64/96, ~320k parameters

### Loss (`losses.py`)

The Gaussian NLL from 2.0 Eq.3. Each output variable is **computed independently,
normalised by its own N, then summed** (`model.jl:132-148`), so the relative weight
between channels is entirely decided by the learned `1/sigma-hat^2`, never set by
hand. The `truth_uncertain` KL branch is implemented in the code but **not used by
default** (neither paper has it).

---

## Three deviations from the paper

| # | Deviation | Reason |
|---|---|---|
| 1 | **The target is ground truth**, not the observation with missing values | The paper has no ground truth (that is the whole reason DINCAE exists), we do. Using the observation as the target would mean deliberately throwing away information we already have |
| 2 | **Each channel's residual is normalised to unit variance** | The paper uses `obs_err_std=1` for every variable, which is fine on SST (residuals are naturally O(1) degC). Our four channels' residual variances differ by three orders of magnitude, and without normalising, the `log sigma-hat^2` term lets a small-magnitude channel earn negative loss for free -- measured: train NLL drops by 5 units while density's dev MSE actually rises from 0.0254 to 0.0419. The reference code's `normalize2` (`data.jl:111-120`) does exactly this |
| 3 | **The `var` channel goes through log1p** | 1.0's conclusion section: the method "can easily be extended to a log-normal distribution to handle concentration-like variables." `var` is non-negative and heavy-tailed. Without the transform its normalised dev MSE oscillates between 8 and 62 (1.0 = the level of just outputting the per-cell mean); with log1p it drops to 0.7-1.6 |

**Not used**: age-dependent sigma^2, multi-scale temporal aggregation, random
per-track dropout, `truth_uncertain`. The first two were implemented at one point
(only `checks/measure_decay_tables.py`'s measurement survives now); the last two
exist in the paper/code but are unused -- none of these are part of the paper's
method.

---

## Two hard constraints on ATC (`checks/check_encoding.py` prints these)

```
single-frame coverage of walkable cells:   55.5%
covered at least once within 3-frame window: 63.1%  -> 36.9% of cells have both
                                                        slices at 0, relying only
                                                        on coordinates + clock +
                                                        per-cell mean
target-defined fraction  density: 67.1%    vx/vy: 13.2%    var: 12.8%
```

The paper's data is **one snapshot per day**; "previous day / today / next day"
gives far higher coverage there. Ours is sampled at 1 Hz with a field that
decorrelates in about 2 seconds (`checks/measure_decorrelation.py`), so the
original paper's setup naturally gets little observational information here. These
two numbers are key to explaining performance, not a bug.

---

## Evaluation convention (`checks/evaluate.py`)

**Both MSE conventions are reported**, because they lead to different conclusions
(a pitfall already hit once in the 4DVarNet work (methods/varnet)):

- `ours` — only on cells where that channel **is defined**, restricted to walkable
  cells. The convention used during training.
- `v4dvar` — exactly follows `methods/varnet/checks/eval_test_days.py`: the raw
  field, **all** cells (including non-walkable ones), four channels unweighted,
  blind = `mask < 0.5`, with the same physical clipping as the EnKF. This
  convention scores us on cells we were never trained on and is unfavourable to
  us; it is reported for comparability.

Both are given in `_noclip` variants too. **The gap between `noclip` and `clip` is
itself a convergence diagnostic**: in the information form, `m = x1*sigma-hat^2`
and sigma-hat^2 is capped at `1/mu = 1000` (Eq.6 clamping), so an unconverged model
can output mu~1000, and after inverting `var`'s transform that becomes an
astronomical number (measured during a smoke test: physical MSE reaching 10^22).
Once converged the two should be close.

Two more items both come from the paper: **sigma-hat calibration** (2.0 sec.5.2,
binning by predicted SD into 10 bins and comparing to the actual RMS; note 2.0
applies a **global adjustment factor**, i.e. the raw sigma-hat's absolute scale is
biased and what's trustworthy is the structure and ordering) and **variance
retention** (1.0 Fig.8 / 2.0 Table 3; RMSE favours a smooth field, so variance must
be reported separately).

---

## Gotchas

- **Module-name shadowing**: `methods/varnet` also has a `losses.py`. So scripts in
  this directory always `sys.path.insert(0, project_root)` before
  `sys.path.append(methods/varnet)` -- getting the order backwards imports the wrong
  file.
- **Changing the encoding means clearing `cache/`.** The cache key includes
  `CACHE_VER` and the observation config, so changing those invalidates it
  automatically; but changing the *contents* of `state_stats.npz` (same day count)
  does not trigger a rebuild, and needs a manual `rm -rf cache`.
- **`/tmp` is node-local** -- a compute node cannot see the login node's `/tmp`.
  Artifacts are written inside the project instead.
- **Do not set separate defaults for the observation config here** -- always read
  `crowdcore/config.yaml` (`dataset.obs_config`), otherwise the two technical
  routes' observation scenarios stop being comparable.

## Checkpoint policy

Every **published** DINCAE number comes from a single checkpoint,
`runs/dincae_full/ckpt_00070.pt`, picked on DINCAE's own validation set —
matching the other three methods, none of which ensemble or average.

The reference implementation instead averages the outputs of checkpoints saved
every 10 epochs, which scores ~1.7% better on RMSE. That path still exists
behind `--average-checkpoints`, and its result is archived in
`check_outputs/eval/`; it is reported as a measured side quantity, not as the
headline. `load_models()` refuses to average silently — passing no
`--ckpt-glob` now raises instead of quietly returning a number ~1.7% away from
the reported one with nothing in the output to distinguish them.
