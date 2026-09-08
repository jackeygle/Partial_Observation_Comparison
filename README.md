# Partial Observation Comparison

**The problem.** Three robots move through a real train-station corridor (the
ATC pedestrian dataset, Osaka). Each senses only the people within a small
radius of itself — that is the "partial observation". The task is to
reconstruct the **full** crowd density and velocity field over the whole
corridor (a 36x12 grid, one frame per second) from those sparse sightings.

**What is compared.** Four independent reconstruction methods — 4DVarNet,
DINCAE, Senseiver and a localised ensemble Kalman filter. Three of them are
additionally scored on whether they know *how wrong* they are (an uncertainty
estimate), because a wrong-but-honest answer is more useful downstream than a
wrong-and-confident one. Two of those uncertainty designs are ours, added to
4DVarNet, which has none in the original paper.

**The headline finding.** The ranking of these four methods is decided by a
scoring convention that most papers do not state. Same predictions, same days:
DINCAE is first under one convention and last under the other. See
[Scoring convention](#scoring-convention-read-this-before-quoting-any-number).

## Contents

| Section | For |
|---|---|
| [The four methods](#the-four-methods-plus-two-uncertainty-designs-of-our-own) · [Scoring convention](#scoring-convention-read-this-before-quoting-any-number) · [Test days and the obstacle region](#test-days-and-the-obstacle-region) | understanding what is here |
| [Looking at results without running anything](#looking-at-results-without-running-anything) | a quick look, no cluster needed |
| [Which file backs which published number](#which-file-backs-which-published-number) | reproducing or citing a specific number |
| [Running things](#running-things) (environment, layout, training, evaluation, figures) | doing the work |
| [Data, and the minimal package to transfer](#data-and-the-minimal-package-to-transfer) | moving this to another machine or person |
| [Troubleshooting](#troubleshooting) | when something misbehaves |

---

# Understanding what is here

## The four methods, plus two uncertainty designs of our own

Counted as **four reconstruction methods**. 4DVarNet appears three times below
because we added two uncertainty designs to it that the paper does not have —
those are variants of one method, not separate methods. This is the framing
agreed with the supervisor on 2026-09-07.

| Directory | Method | Uncertainty output |
|---|---|---|
| `methods/varnet/` | 4DVarNet (plain MSE, Eq.14) — the reconstruction method | none |
| `methods/varnet/` | ... + `aug0`: sigma inside the prior operator `G(x)`, NLL (**ours**) | learned sigma-hat + epistemic |
| `methods/varnet/` | ... + `vsb0`: sigma from a separate read-out head, NLL (**ours**, negative control) | learned sigma-hat + epistemic |
| `methods/dincae/` | DINCAE (single checkpoint, see below) | sigma-hat (information form) |
| `methods/senseiver/` | Senseiver | none |
| `methods/enkf/` | localised EnKF (+ PedPred3 forward model) | ensemble spread |

`aug0` is the design that works — better CRPS, and its sign is stable across
both scoring conventions. `vsb0` is kept as the negative control showing why
bolting uncertainty on afterwards underperforms. DINCAE and the EnKF have
uncertainty *built into* the method (information form, ensemble spread) rather
than as a separate design.

**Every headline row is a single model**, default configuration, seed 0, each
one's checkpoint picked on its own validation set — no ensembling and no
checkpoint averaging, because neither paper reports one. DINCAE's reference
implementation *does* average the outputs of checkpoints saved every 10 epochs,
and `checks/evaluate.py` still supports it behind `--average-checkpoints`, but
the reported number comes from **one** checkpoint, `ckpt_00070.pt`. The
16-checkpoint average is kept as a measured side quantity (worth 1.7% RMSE,
less than picking the right single checkpoint), not as the headline.

`methods/enkf/enkf_lab/` is a **byte-identical, read-only copy** of
`/scratch/work/zhangx29/Partial_observation`, its files deliberately chmod 444.
To change anything, edit `enkf_opt/` instead, and it must pass the bit-identical
comparison in `methods/enkf/checks/verify_enkf_opt.py` (`np.array_equal`, not
`isclose`).

## Scoring convention: read this before quoting any number

**Rankings flip depending on the convention.** Take DINCAE's own two numbers,
same model, same predictions, same days: pooled RMSE **0.329** under one
convention and **0.757** under the other, more than double.

- `defined` — only cells where that physical quantity exists (velocity needs
  density > 0 there). DINCAE was trained exactly this way, so it comes first.
- `allcells` — every cell, including the **88.4%** of blind velocity cells that
  are empty. An empty cell has no people, hence no velocity, and the `0` the
  pipeline stores there is a placeholder, not a measurement. The three methods
  trained on the full field learned to output 0 there and get that part for
  free; DINCAE was never trained to, and is crushed by this term.

**So any single number quoted without saying which cells it scores is
incomplete** — not wrong exactly, but not comparable to a number computed the
other way. `compare/compare5.py` always reports both side by side; the single
definition of the rule lives in `methods/dincae/state.py:channel_valid()`.

Also: 4DVarNet's numbers carry a **~4e-4 reproducibility floor**. Its inference
pass itself backpropagates through autograd, and conv backward's reduction uses
atomicAdd, whose accumulation order differs run to run — so re-running an
identical config differs in the 4th decimal place. That is noise, not a
regression. Senseiver, DINCAE and the EnKF are bit-reproducible. See
`refactor_baseline/README.md`.

## Test days and the obstacle region

7 held-out days, `split=test`, never seen during training or checkpoint
selection:

```
atc-20130811  atc-20130818  atc-20130825  atc-20130901  atc-20130915
atc-20130922  atc-20130929
```

Grid: 36x12 cells, 4 channels (density, vx, vy, var). 290 of the 432 cells are
physically walkable; the remaining 142 (32.9%) are the obstacle region.

**Obstacle-region finding** (`methods/dincae/checks/measure_obstacle_region.py`,
full 7-day split, both runs scored on the checkpoints their reported numbers
use). Obstacle-region truth is non-zero only 2.99–3.11% of the time. The
information-form baseline is wildly wrong there on the velocity channels,
because it is never taught a target for a quantity that only exists where
`density>0` — which is never true in the obstacle region. The
`--full-field-loss` ablation supervises those cells against physical 0 and
closes almost all of the gap:

| channel | baseline obstacle MSE | full-field obstacle MSE | ratio |
|---|---|---|---|
| density | 0.00482 | 0.00115 | 4.2x |
| vx | 1.12610 | 0.00807 | 140x |
| vy | 3.39451 | 0.00335 | **1015x** |
| var | 0.04219 | 0.00024 | 177x |

At essentially no cost on walkable cells (vx walkable MSE actually *improves*,
0.092 vs 0.154). So this is **not a bug**: DINCAE's information form
structurally cannot learn "obstacle = 0" unless the obstacle region is put in
the loss, and doing so is nearly free.

---

# Looking at results without running anything

Everything here is already computed and sitting in the repo as small JSON/PNG
files. No GPU, no Slurm, no environment setup.

| What | Where |
|---|---|
| The main accuracy comparison figure (per-channel, **MSE**) | `compare/results/compare5.png` |
| Raw numbers, all four scoring conventions | `compare/results/compare5_final.json` |
| The current meeting deck (2026-09-07, 6 slides) | `slides/meeting4_deck.pdf`, speaker notes in `slides/meeting4_deck_notes.md` |
| A reconstructed field as a picture, all 4 methods side by side | `compare/results/reconstruction_atc-20130811_dincae-senseiver-varnet-enkf.png` |
| DINCAE's obstacle-region result | `methods/dincae/check_outputs/eval/obstacle_region.json` |
| Everything else (architecture diagrams, uncertainty calibration, speed) | `methods/*/check_outputs/**/*.png` — `unc_*` uncertainty, `mt_*` uncertainty meeting figures, `arch_*`/`fw_*` architecture, `speed_*` latency |

# Which file backs which published number

"The DINCAE weights" is not a well-defined request: `runs/dincae_full/` holds 17
checkpoints and which you pick changes the number. Same for `varnet_best.pt` vs
`varnet_last.pt`. This table is the authoritative mapping; everything else under
`runs/` is an intermediate or an exploratory ablation.

| Published as | Exact file(s) | Result JSON | Why this one |
|---|---|---|---|
| 4DVarNet MSE | `methods/varnet/runs/varnet_mse5_s{0..4}/varnet_best.pt` | `compare/results/compare5_final.json` (`"4DVarNet MSE s*"`) | `compare5_final.json`'s protocol records `ckpt: varnet_best.pt` |
| 4DVarNet `aug0` | `methods/varnet/runs/varnet_aug0_s{0..4}/varnet_best.pt` | `methods/varnet/check_outputs/eval/uncertainty_aug0.json` | same convention |
| 4DVarNet `vsb0` | `methods/varnet/runs/varnet_vsb0_s{0..4}/varnet_best.pt` | `methods/varnet/check_outputs/eval/uncertainty_vsb0.json` | same convention |
| DINCAE | `methods/dincae/runs/dincae_full/ckpt_00070.pt` — **one file** | `methods/dincae/check_outputs/eval_single_00070/dincae_metrics_test.json`, plus `eval/uncertainty_dincae.json` | best on DINCAE's own validation set; recorded in `compare5_final.json` as `source: .../eval_single_00070/...` |
| Senseiver | `methods/senseiver/runs/senseiver_A/best.pt` | folded into `compare5_final.json` (`"Senseiver"`) | only trained model; `best.pt`, not `last.pt` |
| EnKF | no weights — exported fields in `methods/enkf/check_outputs/enkf_k1_full/est_*.npz` | `compare5_final.json` (`"EnKF k1"`), `methods/enkf/check_outputs/eval/uncertainty_enkf_k1.json` | it is a filter, not a trained model |

Two traps, both of which silently produce a *different number* rather than an
error:

1. **Do not average DINCAE's 16 checkpoints.** `checks/evaluate.py` used to do
   this by default (it is the reference implementation's behaviour). It now
   refuses unless you pass `--average-checkpoints`, because the result is ~1.7%
   RMSE away from the reported one with nothing in the output to say which
   configuration produced it.
2. **`varnet_best.pt` vs `varnet_last.pt` is a methodological choice.** `best`
   is selected per-run on validation, so different arms stop at different
   epochs; `last` is always epoch 149 and gives a same-epoch comparison. The
   headline table uses `best`; `compare5.py --ckpt-name` documents the trade-off.

The EnKF's overconfidence is diagnosed separately in
`methods/varnet/check_outputs/eval/enkf_spread_growth.json` (via
`methods.enkf.checks.diag_enkf_spread_growth`) — verdict `"contractive"`: the
deterministic PedPred3 forecast model damps ensemble spread by ~65% per step
regardless of injected noise. Not a bug to fix; a property of the forward model
to report as-is.

---

# Running things

## Environment

```bash
cd /scratch/work/zhangx29/Thesis_Project
source sbatch/_env.sh          # module load + PYTHONPATH + PYTHONSAFEPATH=1
```

Once per shell, and afterwards always invoke code as `python3 -m <package.module>`.

**Never use a bare script path.** `python3 methods/varnet/train.py` puts
`methods/varnet/` on `sys.path[0]`, and the three methods each ship their own
**different** `losses.py` (same for `dataset.py`, `model.py`) — whichever hits
the search path first wins, silently. `PYTHONSAFEPATH=1` in `_env.sh` is what
blocks that.

The login node has **no GPU**. `--help` and pure-Python checks are fine there;
anything calling `torch` on real data needs `sbatch`, or an interactive node:

```bash
srun -p gpu-debug --gres=gpu:1 -t 00:15:00 --pty bash
```

## Repository layout

```
crowdcore/                shared by every method, method-agnostic
  config.py + config.yaml   single source of truth for every parameter
                            (grid resolution, robot count, sensing radius,
                            noise level, time-window length, ...)
  navigation.py             the real ATC map -> which cells are walkable
  observation_model.py      simulates the 3 robots sensing the crowd
  paths.py                  single source of truth for in-repo paths
  data/                     raw CSV -> gridded .h5 pipeline

methods/                  one subdirectory per method, none import each other
  varnet/                   4DVarNet (plain MSE, plus the 2 uncertainty designs)
  dincae/                   DINCAE convolutional autoencoder inpainting
  senseiver/                Senseiver sparse-sensor reconstruction
  enkf/                     localised EnKF (read-only vendor copy + editable copy)
  each has:
    train.py                (not enkf/, which is not learned)
    checks/                 evaluation and diagnostic scripts
    runs/                   trained checkpoints (gitignored)
    check_outputs/          figures and result JSON (tracked in git)
    sbatch/                 Slurm job scripts

compare/                  the ONLY place allowed to import more than one method
                          at once -- cross-method comparison and plots
slides/                   the current meeting deck and its builder
sbatch/_env.sh            the single environment entry point
```

## Run names under `runs/`

Only these matter for the headline numbers. Everything else (`a2`, `a4`, `b0`,
`ml5`, `nf5`, `vrb0`, `vrb1`, ...) is an exploratory ablation from along the way
and is **not** part of any reported comparison.

| Codename | Method | Means |
|---|---|---|
| `mse5` | 4DVarNet | plain MSE loss (Eq.14), 5 random seeds, no uncertainty output |
| `aug0` | 4DVarNet | sigma **inside** the prior operator `G(x)`, NLL — ours, the design that works |
| `vsb0` | 4DVarNet | sigma from a **separate read-out head**, NLL — ours, the negative control |
| `dincae_full` | DINCAE | baseline: information-form supervision (only defined cells in the loss) |
| `dincae_ff` | DINCAE | `--full-field-loss` ablation: every cell in the loss, obstacle target = 0 |
| `senseiver_A` | Senseiver | the one trained model |
| `enkf_k1_full` | EnKF | exported ensemble fields, `obs_every_k=1`, full 7-day test split |

`s0`..`s4` are random seeds (weight init + data ordering); the headline table
reports mean±std across all five. `obs_every_k=4` was studied earlier and
dropped entirely on 2026-09-08.

## Sanity checks before committing to a long run

Fastest possible (seconds, no GPU, no real data — synthetic tensors through the
4DVarNet solver, confirms imports and that both uncertainty designs produce
finite gradients):

```bash
python3 -m methods.varnet.scratch.smoke_aug
```

With real data (needs a GPU; a few minutes each):

```bash
srun -p gpu-debug --gres=gpu:1 -t 00:15:00 bash -c '
  cd methods/varnet   && python3 -u -m methods.varnet.train --steps 5 --allow-cpu 2>&1 | tail -5
  cd ../dincae        && python3 -u -m methods.dincae.train --cache-dir "" --days 2 --epochs 2 \
                          --max-frames 3000 --amp --out runs/smoke_check 2>&1 | tail -5
  cd ../senseiver     && python3 -u -m methods.senseiver.train --days 1 --frames 200 --epochs 1 \
                          --allow-cpu --out runs/smoke_check 2>&1 | tail -5
'
```

`--cache-dir ""` for DINCAE matters: without it a smoke run reads and writes the
same 13 GB cache the real training run uses.

## Training

4DVarNet's `*_chain.sbatch` scripts **self-resubmit**: one `sbatch` call runs for
its `--time` budget then submits its own successor (up to `MAX_GEN=12`), so a
~150-epoch run survives any single job's time limit. Each call trains **one
seed** — loop over seeds yourself.

```bash
# 4DVarNet -- plain MSE (the headline "4DVarNet" row), 5 seeds
for s in 0 1 2 3 4; do
  sbatch --job-name=varnet_mse5_s$s methods/varnet/sbatch/submit_mse5_chain.sbatch $s 0
done

# 4DVarNet -- uncertainty design 1: sigma inside G(x) (aug0), 5 seeds
for s in 0 1 2 3 4; do
  sbatch --job-name=varnet_aug0_s$s methods/varnet/sbatch/submit_aug_chain.sbatch $s 0
done

# 4DVarNet -- uncertainty design 2: read-out head only (vsb0). No dedicated chain
# script; run train.py with enough --time, or copy submit_aug_chain.sbatch and
# swap --augmented-var for --var-h-only
python3 -m methods.varnet.train --loss nll --var-h-only --outdir runs/varnet_vsb0_s0

# DINCAE -- baseline and the obstacle-region ablation (~16-27 GPU-hours each)
sbatch methods/dincae/sbatch/submit_train.sbatch --out runs/dincae_full
sbatch methods/dincae/sbatch/submit_train.sbatch --out runs/dincae_ff --full-field-loss

# Senseiver (up to 1 day)
sbatch methods/senseiver/sbatch/submit_train.sbatch --out runs/senseiver_A
```

The EnKF has no training step — it is a filter, not a learned model. It does
need *exporting*: simulate the robots' observations per day, then run the filter
over them. CPU-only (no GPU helps a Kalman filter) and genuinely slow — one real
day took ~53 CPU-hours in the run that produced `enkf_k1_full/`. Submit one job
per day so the seven run in parallel:

```bash
python3 -m methods.enkf.checks.export_obs_for_enkf --outdir check_outputs/enkf_k1_full --obs-every-k 1 --frames 0
for day in atc-20130811 atc-20130818 atc-20130825 atc-20130901 atc-20130915 atc-20130922 atc-20130929; do
  sbatch --wrap="source sbatch/_env.sh && cd methods/enkf && python3 -u -m methods.enkf.checks.run_enkf_baseline \
    --dir check_outputs/enkf_k1_full --only $day --ensemble 100 --radius 7 --frames 0" \
    --time=60:00:00 --cpus-per-task=4 --mem=64G --job-name=enkf_$day
done
```

`run_enkf_baseline.py --help` does **not** list `--dir`/`--frames`/`--ensemble`/
`--radius`/`--only`: it parses vendor config args first and prints that parser's
help before reaching its own `argparse.ArgumentParser()` further down
(`checks/run_enkf_baseline.py:123-133`). The flags above are real.

Checkpoints land in each method's `runs/<name>/` (gitignored). Slurm logs land
alongside as `runs/slurm_<job>_<id>.out` and **are** tracked, so you can always
check what a past run did without re-running it.

## Reproducing the headline comparison

```bash
sbatch sbatch/submit_compare5.sbatch      # ~2h, needs the checkpoints above
python3 -m compare.plot_compare5          # regenerates compare/results/compare5.png
```

The defaults reproduce exactly the published row set — all four methods, both
uncertainty designs, five seeds each, and DINCAE's single epoch-70 checkpoint.
A rerun writes `compare/results/compare5.json`; the published artefact is
`compare5_final.json`, kept as a separate file so a rerun cannot silently
overwrite the numbers this README quotes.

**Expected output**, pooled **RMSE**. A rerun should land within ~0.01 of these
(4DVarNet rows carry the reproducibility floor described above). Note the figure
`compare5.png` plots per-channel **MSE**, so its totals are these numbers
squared — 0.329 there appears as 0.108:

| Method | `defined` convention | `allcells` convention |
|---|---|---|
| Senseiver | 0.343 | 0.168 |
| 4DVarNet MSE (mean of 5 seeds) | 0.352 | 0.186 |
| 4DVarNet `vsb0` (mean of 5 seeds) | 0.388 | 0.212 |
| 4DVarNet `aug0` (mean of 5 seeds) | 0.401 | 0.229 |
| EnKF | 0.368 | 0.215 |
| DINCAE | 0.329 | **0.757** |

Off by 2x or more means something is broken — check the convention first.

## Per-method diagnostics

| Question | Script |
|---|---|
| DINCAE accuracy + sigma-hat calibration + variance retention | `methods.dincae.checks.evaluate` |
| Does full-field supervision fix DINCAE's obstacle-region behaviour? | `methods.dincae.checks.measure_obstacle_region` |
| Is a method's uncertainty better than "just guess the average error"? (CRPS/NLL/spread vs a constant-sigma null model) | `methods.dincae.checks.eval_uncertainty_dincae`, `methods.enkf.checks.eval_uncertainty_enkf`, `methods.varnet.checks.eval_uncertainty` |
| Does the EnKF's ensemble sustain spread, or collapse? | `methods.enkf.checks.diag_enkf_spread_growth` |
| 4DVarNet variational solver sanity (gradients, convergence) | `methods.varnet.checks.check_variational_solver` |
| Senseiver pipeline trace (tensor shapes end to end) | `methods.senseiver.checks.trace_pipeline` |
| Is `enkf_opt/` still bit-identical to the vendored `enkf_lab/`? | `methods.enkf.checks.verify_enkf_opt` |
| What does a reconstructed field actually look like? | `compare.plot_reconstruction` |

All take `--help`; most default to `--split test` on the 7 held-out days.

## Reconstructed fields as pictures

Density as a Blues heatmap, velocity as black heading arrows.

```bash
# One busy frame, all 4 methods side by side
sbatch sbatch/submit_plot_reconstruction.sbatch --day atc-20130811
sbatch sbatch/submit_plot_reconstruction.sbatch --methods dincae,senseiver --day atc-20130818

# N consecutive frames, one PNG per frame (for temporal consistency, or a video)
sbatch sbatch/submit_plot_reconstruction_sequence.sbatch --day atc-20130811 --n 5000
```

Single frames land in `compare/results/reconstruction_<day>_<methods>.png`;
sequences in `compare/results/seq_ppt_<day>/` (gitignored — 5000 frames is
~600 MB, cheap to regenerate). To watch a sequence rather than scroll it:

```bash
ffmpeg -framerate 10 -i compare/results/seq_ppt_atc-20130811/frame_%05d.png \
       -pix_fmt yuv420p compare/results/seq_ppt_atc-20130811.mp4
```

---

# Data, and the minimal package to transfer

## Where the data lives

Three pipeline stages in two locations, and **only the last is read at run
time**:

| Stage | Location | Size | Read at run time? |
|---|---|---|---|
| (1) raw ATC CSVs, 92 recording days | `/scratch/work/zhangx29/ATC/` | 225 GB | no |
| (2) trajectory H5, Sundays only, 46 days | `/scratch/work/zhangx29/data/ATC/Sundays/` | 36 GB | no — naming manifest only |
| (3) `grid_cache`, 4-channel 36x12 fields, 46 days | `/scratch/work/zhangx29/data/grid_cache/` | 3.0 GB | **yes** |

Stage (2) looks load-bearing because the split lists
(`data/sunday_atc_{train,valid,test}.lst`) name `ATC/Sundays/atc-YYYYMMDD.h5` —
but `observation_model.split_files()` only takes the filename stem and opens the
corresponding `grid_cache` file instead. Those 36 GB are never opened.

Two external paths are configured in `crowdcore/config.yaml`, and are the only
ones to repoint on another machine:

- `data.root` → `/scratch/work/zhangx29/data` (stages 2 and 3, plus the split lists)
- `navigation.map_dir` → the real ATC map (`localization_grid.pgm` + `.yaml`,
  3.3 MB), which lives **outside both this repo and the data root**, under
  `project_analysis/.../robot_exploration/atc_map/`. This is the easiest
  dependency to miss.

Rebuild stages 1 → 2 → 3 only if the raw CSVs or the grid resolution changed;
see `crowdcore/data/DOC_data_pipeline.md`.

## What is not in the repo

Code, config, metrics and figures are tracked. About 21 GB of generated data is
not (sizes as of 2026-09-08):

- `methods/dincae/cache/` grid encoding cache (~13 GB)
- `*.npz` EnKF exported fields (`methods/enkf/check_outputs/`, ~6.3 GB)
- `*.pt` model weights (`methods/*/runs/`, ~1.5 GB)
- `**/seq_ppt_*/` per-frame sequence figures

**The gridded ATC data itself is also not in git** — `git ls-files | grep '\.h5$'`
returns zero. It never has been, most likely because the ATC dataset carries its
own redistribution terms, not only because of size. So running inference (as
opposed to reading the already-computed results in `check_outputs/`) needs the
code **and** a copy of the gridded data **and** the checkpoints: three separate
things, not one repo clone.

## The minimal package: ~1.4 GB, not 225 GB

| Item | Path | Size |
|---|---|---|
| gridded fields, 7 test days only | `data/grid_cache/atc-{20130811,20130818,20130825,20130901,20130915,20130922,20130929}_corridor_1.0s.h5` | **331 MB** (all 46 days would be 3.0 GB) |
| split lists | `data/sunday_atc_{train,valid,test}.lst` | a few KB |
| the real ATC map | `.../robot_exploration/atc_map/` (`localization_grid.pgm` + `.yaml`) | 3.3 MB |
| DINCAE normalisation stats | `methods/dincae/artifacts/state_stats.npz` | 17 KB (already in git) |
| trained weights | `methods/*/runs/<run>/` | ~1 GB |
| | **total** | **~1.4 GB** |

**DINCAE needs that one extra small file that weights alone do not carry.**
`state_stats.npz` holds the per-cell mean field and per-channel residual std,
computed once from the training data and read separately from the checkpoint at
inference time by `state.StateStats`. It used to be gitignored, caught by the
blanket `*.npz` rule meant for multi-GB files; it is now tracked via an explicit
negation, so a fresh clone has it. 4DVarNet and Senseiver do not have this
problem: Senseiver's `in_mean`/`in_std` are model buffers saved in the same
`state_dict`, and 4DVarNet works in raw physical units.

(`paths.REFERENCE_IMPL` → `/scratch/work/zhangx29/Partial_observation` is a third
external path but **not** a runtime dependency: the EnKF's byte-identical vendor
copy already lives in-repo at `methods/enkf/enkf_lab/`. The original is
referenced only for provenance checks and speed benchmarking.)

---

# Troubleshooting

- **`ModuleNotFoundError` on the first import** → you forgot
  `source sbatch/_env.sh` in this shell.
- **A number matches a *different* method's expected result** → you ran
  `python3 methods/<x>/train.py` instead of `python3 -m methods.<x>.train`. The
  three methods each ship their own `losses.py`/`dataset.py`/`model.py`; a bare
  path puts that directory first on `sys.path` and silently imports the wrong
  one. This is the single most common way to get a confusing result here.
- **Numbers differ from another run of "the same" model** → check you are
  comparing the same scoring convention, and that you are not comparing a
  16-checkpoint DINCAE average against the single published checkpoint.
- **`RuntimeError: CUDA not available` / training refuses to start** → you are on
  the login node. Use `sbatch` or `srun -p gpu-debug --gres=gpu:1`. `--allow-cpu`
  exists but is off by default on purpose: a silent CPU fallback under a GPU
  allocation costs ~30x the wall-clock.
- **`enkf_opt/` edits do nothing** → you may be editing `enkf_lab/`, the
  read-only (chmod 444) byte-identical vendor copy. Edit `enkf_opt/`, then re-run
  `methods.enkf.checks.verify_enkf_opt` to confirm it still matches.
- **Still stuck** → every script's `--help` and module docstring documents its
  own design decisions in detail; they are written to be read, not skimmed.
  `methods/*/README.md` covers each method's deviations from its reference
  implementation, and `methods/varnet/SUPERSEDED.md` records arms and
  conclusions that no longer hold.
