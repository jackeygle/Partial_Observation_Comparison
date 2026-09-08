# How to use this repository

A practical, hands-on guide — written so someone who has never touched this
codebase can either (a) look at the results without running anything, or
(b) reproduce them from scratch, without needing to read the source first.
`README.md` covers the *structure*; this covers *what to actually type*.

## 0. What this project is, in one paragraph

Three robots move through a real train-station corridor (the ATC dataset,
Osaka) and each senses pedestrians in a small radius around itself — that is
the "partial observation." The task is to reconstruct the **full** crowd
density and velocity field (a 36×12 grid, updated every second) from those
sparse robot sightings. Four different methods attempt this reconstruction
(4DVarNet, DINCAE, Senseiver, an ensemble Kalman filter); three of them are
also scored on whether they know **how wrong** their own reconstruction is
(a "σ̂" or ensemble-spread uncertainty estimate), because a wrong-but-honest
answer is more useful downstream than a wrong-and-confident one.

## 1. If you just want to look at results (no cluster, no running anything)

Everything below is already computed and sitting in the git repo as small
JSON/PNG files — clone the repo and open these directly:

| What | Where |
|---|---|
| The main 4-way accuracy comparison figure | `compare/results/compare5.png` |
| The main 4-way comparison, raw numbers, all 4 scoring conventions | `compare/results/compare5_final.json` |
| The 2026-09-07 meeting deck (6 slides, the current headline story) | `slides/meeting4_deck.pdf`, or `slides/meeting4_deck_notes.md` for the speaker notes |
| DINCAE's obstacle-region finding (this week's action item) | `methods/dincae/check_outputs/eval/obstacle_region.json`, written up in `MODELS_FOR_ADVISOR.md` |
| Which checkpoint is which, and what it corresponds to | `MODELS_FOR_ADVISOR.md` |
| Every other figure (architecture diagrams, training curves, uncertainty calibration, speed benchmarks) | `methods/*/check_outputs/**/*.png` (61 figures total, 55 of them under `methods/varnet/check_outputs/eval/`; filenames are mostly self-describing — `unc_*` = uncertainty, `mt_*` = the 4DVarNet-uncertainty meeting figures, `fw_*`/`arch_*` = architecture diagrams, `speed_*` = latency benchmarks) |
| A reconstructed field as an actual picture, all 4 methods side by side | `compare/results/reconstruction_<day>_<methods>.png` — see §12 to generate one for a different day/method subset |

No GPU, no Slurm, no environment setup needed for any of this — it's static
output already in git.

## 2. Directory map

```
crowdcore/                shared by all methods (not method-specific)
  config.py + config.yaml   single source of truth for every parameter
                            (grid resolution, number of robots, sensing
                            radius, noise level, time-window length, ...)
  navigation.py             the real ATC map -> which cells are walkable
  observation_model.py      simulates the 3 robots sensing the crowd
  data/                     raw CSV -> gridded .h5 (already done; see §7)

methods/                  one subdirectory per method, fully independent
  varnet/                   4DVarNet (plain-MSE, plus 2 uncertainty designs)
  dincae/                   DINCAE
  senseiver/                Senseiver
  enkf/                     the ensemble Kalman filter
  each has:
    train.py                (not applicable to enkf/, which is not learned)
    checks/                 evaluation and diagnostic scripts
    runs/                   trained checkpoints (gitignored, see MODELS_FOR_ADVISOR.md)
    check_outputs/          figures and result JSON (tracked in git)
    sbatch/                 Slurm job scripts

compare/                  the ONLY place allowed to import more than one
                          method at once -- cross-method comparison + plots
slides/                   meeting decks (do not treat as documentation --
                          they simplify for a live audience)
sbatch/_env.sh            the single environment entry point, source this first
```

## 3. Glossary: what the run names mean

Checkpoint directories under `runs/` use short codenames. Only these matter
for reproducing the headline numbers — everything else under `runs/` (`a2`,
`a4`, `b0`, `ml5`, `nf5`, `vrb0`, `vrb1`, ...) is an exploratory ablation from
along the way and is **not** part of any reported comparison; ignore those
unless you're specifically curious about a dead end.

| Codename | Method | Means |
|---|---|---|
| `mse5` | 4DVarNet | plain MSE loss (Eq.14 of the 4DVarNet paper), 5 random seeds, no uncertainty output |
| `aug0` | 4DVarNet | uncertainty σ injected **inside** the prior operator `G(x)`, trained jointly with NLL loss — our design, the one that works |
| `vsb0` | 4DVarNet | uncertainty σ from a **separate read-out head**, NLL loss — our design, kept as the negative control (bolt-on) |
| `dincae_full` | DINCAE | the baseline: information-form supervision (only cells with a defined channel value are in the loss) |
| `dincae_ff` | DINCAE | `--full-field-loss` ablation: **every** cell is in the loss, including the obstacle region (target = physical 0 there) |
| `senseiver_A` | Senseiver | the one trained model (no ablations) |
| `enkf_k1_full` | EnKF | exported ensemble fields at `obs_every_k=1` (observe every frame), full 7-day test split |

`s0`..`s4` suffixes are random seeds (weight init + data ordering); the
headline table reports mean±std across all 5.

## 4. Environment setup

```bash
cd /scratch/work/zhangx29/Thesis_Project
source sbatch/_env.sh
```

This does `module load` + sets `PYTHONPATH` + `PYTHONSAFEPATH=1`. Do it once
per shell, and always run code as `python3 -m <package.module>` afterwards —
**never** a bare script path (`python3 methods/varnet/train.py` silently
imports the wrong `losses.py`; see §9 troubleshooting).

The login node has **no GPU**. `--help` and pure-Python checks are fine
there; anything that calls `torch` on real data needs `sbatch` or:

```bash
srun -p gpu-debug --gres=gpu:1 -t 00:15:00 --pty bash
```

## 5. Quick sanity checks before committing to a long run

**Fastest possible check (seconds, no GPU, no real data, login node is fine)**
— pure synthetic tensors through the 4DVarNet solver, just confirms the code
imports and both uncertainty designs produce finite gradients:

```bash
source sbatch/_env.sh
python3 -m methods.varnet.scratch.smoke_aug
```

**With real data** (needs GPU; each takes a few minutes):

```bash
source sbatch/_env.sh
srun -p gpu-debug --gres=gpu:1 -t 00:15:00 bash -c '
  cd methods/varnet    && python3 -u -m methods.varnet.train --steps 5 --allow-cpu 2>&1 | tail -5
  cd ../dincae          && python3 -u -m methods.dincae.train --cache-dir "" --days 2 --epochs 2 \
                           --max-frames 3000 --amp --out runs/smoke_check 2>&1 | tail -5
  cd ../senseiver        && python3 -u -m methods.senseiver.train --days 1 --frames 200 --epochs 1 \
                           --allow-cpu --out runs/smoke_check 2>&1 | tail -5
'
```

(`--cache-dir ""` for DINCAE matters: without it, a smoke run tries to read/write
the same 13GB cache the real training run uses.) If all three print a loss
value and exit 0, the data pipeline and imports are fine and you can move on
to a real run.

## 6. Training a method for real

4DVarNet's `*_chain.sbatch` scripts **self-resubmit**: one `sbatch` call runs
for its `--time` budget, then submits its own successor (up to `MAX_GEN=12`
generations) so a ~150-epoch run survives past any single job's time limit.
Each call trains **one seed**; loop over seeds yourself:

```bash
# 4DVarNet -- plain MSE (headline "4DVarNet" row), 5 seeds
for s in 0 1 2 3 4; do
  sbatch --job-name=varnet_mse5_s$s methods/varnet/sbatch/submit_mse5_chain.sbatch $s 0
done

# 4DVarNet -- uncertainty design 1: sigma inside G(x) (aug0), 5 seeds
for s in 0 1 2 3 4; do
  sbatch --job-name=varnet_aug0_s$s methods/varnet/sbatch/submit_aug_chain.sbatch $s 0
done

# 4DVarNet -- uncertainty design 2: read-out head only (vsb0)
# no dedicated chain script; run train.py directly with enough --time,
# or copy submit_aug_chain.sbatch and swap in --var-h-only (drop --augmented-var)
python3 -m methods.varnet.train --loss nll --var-h-only --outdir runs/varnet_vsb0_s0

# DINCAE -- baseline and the obstacle-region ablation (~16-27 GPU-hours each,
# 190/140 epochs respectively; submit_train.sbatch's --time=1-00:00:00 covers it)
sbatch methods/dincae/sbatch/submit_train.sbatch --out runs/dincae_full
sbatch methods/dincae/sbatch/submit_train.sbatch --out runs/dincae_ff --full-field-loss

# Senseiver
sbatch methods/senseiver/sbatch/submit_train.sbatch --out runs/senseiver_A           # up to 1 day

# EnKF has no training step -- it's a filter, not a learned model. It DOES
# need "exporting": simulate the robots' observations for each day, then run
# the filter over them. This is CPU-only (no GPU helps a Kalman filter) and
# genuinely slow -- one real day of data took ~53 CPU-hours in the run that
# produced runs/enkf_k1_full/. Submit one Slurm job PER DAY (--only) so the
# 7 days run in parallel instead of one 15-day serial job:
python3 -m methods.enkf.checks.export_obs_for_enkf --outdir check_outputs/enkf_k1_full --obs-every-k 1 --frames 0
for day in atc-20130811 atc-20130818 atc-20130825 atc-20130901 atc-20130915 atc-20130922 atc-20130929; do
  sbatch --wrap="source sbatch/_env.sh && cd methods/enkf && python3 -u -m methods.enkf.checks.run_enkf_baseline \
    --dir check_outputs/enkf_k1_full --only $day --ensemble 100 --radius 7 --frames 0" \
    --time=60:00:00 --cpus-per-task=4 --mem=64G --job-name=enkf_$day
done
```

`run_enkf_baseline.py --help` does **not** show `--dir`/`--frames`/`--ensemble`/
`--radius`/`--only` — it parses vendor config args first and prints that
parser's help before reaching its own `argparse.ArgumentParser()` further
down (`checks/run_enkf_baseline.py:123-133`). The flags above are real; they
just don't show up that way.

Checkpoints land in each method's `runs/<name>/` (gitignored — see
`MODELS_FOR_ADVISOR.md` if you need the actual weight files rather than
retraining). Slurm logs land alongside as `runs/slurm_<job>_<id>.out` and
**are** tracked in git, so you can always check what a past run actually did
without re-running it.

## 7. Data pipeline (already built — you should not need this)

The gridded `.h5` cache already exists on disk. Only rebuild it if the raw
ATC CSVs or the grid resolution changed — see
`crowdcore/data/DOC_data_pipeline.md` for the three stages
(`csv_to_h5.py` → `h5_to_grid.py` → per-method encoding cache).

## 8. Reproducing the headline comparison, with a way to check your answer

```bash
source sbatch/_env.sh
sbatch sbatch/submit_compare5.sbatch      # ~2h, needs every checkpoint in §6 trained already
python3 -m compare.plot_compare5          # regenerates compare/results/compare5.png
```

**Expected output**, pooled RMSE (from the current `compare5_final.json` —
your rerun should land within ~0.01 of these, see the reproducibility-floor
note in §9):

| Method | `defined` convention | `allcells` convention |
|---|---|---|
| Senseiver | 0.344 | 0.169 |
| 4DVarNet MSE (mean of 5 seeds) | 0.352 | 0.186 |
| 4DVarNet NLL/vsb0 (mean of 5 seeds) | 0.388 | 0.211 |
| 4DVarNet AUG/aug0 (mean of 5 seeds) | 0.401 | 0.229 |
| EnKF | 0.368 | 0.215 |
| DINCAE | 0.329 | **0.757** |

If your numbers are wildly different (not ~0.01, but off by 2x or more),
something is broken — check §9 first, in particular the convention pitfall.

## 9. The scoring-convention pitfall — a worked example

Look at DINCAE's own two numbers in the table above: **0.329** vs **0.757** —
more than double, same model, same predictions, same day. That's not a bug,
it's two different definitions of "which cells count":

- `defined`: only cells where that physical quantity exists (velocity needs
  density > 0 there) — DINCAE was trained exactly this way, so it wins.
- `allcells`: every cell, including the ~88% of velocity cells that are
  empty (no people, hence velocity ≡ 0) — DINCAE was never trained to
  predict 0 there, so it loses badly on this convention. (This is the exact
  mechanism `measure_obstacle_region.py` isolates for the *obstacle* subset
  specifically — see `MODELS_FOR_ADVISOR.md`.)

**So: any single number you see quoted without saying which convention it
uses should be treated as incomplete**, not wrong exactly, but not
comparable to a number computed the other way. `compare5.py` always reports
both side by side for this reason.

Also: 4DVarNet's own numbers carry a **~4e-4 reproducibility floor** — its
inference pass itself uses autograd, so re-running the identical config
twice differs in the 4th decimal place. That is noise, not a regression; see
`refactor_baseline/README.md`.

## 10. Per-method diagnostics — which script answers which question

| Question | Script |
|---|---|
| DINCAE accuracy + σ̂ calibration + variance retention | `methods.dincae.checks.evaluate` |
| Does full-field supervision fix DINCAE's obstacle-region behaviour? | `methods.dincae.checks.measure_obstacle_region` |
| Is a method's uncertainty better than "just guess the average error" (CRPS/NLL/spread vs a constant-σ null model)? | `methods.dincae.checks.eval_uncertainty_dincae`, `methods.enkf.checks.eval_uncertainty_enkf`, `methods.varnet.checks.eval_uncertainty` |
| Does the EnKF's ensemble actually sustain spread, or collapse? | `methods.enkf.checks.diag_enkf_spread_growth` |
| 4DVarNet variational solver sanity (gradients, convergence) | `methods.varnet.checks.check_variational_solver` |
| Senseiver pipeline trace (tensor shapes end to end) | `methods.senseiver.checks.trace_pipeline` |
| Is `enkf_opt/` still bit-identical to the vendored `enkf_lab/`? | `methods.enkf.checks.verify_enkf_opt` |
| What does a reconstructed field actually look like (not just its error number)? | `compare.plot_reconstruction` (all 4 methods) — see §12 |

All take `--help`; most default to `--split test` on the 7 held-out days
(`atc-20130811, atc-20130818, atc-20130825, atc-20130901, atc-20130915,
atc-20130922, atc-20130929` — never used in training or checkpoint selection).

## 11. Troubleshooting

- **`ModuleNotFoundError` on the first import** → you forgot
  `source sbatch/_env.sh` in this shell.
- **Numbers come out wildly different from another run of "the same" model**
  → check you're comparing the same convention (§9), and that you're not
  accidentally running a bare script path (see next point).
- **A number matches a *different* method's expected result** → you ran
  `python3 methods/<x>/train.py` instead of `python3 -m methods.<x>.train`.
  The three methods each ship their own `losses.py`/`dataset.py`/`model.py`;
  a bare path puts that directory first on `sys.path` and silently imports
  the wrong one. This is the single most common way to get a confusing
  result in this repo.
- **`RuntimeError: CUDA not available` / training refuses to start** →
  you're on the login node. Use `sbatch` or `srun -p gpu-debug --gres=gpu:1`.
  `--allow-cpu` exists but is off by default on purpose — a silent CPU
  fallback under a GPU allocation costs ~30x the wall-clock.
- **`enkf_opt/` edits don't seem to do anything** → you may be editing
  `enkf_lab/` instead, which is a read-only (chmod 444) byte-identical
  vendor copy. Edit `enkf_opt/`, then re-run
  `methods.enkf.checks.verify_enkf_opt` to confirm it still matches.
- **Still confused** → `README.md` has the structural overview,
  `MODELS_FOR_ADVISOR.md` has exact checkpoint-to-result mappings, and every
  script's own `--help` / module docstring documents its specific design
  decisions in detail (they're written to be read, not just skimmed).

## 12. Looking at a reconstructed field as a picture, not just a number

Two scripts, both save a PNG with density as a Blues heatmap and velocity as
black heading arrows:

```bash
# All 4 methods side by side on one day (auto-picks a busy frame)
source sbatch/_env.sh
sbatch sbatch/submit_plot_reconstruction.sbatch --day atc-20130811
# or just a subset:
sbatch sbatch/submit_plot_reconstruction.sbatch --methods dincae,senseiver --day atc-20130818

# 4DVarNet + EnKF specifically, in the EnKF project's own plotting style
# (older, narrower script -- prefer submit_plot_reconstruction.sbatch above for anything new)
srun -p gpu-debug --gres=gpu:1 -t 00:10:00 bash -c '
  cd methods/varnet && python3 -u -m compare.plot_reconstruction_enkf \
    --ckpt runs/varnet_mse5_s0/varnet_best.pt --day atc-20130811
'
```

Output lands at `compare/results/reconstruction_<day>_<methods>.png` (an
example is already committed: `reconstruction_atc-20130811_dincae-senseiver-varnet-enkf.png`).
Panels: true state | partial obs (illustrative — see the script's docstring
for why it isn't per-method-exact) | one panel per requested method | EnKF
spread (only if `enkf` is in `--methods`).

For a *sequence* of consecutive frames (to eyeball temporal consistency, or
stitch into a GIF) rather than one snapshot, see
`methods.varnet.checks.plot_reconstruction_sequence` in §6 above — that one
is 4DVarNet(+EnKF)-only, there is no per-frame sequence version for
DINCAE/Senseiver yet.
