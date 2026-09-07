# How to use this repository

This is a practical, step-by-step companion to `README.md` (which explains the
*structure*). Read the "Scoring convention" section of `README.md` first — it
changes how every number in this repo should be read.

## 0. Environment

Everything runs on Aalto Triton (Slurm). One line sets up modules, `PYTHONPATH`,
and `PYTHONSAFEPATH`:

```bash
cd /scratch/work/zhangx29/Thesis_Project
source sbatch/_env.sh
```

Do this once per shell / at the top of every sbatch script. After sourcing it,
always invoke code as `python3 -m <package.module>`, never as a bare script
path — see README's "How to run" for why (three methods each have their own
`losses.py`/`dataset.py`/`model.py`, and running a bare path breaks the
isolation between them).

The login node has no GPU. Anything that touches `torch` with real data
(training, evaluation, most `checks/` scripts) must go through `sbatch` or
`srun -p gpu-debug --gres=gpu:1`. Dry runs / `--help` / pure-Python checks are
fine on the login node.

## 1. Data pipeline (already built — you normally skip this)

The processed grid cache already exists. If you need to rebuild it from raw
ATC CSVs, see `crowdcore/data/DOC_data_pipeline.md` for the three stages
(`csv_to_h5.py` → `h5_to_grid.py` → per-method encoding caches). Do not rerun
this unless the raw data or the grid resolution changed.

## 2. Training a method

Each method's `train.py` takes `--help` for the full list; the essentials:

```bash
# 4DVarNet — plain MSE (Eq.14, the paper's loss)
sbatch methods/varnet/sbatch/submit_bench4.sbatch --loss supervised --outdir runs/varnet_mse5_s0 --seed 0

# 4DVarNet — the two uncertainty designs (see README's uncertainty table)
python3 -m methods.varnet.train --loss nll --augmented-var --outdir runs/varnet_aug0_s0   # aug0: sigma inside G(x)
python3 -m methods.varnet.train --loss nll --var-h-only    --outdir runs/varnet_vsb0_s0   # vsb0: read-out head only

# DINCAE
sbatch methods/dincae/sbatch/submit_train.sbatch --out runs/dincae_full
sbatch methods/dincae/sbatch/submit_train.sbatch --out runs/dincae_ff --full-field-loss   # obstacle-region ablation

# Senseiver
sbatch methods/senseiver/sbatch/submit_train.sbatch --out runs/senseiver_A

# EnKF has no training step — it's a filter, not a learned model. See
# methods/enkf/checks/run_enkf_baseline.py to produce its estimated fields.
```

Checkpoints land under each method's `runs/<name>/` (`.pt` files, gitignored —
see README's "What is not in the repo"). Training logs go to
`runs/slurm_<jobname>_<jobid>.out`.

## 3. Reproducing the headline comparison

This is the one command that regenerates the main results:

```bash
source sbatch/_env.sh
sbatch sbatch/submit_compare5.sbatch          # or run interactively on a GPU node
python3 -m compare.plot_compare5              # the main figure, from compare5's json
```

`compare/compare5.py` scores four methods under all four cell-scoping
conventions (`defined` / `defined_full` / `allcells` / `full` — see its module
docstring) in one pass, including the 4DVarNet uncertainty arms
(`--arms aug0=varnet_aug0_s{seed},vsb0=varnet_vsb0_s{seed},...`) with
cross-seed mean±std. Its output json is what every comparison figure in
`compare/` and `slides/` is built from — regenerate it before regenerating any
figure, or the two will silently disagree.

## 4. Per-method evaluation and diagnostics

Each method has its own `checks/` with narrower, single-purpose scripts. A
few worth knowing about:

| Question | Script |
|---|---|
| DINCAE accuracy + σ̂ calibration + variance retention | `methods.dincae.checks.evaluate` |
| Does full-field supervision fix DINCAE's obstacle-region behaviour? | `methods.dincae.checks.measure_obstacle_region` |
| DINCAE/EnKF/4DVarNet uncertainty vs a constant-σ null model (CRPS/NLL/spread) | `methods.dincae.checks.eval_uncertainty_dincae`, `methods.enkf.checks.eval_uncertainty_enkf`, `methods.varnet.checks.eval_uncertainty` |
| Does the EnKF's ensemble actually sustain spread? | `methods.enkf.checks.diag_enkf_spread_growth` |
| 4DVarNet variational solver sanity (gradients, convergence) | `methods.varnet.checks.check_variational_solver` |
| Senseiver pipeline trace (tensor shapes end to end) | `methods.senseiver.checks.trace_pipeline` |
| Is `enkf_opt/` still bit-identical to the vendored `enkf_lab/`? | `methods.enkf.checks.verify_enkf_opt` |

All of these take `--help`; most default to `--split test` on the 7 held-out
days.

## 5. Where results end up

- `methods/*/check_outputs/eval/*.json` — numeric results, tracked in git (small)
- `methods/*/check_outputs/**/*.png` — figures, tracked in git
- `slides/` — meeting decks built from the above (`slides/build_slides.py`,
  `slides/build_meeting4_deck.py`); regenerate a deck only after its source
  jsons/figures change, and always via `python3 -m slides.<script>`
- `runs/`, `check_outputs/eval/seq_ppt_*/`, `*.npz` EnKF fields — gitignored,
  regenerate locally (see README's "What is not in the repo" for exact list
  and sizes)

## 6. Gotchas

- **`python3 -m`, always.** A bare `python3 methods/varnet/train.py` puts that
  directory on `sys.path[0]` and silently imports the wrong `losses.py`.
- **`enkf_lab/` is read-only** (chmod 444, byte-identical vendor copy). Edit
  `enkf_opt/` instead and re-run `verify_enkf_opt.py` before trusting a change.
- **Any single number needs its convention stated.** "DINCAE's MSE is 0.33" is
  meaningless without saying `defined` or `allcells` — see README.
- **4DVarNet has a ~4e-4 reproducibility floor** (its inference pass
  backpropagates through autograd) — a 4th-decimal-place difference between
  two runs of the same config is noise, not a regression. See
  `refactor_baseline/README.md`.
- **`--allow-cpu` is off by default** in `train.py` scripts on purpose — a
  silent CPU fallback under a GPU allocation burns ~30x the wall-clock and
  usually times out before producing anything.
