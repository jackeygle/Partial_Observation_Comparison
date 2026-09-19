# Partial Observation Comparison

**The problem.** Three robots move through a real shopping-centre corridor (the
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

## Contents

| Section | For |
|---|---|
| [The four methods](#the-four-methods-plus-two-uncertainty-designs-of-our-own) · [Scoring scope](#scoring-scope-walkable-cells) · [The dataset and how it is split](#the-dataset-and-how-it-is-split) · [Uncertainty: how it is scored](#uncertainty-how-it-is-scored) | understanding what is here |
| [Looking at results without running anything](#looking-at-results-without-running-anything) | a quick look, no cluster needed |
| [Which file backs which published number](#which-file-backs-which-published-number) | reproducing or citing a specific number |
| [Running things](#running-things) (environment, layout, training, evaluation, figures) | doing the work |
| [Data, and the minimal package to transfer](#data-and-the-minimal-package-to-transfer) | moving this to another machine or person |

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
| `methods/dincae/` | DINCAE, trained with full-field supervision (single checkpoint, see below) | sigma-hat (the network's own variance output) |
| `methods/senseiver/` | Senseiver | none |
| `methods/enkf/` | localised EnKF (+ PedPred3 forward model) | ensemble spread |

How good each σ̂ is, and what asking for one costs in reconstruction
accuracy, is in [Uncertainty: how it is scored](#uncertainty-how-it-is-scored) —
kept in one place so two copies of the numbers cannot drift apart. DINCAE and
the EnKF have uncertainty *built into* the method (a variance output, ensemble
spread) rather than as a separate design.

**Every row of the accuracy table is one model** — no ensembling, no seed
averaging, no checkpoint averaging, because none of the papers reports one — and
every model is chosen on the **validation** split: DINCAE's and Senseiver's
checkpoint, and for 4DVarNet both each seed's epoch
(`methods/varnet/checks/select_checkpoint.py`) and which of its five MSE seeds
represents it (the lowest validation error, recorded by `compare5` as
`single_model`). The mean of five 4DVarNet seeds next to the other methods'
single model would give 4DVarNet five times the training.
DINCAE's reference implementation *does* average the outputs of checkpoints
saved every 10 epochs, and `methods/dincae/checks/evaluate.py` still supports
it behind `--average-checkpoints`, but the reported number comes from **one**
checkpoint, `methods/dincae/runs/dincae_ff/ckpt_00060.pt`, chosen on the validation
split by `methods/dincae/checks/select_checkpoint.py`.

`methods/enkf/enkf_lab/` is a **byte-identical, read-only copy** of
the original `Partial_observation` EnKF code, trained surrogate weights included, its
files deliberately chmod 444.
To change anything, edit `enkf_opt/` instead, and it must pass the bit-identical
comparison in `methods/enkf/checks/verify_enkf_opt.py` (`np.array_equal`, not
`isclose`).

## Scoring scope: walkable cells

Every reported number is scored inside the **walkable** region: 290 of the 432 grid cells.
The other 142 (32.9%) are map obstacles — walls and the stalls lining the corridor — which
nobody can stand in and no method is asked to reconstruct. All four channels, every frame of
every test day. Each result is reported on two cell sets:

- **blind walkable cells** — the cells no robot sees in that frame (about 44% of all
  walkable cell-frames). This is the reconstruction task proper: there is no observation there.
- **all walkable cells** — observed cells included, so a method must also stay faithful to
  what the robots do see.

Empty walkable cells are included in both: their true density and velocity are 0, and
getting "nobody is here" right is part of the task. Squared errors are pooled over all
scored cells before the square root, and every method's predictions are clipped to the
same physical bounds.

The result files name the two differently: `compare5_final.json` calls them `walkable`
(blind) and `walkable_full` (all); the uncertainty JSONs call them `walkable_blind` and
`walkable`. `compare/compare5.py` also computes other cell sets (obstacle cells included,
only cells where a velocity is defined) and keeps them in `compare5_final.json` for
reference; none of them is reported.

Also: 4DVarNet's numbers carry a **~4e-4 reproducibility floor**. Its inference
pass itself backpropagates through autograd, and conv backward's reduction uses
atomicAdd, whose accumulation order differs run to run — so re-running an
identical config differs in the 4th decimal place. That is noise, not a
regression. Senseiver, DINCAE and the EnKF are bit-reproducible. See
`refactor_baseline/README.md`.

## The dataset and how it is split

The ATC pedestrian tracking dataset (Osaka) records a shopping-centre corridor
on alternating Sundays and Wednesdays across roughly a year, 92 recording days
in total. **This project uses the 46 Sundays only** — one weekday pattern, so
crowd behaviour is comparable across days and a model is not asked to
generalise between a Sunday and a Wednesday. The split lists that pin this down
are `data/sunday_atc_{train,valid,test}.lst`.

The split is **chronological, never shuffled**: training is the earliest
stretch, validation the middle, test the latest. A random split would let a
model see the same afternoon from both sides of the boundary.

| Split | Days | Frames | Date range | Used for |
|---|---|---|---|---|
| train | 32 | 1,262,518 | 2012-10-28 → 2013-06-09 | fitting weights |
| valid | 7 | 269,743 | 2013-06-16 → 2013-07-28 | choosing the checkpoint |
| test | 7 | 277,543 | 2013-08-11 → 2013-09-29 | every reported number |
| | **46** | **1,809,804** | | |

The seven test days, never seen during training or checkpoint selection:

```
atc-20130811  atc-20130818  atc-20130825  atc-20130901  atc-20130915
atc-20130922  atc-20130929
```

The gap at 2013-09-08 is in the source data — ATC did not record that Sunday.

Each day is a full recording, roughly 38,000–43,000 frames at one frame per
second (about 11 hours). Nothing is subsampled for the reported numbers: every
frame of every test day is scored.

## Uncertainty: how it is scored

Three of the four methods also report how wrong they think they are. They are
compared here **as ensembles**, because that is how two of them produce σ̂: the
EnKF's σ̂ is the spread of its 100 members, and each 4DVarNet design is a 5-seed
deep ensemble whose σ̂ combines each member's own learned variance with the
members' disagreement (Lakshminarayanan et al. 2017, Sec. 2.4). DINCAE is the
exception and is marked as such: one model that outputs σ̂ directly
(a mean and a variance for every cell), kept as the non-ensemble reference.

Judging a σ̂ needs a reference, because it can look plausible while carrying no
information: **the null model replaces every cell's σ̂ with one constant, that
split's own RMSE**. It knows how big the error is on average and nothing about
*where* — and it scores a perfect 1.00 spread/skill for free. A useful σ̂ has to
beat it.

**Blind walkable cells** (no robot sees them in that frame), 7 test days:

| Method | RMSE | vs MSE ensemble | vs null | spread/skill | 90% cov. |
|---|---|---|---|---|---|
| 4DVarNet MSE, 5-seed ensemble (no σ̂) | 0.228 | — | — | — | — |
| 4DVarNet `vsb0`, 5-seed ensemble | 0.266 | +16.9% | +29.1% | 1.912 | 97.1% |
| 4DVarNet `aug0`, 5-seed ensemble | 0.290 | +27.5% | **−22.7%** | 0.755 | 94.8% |
| EnKF, 100 members | 0.263 | +15.7% | +26.0% | 0.010 | 1.6% |
| DINCAE ‡ (one model) | 0.227 | −0.4% | **−23.3%** | 0.640 | 94.3% |

**All walkable cells** (observed cells included), 7 test days:

| Method | RMSE | vs MSE ensemble | vs null | spread/skill | 90% cov. |
|---|---|---|---|---|---|
| 4DVarNet MSE, 5-seed ensemble (no σ̂) | 0.189 | — | — | — | — |
| 4DVarNet `vsb0`, 5-seed ensemble | 0.217 | +14.6% | +43.4% | 2.302 | 98.1% |
| 4DVarNet `aug0`, 5-seed ensemble | 0.230 | +21.5% | **−15.1%** | 0.956 | 95.9% |
| EnKF, 100 members | 0.275 | +45.4% | +26.4% | 0.009 | 1.3% |
| DINCAE ‡ (one model) | 0.182 | −3.7% | **−25.8%** | 0.591 | 94.1% |

Where each σ̂ comes from is in the [method table](#the-four-methods-plus-two-uncertainty-designs-of-our-own).

‡ **DINCAE is not an ensemble.** It is here as the reference for what one
model's own σ̂ achieves; its row is a different mechanism, not a like-for-like
comparison with the ensemble rows. Its σ̂ is scored in its normalised space (log1p on
density/var, then per-channel standardisation); every column except RMSE is a ratio or a
coverage, so the row still compares, and its RMSE comes from compare5 in data units.

**The MSE ensemble has no σ̂**; it is in the table for its RMSE. `vsb0` and
`aug0` are the same network trained with an NLL loss instead of MSE, so for them
*vs MSE ensemble* reads directly as what asking for a σ̂ costs in reconstruction
accuracy. For the EnKF and DINCAE the same column is a different method measured
against the same yardstick, not a cost of anything.

The RMSE column is compare5's RMSE on the same cells (the ensemble mean for the three
4DVarNet rows), so it agrees with the accuracy table. The uncertainty scripts also compute
an RMSE, on unclipped predictions; for `vsb0`/`aug0` it differs from compare5's by at most
1.3%.

**Reading the columns**

- **RMSE** — reconstruction error of the ensemble mean (DINCAE: its single prediction).
- **vs MSE ensemble** — RMSE relative to the 4DVarNet MSE ensemble's on the same cells.
  Positive means worse reconstruction.
- **vs null** — CRPS relative to the constant-σ null model on the same cells. Negative
  means the σ̂ carries spatial information the constant does not. CRPS (Continuous Ranked
  Probability Score) scores the whole predicted distribution, centre and spread together,
  and reduces to MAE as σ → 0; the absolute values are in the JSONs. NLL is also computed
  but not used: its `(x−μ)²/2σ²` term is unbounded as σ → 0, so the EnKF's collapsed
  ensemble produces an NLL of order 1e18, which cannot rank anything.
- **spread/skill** — mean predicted σ̂ over actual RMSE. 1.0 is calibrated,
  below 1 is overconfident, above 1 underconfident.
- **90% cov.** — the fraction of cells whose truth falls inside the nominal 90% interval.
  94.8% is slightly too wide; 1.6% means the interval has collapsed.

The EnKF's row is a collapse, not a miscalibration: its ensemble spread settles
around 0.0025 while its actual error is 0.263 on blind cells, a ratio of 0.01. The forecast
model damps member disagreement about 65% per step regardless of injected
noise — measured directly in
`methods/varnet/check_outputs/eval/enkf_spread_growth.json`, verdict
`contractive`. Senseiver is absent from these tables because it is a
deterministic decoder and produces no σ̂ at all.

**Reproduce**

```bash
sbatch methods/varnet/sbatch/submit_unc_aug.sbatch      # -> uncertainty_aug0.json
sbatch methods/varnet/sbatch/submit_unc_vsb0.sbatch     # -> uncertainty_vsb0.json
sbatch methods/dincae/sbatch/submit_unc_dincae.sbatch   # -> uncertainty_dincae.json
sbatch methods/enkf/sbatch/submit_unc_enkf.sbatch       # -> uncertainty_enkf_k1.json
```

The shared scoring code is `compare/score_uncertainty.py`; each method's wrapper
only supplies its own σ̂.

---

# Looking at results without running anything

Everything here is already computed and sitting in the repo as small JSON/PNG
files. No GPU, no Slurm, no environment setup.

| What | Where |
|---|---|
| The main accuracy comparison figure (per-channel, **MSE**) | `compare/results/compare5.png` |
| Raw numbers (`walkable` = blind walkable cells, `walkable_full` = all walkable cells; other cell sets are kept for reference) | `compare/results/compare5_final.json` |
| A reconstructed field as a picture, all 4 methods side by side | `compare/results/reconstruction_atc-20130811_dincae-senseiver-varnet-enkf.png` |
| Everything else | `methods/varnet/check_outputs/eval/unc_spread_decay.png` (why the EnKF's ensemble cannot hold a spread), `methods/varnet/check_outputs/eval/de_*.png` (the variational cost and solver drawn as diagrams), `methods/varnet/check_outputs/navigation/` (walkable mask and obstacles on the real map) |

# Which file backs which published number

"The DINCAE weights" is not a well-defined request: `runs/dincae_ff/` holds 12
checkpoints and which you pick changes the number. Same for a 4DVarNet run's
periodic `ckpt_<epoch>.pt` snapshots. This table is the authoritative mapping; everything else under
`runs/` is an intermediate or an exploratory ablation.

| Published as | Exact file(s) | Result JSON | Why this one |
|---|---|---|---|
| 4DVarNet MSE | `methods/varnet/runs/varnet_mse5_h96_s{0..4}/ckpt_<epoch>.pt`, epoch per run in that directory's `select_valid.json` | `compare/results/compare5_final.json`: `"4DVarNet MSE s<seed>"` for the accuracy table, seed = `single_model.MSE.seed`; `"4DVarNet MSE ens5"` for the uncertainty table's RMSE | chosen on the validation split; every member's file and epoch is in `compare5_final.json` → `protocol.checkpoints` |
| 4DVarNet `aug0` | `methods/varnet/runs/varnet_aug0_h96_s{0..4}/ckpt_<epoch>.pt`, per `select_valid.json` | σ̂: `methods/varnet/check_outputs/eval/uncertainty_aug0.json`; RMSE: `compare5_final.json` `"4DVarNet AUG ens5"` | same selection; the uncertainty job's log lists each member's file and epoch |
| 4DVarNet `vsb0` | `methods/varnet/runs/varnet_vsb0_h96_s{0..4}/ckpt_<epoch>.pt`, per `select_valid.json` | σ̂: `methods/varnet/check_outputs/eval/uncertainty_vsb0.json`; RMSE: `compare5_final.json` `"4DVarNet NLL ens5"` | same selection |
| DINCAE | `methods/dincae/runs/dincae_ff/ckpt_00060.pt` — **one file** | `methods/dincae/check_outputs/eval_ff_00060/dincae_metrics_test.json`, plus `methods/dincae/check_outputs/eval/uncertainty_dincae.json` | chosen on the validation split (`methods/dincae/check_outputs/eval/select_dincae_ff_valid.json`); recorded in `compare5_final.json` as `source: .../eval_ff_00060/...` |
| Senseiver | `methods/senseiver/runs/senseiver_A/best.pt` | folded into `compare5_final.json` (`"Senseiver"`) | only trained model; `best.pt`, not `last.pt` |
| EnKF | no weights — exported fields in `methods/enkf/check_outputs/enkf_k1_full/est_*.npz` | `compare5_final.json` (`"EnKF k1"`), `methods/enkf/check_outputs/eval/uncertainty_enkf_k1.json` | it is a filter, not a trained model |
| Learned-cov KF (the EnKF's σ̂ replaced) | `methods/enkf/runs/joint_j4_5f_r32_nofcst_s0/best.pt` over the frozen mean `methods/enkf/runs/surrogate_mean_s0/best.pt`, at **α = 2** | `methods/enkf/check_outputs/eval/uncertainty_j4nofcst_a2_test7.json` | chosen in `check_outputs/eval/controlled_grid_calibration_4096.json`, a seven-model grid in which every pair differs by one factor; α comes from a full-day sequential sweep, **not** from that grid — see the warning below |

Two traps, both of which silently produce a *different number* rather than an
error:

1. **Do not average DINCAE's checkpoints.** `methods/dincae/checks/evaluate.py` used to do
   this by default (it is the reference implementation's behaviour). It now
   refuses unless you pass `--average-checkpoints`, because the result is a
   different number from the reported one with nothing in the output to say which
   configuration produced it.
2. **Do not score a 4DVarNet run at `varnet_best.pt`.** Despite the name it is
   selected on the **training** split (`train.py`'s `--split` defaults to
   `train`), on the first `--n-eval` windows — the emptiest part of a recording
   day. It also lands runs at different points of the §3.4 iteration
   curriculum: on the earlier hidden=32 runs, `aug0`'s best fell at epoch 16-40
   with the solver at 5-10 of its 20 iterations while `mse5`'s fell at 78-143 at
   20, so the two differed in solver depth as well as in loss. Reported numbers
   use the checkpoint `methods/varnet/checks/select_checkpoint.py` chose on the
   **validation** split among the snapshots at 20 iterations
   (`train.py --ckpt-every`), recorded per run in `select_valid.json`.
   `compare5` and `eval_uncertainty` both resolve it through
   `methods/varnet/checks/model_io.py:reported_ckpt`, so the accuracy table and
   the uncertainty table score the same file. A run without `select_valid.json`
   falls back to `varnet_best.pt` and prints a `[ckpt] ... falling back` line —
   if you see that line, the number is not the reported one.

The EnKF's overconfidence is diagnosed separately in
`methods/varnet/check_outputs/eval/enkf_spread_growth.json` (via
`methods.enkf.checks.diag_enkf_spread_growth`) — verdict `"contractive"`: the
deterministic PedPred3 forecast model damps ensemble spread by ~65% per step
regardless of injected noise. Not a bug to fix; a property of the forward model
to report as-is. The learned-covariance row above is the answer to it: the same
filter with its ensemble spread replaced by a learned `B = U Uᵀ + diag(d)` goes
from spread/skill 0.010 and 1.6% coverage of a nominal 90% interval to 0.500 and
92.9%, and from 26.0% *worse* than the constant-σ null to 28.3% better.

**A third trap, specific to that row: α is convention-dependent.** The
seven-model grid scores 4096 held-out one-step pairs and picks α = 0.5; a
full-day sequential run picks α = 2, and at α = 2 every column improves at once
(CRPS −4.8%, spread/skill 0.249 → 0.500, coverage 83.3% → 92.9%). Sequential
error accumulates and one-step scoring cannot see it, so **a calibration constant
has to be fitted under the convention it will be reported in.** Reusing the
grid's α in a sequential table is not a small approximation; it is the difference
between a badly overconfident σ̂ and a calibrated one.

## Checkpoint fingerprints

`*.pt` is gitignored (the weights are ~30 of the raw 31 GB), so a checkout
reproduces the code and the result JSON but not the weights. These MD5s are the
link between the two: they identify which file produced each published number.
Regenerate with `md5sum <path>`.

| Published as | File | MD5 |
|---|---|---|
| 4DVarNet MSE s0–s4 | `varnet_mse5_h96_s{0..4}/ckpt_000{80,90,80,80,80}.pt` | `d2a03364…` `4d0a4643…` `8007503d…` `4c2245c2…` `b86e8625…` |
| 4DVarNet `aug0` s0–s4 | `varnet_aug0_h96_s{0..4}/ckpt_00{149,100,100,149,130}.pt` | `7402b99b…` `11a53aa0…` `bd126f33…` `8acaad80…` `c85d10f6…` |
| 4DVarNet `vsb0` s0–s4 | `varnet_vsb0_h96_s{0..4}/ckpt_00{080,090,149,140,140}.pt` | `8c29259f…` `3d6abcbb…` `817fd888…` `3558260d…` `110afbb8…` |
| DINCAE | `dincae/runs/dincae_ff/ckpt_00060.pt` | `15445c96a88515bf31417a1d59757fd8` |
| Senseiver | `senseiver/runs/senseiver_A/best.pt` | `ad0f002ef20efb34432ba800d6ddf118` |
| Senseiver k=16 (extension, not the main table) | `senseiver/runs/capacity/base32_k16_s123/best.pt` | `68f2f02f59957e193ffc398f3dd2a9cf` |
| EnKF surrogate (tracked in git, 14 MB) | `enkf/enkf_lab/apt-ibex_train_model_28D.pth` | `17562f3e2d969611fe2a2b0d40c51d42` |
| PedPred3 mean, 5→5 | `enkf/runs/pedpred3_5to5_s0/best.pt` | `077c26f0dce49fc530f064223a293d90` |
| Mean net under the learned covariance | `enkf/runs/surrogate_mean_s0/best.pt` | `a37ad0cd283446561cf1096a03d03dab` |
| **Learned-cov KF** | `enkf/runs/joint_j4_5f_r32_nofcst_s0/best.pt` | `9e7cb511ed2f3bce36ab8565e1c401a1` |

4DVarNet paths are relative to `methods/varnet/runs/`, the rest to `methods/`.
The per-seed epochs come from each run's `select_valid.json`; the truncated
hashes are the first 8 characters, enough to tell the five seeds apart.

---

# Running things

## Environment

```bash
cd path/to/this/repo           # the repo root
source sbatch/_env.sh          # module load + PYTHONPATH + PYTHONSAFEPATH=1
```

Once per shell, and afterwards always invoke code as `python3 -m <package.module>`.

Submit every `sbatch` command from the repo root, as written below: job scripts find the
repo through `$SLURM_SUBMIT_DIR`, and their `#SBATCH --output` paths are relative to it.

`_env.sh` loads Aalto Triton's `scicomp-pytorch-env/2026.1` module (Python 3.12). On any
other machine, install the same package versions instead: `pip install -r requirements.txt`.

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
  assets/atc_map/           that map: a ROS occupancy grid (.pgm image + .yaml)
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
sbatch/_env.sh            the single environment entry point
```

## Run names under `runs/`

Only these matter for the headline numbers. Everything else (`a2`, `a4`, `b0`,
`ml5`, `nf5`, `vrb0`, `vrb1`, ...) is an exploratory ablation from along the way
and is **not** part of any reported comparison.

| Codename | Method | Means |
|---|---|---|
| `mse5_h96` | 4DVarNet | plain MSE loss (Eq.14), 5 random seeds, no uncertainty output |
| `aug0_h96` | 4DVarNet | sigma **inside** the prior operator `G(x)`, NLL — ours |
| `vsb0_h96` | 4DVarNet | sigma from a **separate read-out head**, NLL — ours |
| `dincae_ff` | DINCAE | full-field supervision: every cell in the loss (`--full-field-loss`, the default) |
| `senseiver_A` | Senseiver | the one trained model |
| `enkf_k1_full` | EnKF | exported ensemble fields, `obs_every_k=1`, full 7-day test split |

`s0`..`s4` are random seeds (weight init + data ordering). The accuracy table
reports one of them (the validation-best MSE seed); the uncertainty table reports
each arm's 5-seed ensemble. `h96` is the prior width (hidden=96, kt=5, the
`crowdcore/config.yaml` default since 2026-09-09); the same names without it are
the earlier hidden=32 runs and are not reported. `obs_every_k=4` was studied
earlier and dropped entirely on 2026-09-08.

## Sanity checks before committing to a long run

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
# 4DVarNet -- plain MSE (the accuracy table's 4DVarNet row), 5 seeds
for s in 0 1 2 3 4; do
  sbatch --job-name=varnet_mse5_h96_s$s methods/varnet/sbatch/submit_mse5_chain.sbatch $s 0
done

# 4DVarNet -- uncertainty design 1: sigma inside G(x) (aug0), 5 seeds
for s in 0 1 2 3 4; do
  sbatch --job-name=varnet_aug0_h96_s$s methods/varnet/sbatch/submit_aug_chain.sbatch $s 0
done

# 4DVarNet -- uncertainty design 2: sigma from a read-out head on [h, x_hat] (vsb0), 5 seeds
for s in 0 1 2 3 4; do
  sbatch --job-name=varnet_vsb0_h96_s$s methods/varnet/sbatch/submit_vs_chain.sbatch $s 0.0
done

# then choose every run's reported epoch on the validation split (writes select_valid.json).
# ~11 min per run on one GPU; RUNS= splits the 15 across parallel jobs, see the script header
sbatch methods/varnet/sbatch/submit_select.sbatch

# DINCAE -- full-field supervision (~16-27 GPU-hours), then choose its checkpoint on validation
sbatch methods/dincae/sbatch/submit_train.sbatch --out runs/dincae_ff
python3 -m methods.dincae.checks.select_checkpoint     # defaults to runs/dincae_ff

# Senseiver (up to 1 day)
sbatch methods/senseiver/sbatch/submit_train.sbatch --out runs/senseiver_A
```

The EnKF has no training step — it is a filter, not a learned model. It does
need *exporting*: simulate the robots' observations per day, then run the filter
over them. CPU-only (no GPU helps a Kalman filter) and genuinely slow — each test day took
48–54 hours of wall-clock time on 4 cores in the run that produced `enkf_k1_full/`
(`timing_<day>.json` there). Submit one job
per day so the seven run in parallel:

```bash
python3 -m methods.enkf.checks.export_obs_for_enkf --outdir methods/enkf/check_outputs/enkf_k1_full --obs-every-k 1 --frames 0
for day in atc-20130811 atc-20130818 atc-20130825 atc-20130901 atc-20130915 atc-20130922 atc-20130929; do
  sbatch --wrap="source sbatch/_env.sh && cd methods/enkf && python3 -u -m methods.enkf.checks.run_enkf_baseline \
    --dir check_outputs/enkf_k1_full --only $day --ensemble 100 --radius 7 --frames 0" \
    --time=60:00:00 --cpus-per-task=4 --mem=64G --job-name=enkf_$day
done
```

`run_enkf_baseline.py --help` does **not** list `--dir`/`--frames`/`--ensemble`/
`--radius`/`--only`: it parses vendor config args first and prints that parser's
help before reaching its own `argparse.ArgumentParser()` further down
(`checks/run_enkf_baseline.py:120-131`). The flags above are real.

Checkpoints land in each method's `runs/<name>/` (gitignored). Slurm logs land
alongside as `runs/slurm_<job>_<id>.out` and **are** tracked, so you can always
check what a past run did without re-running it.

## Reproducing the headline comparison

```bash
# needs every run's select_valid.json (see Training above)
sbatch sbatch/submit_compare5.sbatch      # ~15-30 min on one GPU
python3 -m compare.plot_compare5          # regenerates compare/results/compare5.png
```

The defaults reproduce exactly the published JSON: all four methods; each
4DVarNet arm's five seeds at their validation-selected checkpoints, plus its
5-member ensemble (`ens5`, which the uncertainty table uses); DINCAE's single
validation-selected checkpoint; and `single_model`, the MSE seed the accuracy table
reports. A rerun writes `compare/results/compare5.json`; the published artefact
is `compare5_final.json`, kept as a separate file so a rerun cannot silently
overwrite the numbers this README quotes. Its `protocol.checkpoints` lists the
file and epoch every 4DVarNet member was scored at.

**Expected output**, pooled **RMSE**, one model per method (the rule is in
[The four methods](#the-four-methods-plus-two-uncertainty-designs-of-our-own)).
A rerun should land within ~0.01 of these (4DVarNet rows carry the
reproducibility floor described above). The figure `compare5.png` plots
per-channel **MSE** (top row blind walkable cells, bottom row all walkable
cells), so its totals are these numbers squared — DINCAE's 0.227 there appears
as 0.051.

| Method | blind walkable cells | all walkable cells |
|---|---|---|
| Senseiver | 0.222 | 0.156 |
| DINCAE | 0.227 | 0.182 |
| 4DVarNet MSE (one model: seed 3, validation-best) | 0.230 | 0.192 |
| EnKF | 0.263 | 0.275 |

The uncertainty designs `vsb0` and `aug0` are not in this table. What they cost
in accuracy is in the [uncertainty tables](#uncertainty-how-it-is-scored), next to
the MSE ensemble they are measured against.

Off by 2x or more means something is broken — check the scope first.

## Per-method diagnostics

| Question | Script |
|---|---|
| DINCAE accuracy + sigma-hat calibration + variance retention | `methods.dincae.checks.evaluate` |
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

# N consecutive frames, one PNG per frame (for temporal consistency)
sbatch sbatch/submit_plot_reconstruction_sequence.sbatch --day atc-20130811 --n 5000
```

Single frames land in `compare/results/reconstruction_<day>_<methods>.png`;
sequences in `compare/results/seq_ppt_<day>/` (gitignored — 5000 frames is
~600 MB, cheap to regenerate).

---

# Data, and the minimal package to transfer

## Where the data lives

Three pipeline stages in two locations, and **only the last is read at run
time**:

| Stage | Location | Size | Read at run time? |
|---|---|---|---|
| (1) raw ATC CSVs, 92 recording days | `/scratch/work/zhangx29/ATC/` | 225 GB | no |
| (2) trajectory H5, Sundays only, 46 days | `/scratch/work/zhangx29/data/ATC/Sundays/` | 36 GB | no — naming manifest only |
| (3) `grid_cache`, 4-channel 36x12 fields, 46 days | `/scratch/work/zhangx29/data/grid_cache/` | 3.2 GB | **yes** |

Stage (2) looks load-bearing because the split lists
(`data/sunday_atc_{train,valid,test}.lst`) name `ATC/Sundays/atc-YYYYMMDD.h5` —
but `observation_model.split_files()` only takes the filename stem and opens the
corresponding `grid_cache` file instead. Those 36 GB are never opened.

The data is the only thing outside the repo. One path in `crowdcore/config.yaml` points to
it, and it is what to repoint on another machine:

- `data.root` → `/scratch/work/zhangx29/data` (stages 2 and 3, plus the split lists)

Everything else the code reads is in the repo: the real ATC map
(`crowdcore/assets/atc_map/`, `localization_grid.pgm` + `.yaml`, 3.4 MB — the walkable
mask and line of sight are computed from it), the EnKF's code and its trained surrogate
weights (`methods/enkf/enkf_lab/`, 14 MB), and DINCAE's normalisation stats.

One exception, verified 2026-09-09: `methods/varnet/checks/check_data_pipeline.py`
hardcodes three absolute paths as its argparse **defaults** (`DEFAULT_CSV`,
`DEFAULT_REF_H5`, `DEFAULT_REF_CACHE`). It is the only script that still does —
it reaches back to pipeline stages 1 and 2, which nothing else touches, and it
accepts those paths as arguments. Everything else resolves through
`config.yaml`.

Rebuild stages 1 → 2 → 3 only if the raw CSVs or the grid resolution changed;
see `crowdcore/data/DOC_data_pipeline.md`.

## What is not in the repo

Code, config, metrics and figures are tracked. About 31 GB of generated data is
not (sizes as of 2026-09-13):

- `methods/dincae/cache/` grid encoding cache (~13 GB)
- `*.npz` EnKF exported fields (`methods/enkf/check_outputs/`, ~6.7 GB)
- `*.pt` model weights (`methods/*/runs/`, ~11 GB — mostly the periodic `ckpt_<epoch>.pt`
  snapshots of the 15 hidden=96 runs; the reported numbers need 0.54 GB of them, see below)
- `**/seq_ppt_*/` per-frame sequence figures

**The gridded ATC data itself is also not in git** — `git ls-files | grep '\.h5$'`
returns zero. It never has been, most likely because the ATC dataset carries its
own redistribution terms, not only because of size. So running inference (as
opposed to reading the already-computed results in `check_outputs/`) needs the
code **and** a copy of the gridded data **and** the checkpoints: three separate
things, not one repo clone.

## The minimal package: ~1.0 GB, not 225 GB

| Item | Path | Size |
|---|---|---|
| gridded fields, 7 test days only | `data/grid_cache/atc-{20130811,20130818,20130825,20130901,20130915,20130922,20130929}_corridor_1.0s.h5` | **483 MB** (all 46 days would be 3.2 GB) |
| split lists | `data/sunday_atc_{train,valid,test}.lst` | a few KB |
| the real ATC map | `crowdcore/assets/atc_map/` (`localization_grid.pgm` + `.yaml`) | 3.4 MB (already in git) |
| DINCAE normalisation stats | `methods/dincae/artifacts/state_stats.npz` | 17 KB (already in git) |
| trained weights | the reported checkpoints only: the 15 `methods/varnet/runs/varnet_*_h96_s*/ckpt_<epoch>.pt` named in each run's `select_valid.json`, `methods/dincae/runs/dincae_ff/ckpt_00060.pt`, `methods/senseiver/runs/senseiver_A/best.pt` | 0.54 GB |
| | **total** | **~1.0 GB** |

**DINCAE needs that one extra small file that weights alone do not carry.**
`state_stats.npz` holds the per-cell mean field and per-channel residual std,
computed once from the training data and read separately from the checkpoint at
inference time by `state.StateStats`. It used to be gitignored, caught by the
blanket `*.npz` rule meant for multi-GB files; it is now tracked via an explicit
negation, so a fresh clone has it. 4DVarNet and Senseiver do not have this
problem: Senseiver's `in_mean`/`in_std` are model buffers saved in the same
`state_dict`, and 4DVarNet works in raw physical units.

The EnKF's surrogate weights, `methods/enkf/enkf_lab/apt-ibex_train_model_28D.pth` (14 MB),
are tracked through an explicit `.gitignore` negation like `state_stats.npz`;
`enkf_opt/` links to the same file. Without them the filter cannot run.

