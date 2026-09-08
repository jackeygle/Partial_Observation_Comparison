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

**Where to start**

| If you want to | Read |
|---|---|
| understand what is here and why the numbers move | this file |
| actually run something | [HOW_TO_USE.md](HOW_TO_USE.md) |
| know which checkpoint produced which published number, or what to transfer to reproduce it | [MODELS_FOR_ADVISOR.md](MODELS_FOR_ADVISOR.md) |

## Structure

```
crowdcore/          shared by every method: the method-agnostic half
  config.py+yaml      single source of truth for all parameters
  navigation.py       walkable region, A* path planning, the real ATC map
  observation_model.py multi-robot sensor motion and observation generation, Omega, X0 fill-in
  paths.py            single source of truth for in-repo paths
  data/               CSV -> h5 -> 4-channel grid

methods/            each subpackage is one method, none import each other
  varnet/             4DVarNet variational assimilation (plain MSE and NLL/uncertainty-head arms)
  enkf/               localised EnKF -- a read-only vendor copy of Partial_observation
  dincae/             DINCAE convolutional autoencoder inpainting
  senseiver/          Senseiver sparse-sensor reconstruction

compare/            cross-method evaluation and plotting. The only place allowed to
                    import multiple methods at once
slides/             decks for meetings
sbatch/_env.sh      the single environment entry point
```

## How to run

```bash
source sbatch/_env.sh                    # module load + PYTHONPATH + PYTHONSAFEPATH
python3 -m compare.compare5              # all methods and arms, four scoring conventions
python3 -m compare.plot_compare5         # the main results figure
python3 -m methods.varnet.train --help
python3 -m methods.enkf.checks.verify_enkf_opt --frames 12
```

**You must use `python3 -m`.** Running `python3 methods/varnet/train.py` directly
puts `methods/varnet/` onto `sys.path[0]`, and the three methods each have their
own **different** `losses.py` (same for `dataset.py`, `model.py`) — whichever hits
the search path first wins. `PYTHONSAFEPATH=1` in `_env.sh` is what blocks that.

Job scripts live under each method's `sbatch/`; they cd into the method directory
themselves, so relative paths like `runs/`, `check_outputs/` keep working as usual:

```bash
sbatch methods/varnet/sbatch/submit_varnet.sbatch --days 32 --dT 200
sbatch methods/dincae/sbatch/submit_eval.sbatch
```

## Four methods, plus two uncertainty designs of our own

Counted as **four reconstruction methods**. 4DVarNet appears three times
because we added two uncertainty designs to it that the paper does not have —
those are variants of one method, not separate methods. This is the framing
agreed with the supervisor on 2026-09-07.

| Directory | Method | Uncertainty output |
|---|---|---|
| `methods/varnet/` | 4DVarNet (plain MSE, Eq.14) — the reconstruction method | none |
| `methods/varnet/` | ... + `aug0`: sigma inside the prior operator `G(x)`, NLL (**ours**) | learned sigma-hat + epistemic |
| `methods/varnet/` | ... + `vsb0`: sigma from a separate read-out head, NLL (**ours**, negative control) | learned sigma-hat + epistemic |
| `methods/dincae/` | DINCAE (single checkpoint; see note below) | sigma-hat (information form) |
| `methods/senseiver/` | Senseiver | none |
| `methods/enkf/` | localised EnKF (+ PedPred3 forward model) | ensemble spread |

**Every headline row is a single model**, default configuration, seed 0, each
one's checkpoint picked on its own validation set — no ensembling and no
checkpoint averaging, because neither paper reports one. DINCAE's reference
implementation *does* average the outputs of checkpoints saved every 10 epochs,
and `checks/evaluate.py` still supports it (that is its default `--ckpt-glob`),
but the reported number comes from **one** checkpoint, `ckpt_00070.pt`. The
16-checkpoint average is kept as a measured side quantity (worth 1.7%, less
than picking the right single checkpoint), not as the headline. See
`MODELS_FOR_ADVISOR.md` for exactly which file backs which published number.

`methods/enkf/enkf_lab/` is a **byte-identical, read-only copy** of
`/scratch/work/zhangx29/Partial_observation`, its files deliberately chmod 444. To
change anything, edit `enkf_opt/` instead, and it must pass the bit-identical
comparison in `methods/enkf/checks/verify_enkf_opt.py` (`np.array_equal`, not
`isclose`).

## Scoring convention: a pitfall you must read

**Rankings flip depending on the convention.** On the same predictions, under the
"all cells" convention Senseiver comes first and DINCAE comes last, 12x worse;
switch to the "channel-defined cells" convention and DINCAE comes first, Senseiver
third.

The cause is the velocity channels: an empty cell has no people, hence no
velocity, and the `0` the data pipeline stores there is a placeholder, not a
measurement. And **88.4%** of blind cells are empty. The three methods trained on
the full field learned to output 0 on empty cells and get that part for free;
DINCAE was only ever trained on defined cells, and gets crushed by this term.

So **any chart that reports a single number without saying which cells it scores is
misleading**. `compare/compare5.py` computes both conventions side by side; the
single definition of the rule lives in `methods/dincae/state.py:channel_valid()`.

Also, 4DVarNet's numbers have a **~4e-4 reproducibility floor** (its inference pass
also backpropagates through autograd), so the 4th decimal place is noise — see
`refactor_baseline/README.md`.

## What is not in the repo

The repo holds only code, config, metrics, and figures. The following generated
data, about 21 GB, is excluded by `.gitignore` (sizes as of 2026-09-08):

- `methods/dincae/cache/` grid encoding cache (~13 GB)
- `*.npz` EnKF exported fields (`methods/enkf/check_outputs/`, ~6.3 GB)
- `*.pt` model weights (`methods/*/runs/`, ~1.5 GB)
- `**/seq_ppt_*/` per-frame sequence figures

**The gridded ATC data itself is also not here** (`*.h5`, `git ls-files` returns
zero) and is needed to run anything. See `MODELS_FOR_ADVISOR.md` for the minimal
~1.4 GB package that reproduces the published inference.
