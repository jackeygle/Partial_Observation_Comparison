# Partial Observation Comparison

Comparing five methods for reconstructing crowd density/velocity fields from
**partial observations** on the ATC pedestrian trajectory dataset.

For step-by-step commands (training, reproducing the headline comparison,
per-method diagnostics), see [HOW_TO_USE.md](HOW_TO_USE.md).

## Structure

```
crowdcore/          shared by all five methods: the method-agnostic half
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
python3 -m compare.compare5              # five-way comparison, four conventions
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

## The five methods

| Directory | Method | Uncertainty output |
|---|---|---|
| `methods/varnet/` | 4DVarNet (plain MSE, Eq.14) | none |
| `methods/varnet/` | 4DVarNet + uncertainty head (NLL, 5-member ensemble) | learned sigma-hat + epistemic |
| `methods/dincae/` | DINCAE (16-checkpoint output averaging) | sigma-hat (information form) |
| `methods/senseiver/` | Senseiver | none |
| `methods/enkf/` | localised EnKF (+ PedPred3 forward model) | ensemble spread |

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
data, about 30 GB, is excluded by `.gitignore`:

- `*.pt` model weights (`methods/*/runs/`, ~1 GB)
- `*.npz` EnKF estimated fields (`methods/varnet/check_outputs/enkf*`, ~18 GB)
- `methods/dincae/cache/` grid cache (~13 GB)
- `**/check_outputs/eval/seq_ppt_*/` per-frame sequence figures
