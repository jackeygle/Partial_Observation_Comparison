# Evaluation outputs — what is where

| Folder | What | Used by |
|---|---|---|
| `full_6m_all92/` | **Final results**: six methods, all 92 ATC days (63/14/15), recording-gap blocks excluded (`EVAL_VARIANT=six_allwalk_all92`, `DATA_VARIANT=all92`) | `Thesis_writing/Thesis_paper20260219` (figures: `THESIS_VARIANT=6methods_all92`) |
| `reselect_all_walkable_all92_gapmask/` | Route-2 checkpoint selection (Senseiver-A/G, DINCAE) on all walkable validation cells, gap blocks excluded | read by `EVAL_VARIANT=six_allwalk_all92` |
| `dev/varnet_*_all92_gapmask_20261015/` | Route-2 selection of the two 4DVarNet models | read by `EVAL_VARIANT=six_allwalk_all92` |
| `full_6m_allwalk/` | Route 1 (Sundays only), six methods — fallback | `EVAL_VARIANT=six_allwalk` |
| `full_5m_allwalk_gaussdincae_20261011/` | Route 1, five methods | `Thesis_writing/Thesis_paper_5methods` |
| `full_5m_allwalk_20261011/` | Route 1, five methods, before the Gaussian DINCAE (its accuracy files are reused by the five-method jobs) | `sbatch/full_5m_*.sbatch` |
| `reselect_all_walkable_20261010/` | Route-1 checkpoint selection on all walkable cells | route-1 variants |
| `reselect/` | Route-1 EnKF configuration selection (validation arrays) | `reselect_validation.py`, `checks/reselect_all_walkable.py` |
| `full/` | Original default output directory of `evaluate.py` (oldest six-method results) | default `--output-dir`, default figure variant |
| `dev/` | Test runs, smoke tests and selection runs (not reported results) | — |
| `logs/` | Slurm logs of the evaluation jobs | — |
| `_archive/` | Superseded evaluations, kept for the record; nothing reads them | — |
