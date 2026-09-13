# check_outputs/ — output of the diagnostic and figure scripts (all regenerable)

Scripts in `checks/` do **validation and plotting**; their output lands here.
Directories are created on demand, so a fresh clone has fewer of them than this
table lists — that is expected, not a missing file.

| Subdirectory | Contents | Producing script |
|---|---|---|
| `eval/` | the bulk of it: metric JSON (`uncertainty_*.json`, `enkf_spread_growth.json`, ...) and two kinds of figure — `unc_spread_decay.png` (the EnKF's ensemble collapse) and `de_*` method diagrams. Figures drawn from the deleted hidden=32 runs (`arch_*`, `genn_*`, `math_*`, `results_*`, `train_*`, `mt_*`, `fw_*`, `speed_*`, and the ml5 `unc_*` plots) were removed on 2026-09-13. `arch_*`/`genn_*`/`math_*` can be redrawn from the current model (`plot_architecture.py`, `plot_genn_detail.py`, `plot_math_detail.py`, after `diag_channel_budget.py`/`diag_solver_trace.py`); the others' scripts were deleted with them | most of `checks/`, plus `compare/` |
| `navigation/` | `nav_mask_on_map.png`, `obstacle_map.png` — the walkable/obstacle mask on the real ATC map | `checks/check_navigation.py`, `checks/check_map_orientation.py`, `checks/plot_nav_mask.py`, `checks/plot_obstacle_map.py` |
| `observation_model/` | 4-panel mask-generation frames, `coverage_vs_range.png`, `io_report.txt`, `stats.json` — created when the check is run | `checks/check_observation_model.py` |
| `_k1rate/`, `_prof/` | one-off EnKF rate/profiling working directories, kept for the measured numbers in their `.out` files | self-contained, not referenced elsewhere |

Regenerate the map figures and the observation-model check:

```bash
source sbatch/_env.sh
python3 -m methods.varnet.checks.check_navigation
python3 -m methods.varnet.checks.check_observation_model --file first --frames 60 --init prev
```

Note: `check_observation_model.py` **clears** its own output subdirectory on every
run before writing to it.
