# check_outputs/ — output of the diagnostic and figure scripts (all regenerable)

Scripts in `checks/` do **validation and plotting**; their output lands here.
Directories are created on demand, so a fresh clone has fewer of them than this
table lists — that is expected, not a missing file.

| Subdirectory | Contents | Producing script |
|---|---|---|
| `eval/` | the bulk of it: metric JSON and every figure the decks and README use (`m4_*` current deck, `unc_*` uncertainty, `arch_*`/`genn_*` architecture, `speed_*`, `mt_*`, per-day `test_metrics_*.json`) | most of `checks/`, plus `compare/` |
| `navigation/` | `nav_mask_on_map.png`, `obstacle_map.png` — the walkable/obstacle mask on the real ATC map | `checks/check_navigation.py`, `checks/check_map_orientation.py`, `checks/plot_nav_mask.py`, `checks/plot_obstacle_map.py` |
| `observation_model/` | 4-panel mask-generation frames, `coverage_vs_range.png`, `io_report.txt`, `stats.json` — created when the check is run | `checks/check_observation_model.py` |
| `_k1rate/`, `_prof/` | one-off EnKF rate/profiling working directories, kept for the measured numbers in their `.out` files | self-contained, not referenced elsewhere |

Regenerate the map figures and the observation-model check:

```bash
source sbatch/_env.sh
python3 -m methods.varnet.checks.check_navigation
python3 -m methods.varnet.checks.check_observation_model --file first --frames 60 --init prev
```

The current deck and its figures are built separately:

```bash
python3 -m compare.plot_meeting4          # -> eval/m4_*.png
python3 -m slides.build_meeting4_deck     # -> slides/meeting4_deck.{pptx,pdf} + notes
```

Note: `check_observation_model.py` **clears** its own output subdirectory on every
run before writing to it.
