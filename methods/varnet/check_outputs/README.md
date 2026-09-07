# check_outputs/ — output of the diagnostic scripts (figures/reports, all regenerable)

The scripts in `checks/` are responsible for **validation and plotting**; their
output all lands here, one subdirectory per module:

| Subdirectory | Contents | Producing script |
|---|---|---|
| `data_pipeline/` | pipeline_1~4_*.png: figures explaining the CSV->H5->grid pipeline (coordinate transform / triangular kernel / 4 channels / windowing) | `checks/plot_data_pipeline.py` |
| `navigation/` | astar_paths.png: visualisation of A* paths on the real corridor mask | `checks/check_navigation.py` |
| `navigation/` | nav_mask_on_map.png / map_orientation_check.png: navigation mask on the real map + orientation diagnostic | `checks/check_map_orientation.py` |
| `observation_model/` | frame_*.png (4-panel mask-generation figures), coverage_vs_range.png, io_report.txt, stats.json | `checks/check_observation_model.py` |
| `training/` | varnet_training.png: blind-cell MSE convergence curve (data from runs/varnet/metrics.jsonl) | `slides/build_slides.py` |

Regenerate everything:

```bash
module load scicomp-pytorch-env/2026.1
python3 checks/check_navigation.py
python3 checks/plot_data_pipeline.py
python3 checks/check_observation_model.py --file first --frames 60 --init prev
python3 slides/build_slides.py          # training curve + PPT/PDF/speaker notes
```

Note: `checks/check_observation_model.py` **clears** its own output subdirectory on
every run before writing to it.
