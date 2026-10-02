# thesis_figures — the thesis's schematics and example figures

Every figure is a matplotlib script in the thesis style: Computer Modern text (the
LaTeX body font), 16 cm wide (the text width), thin black line work, and the method
colours and colormaps of `compare/plotstyle.py`, so they match the result figures in
`supervisor_evaluation/outputs/full/figures/`. Each writes a vector PDF for LaTeX and
a PNG preview to `out/`.

```bash
source sbatch/_env.sh                       # from the repository root
python3 -m thesis_figures.prepare_frame     # once: caches the example frame  -> data/
python3 -m thesis_figures.prepare_day       # once: caches one day's coverage -> data/
python3 -m thesis_figures.<name>            # -> out/<name>.pdf, out/<name>.png
```

| name | figure |
|---|---|
| `pipeline` | the study end to end: truth, robot observations, the six methods, output, evaluation |
| `study_area` | the ATC map with the gridded corridor; walkable cells; the four state channels |
| `observation` | the robot observation model; per-cell and per-second coverage; age of blind cells |
| `time_windows` | which seconds of observations each method uses to reconstruct frame $t$ |
| `senseiver` | Senseiver-A and Senseiver-G (ours) |
| `dincae` | DINCAE: information-form input, U-Net, refinement, mean and sigma-hat |
| `varnet` | 4DVarNet (variational cost, learned gradient descent) and the aug. var. model (ours) |
| `enkf` | the EnKF cycle with structured process noise |
| `comparison` | the six density reconstructions and their errors on one test frame (two frames) |
| `uncertainty_maps` | error against predicted sigma-hat for the three probabilistic methods |
| `attention_block` | inside one Senseiver encoder block: the cross-attention and self-attention layers |
| `pedpred` | the PedPred3 forecast model of the EnKF (encoder and forecaster) |
| `data_processing` | preprocessing of the ATC tracking records into the gridded state |
| `breakdown` | error by age of the blind cell, crowd size and test day (reads `outputs/full/breakdown/`, from `supervisor_evaluation/breakdown.py`) |

The example frame is test day 2013-08-11, frame 32778 (a busy second); `comparison` and
`uncertainty_maps` also draw frame 13551. Reconstructions come from the evaluation's saved
selection, `supervisor_evaluation/outputs/full/images/selected_predictions.npz`
(`evaluate.py images`, final models); rerun `prepare_frame` after regenerating it.
`common.py` holds the shared page, block, thumbnail and colour-bar helpers.
