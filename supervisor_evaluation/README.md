# Supervisor evaluation

One entry point reproduces the frozen comparison.  There is no additional YAML
configuration: model choices, scientific parameters, split dates, and output
metrics are declared at the top of `evaluate.py`.

## Quick start (Triton, GPU)

Everything runs through `evaluate.py`, from the repository root; it needs the
rest of the repository (`crowdcore/`, `compare/`, `methods/`) and the data
directory. `bench_inference.py` is called by it and is not run on its own.

```bash
sbatch supervisor_evaluation/sbatch/smoke.sbatch   # ~5 min on gpu-debug: end-to-end check
sbatch supervisor_evaluation/sbatch/full.sbatch    # complete evaluation -> outputs/full/
```

`full` verifies the packaged checkpoints and data, scores all six methods on
the seven test days, runs the unified uncertainty evaluation and the inference
benchmark, and writes the CSVs and figures. Logs go to `outputs/logs/`; smoke
and verify results go to `outputs/dev/`, never into `outputs/full/`.

## Commands

From the repository root, in the project PyTorch environment:

```bash
module load scicomp-pytorch-env/2026.1

# Check packages, checkpoint hashes, HDF5 schema, shapes, and test dates.
python3 supervisor_evaluation/evaluate.py verify \
  --data-root /scratch/work/zhangx29/data \
  --output-dir supervisor_evaluation/outputs/dev/verify

# Short pipeline check. These numbers are not thesis results.
python3 supervisor_evaluation/evaluate.py smoke \
  --data-root /scratch/work/zhangx29/data \
  --output-dir supervisor_evaluation/outputs/dev/smoke

# Seven held-out test days.
python3 supervisor_evaluation/evaluate.py full \
  --data-root /scratch/work/zhangx29/data \
  --output-dir supervisor_evaluation/outputs/full

# Recheck inference speed on one GPU without rerunning the seven-day scores.
python3 supervisor_evaluation/evaluate.py benchmark \
  --data-root /scratch/work/zhangx29/data \
  --output-dir supervisor_evaluation/outputs/full

# One image per matched time point: Truth + observations + all five methods.
# Run after `full`, because the EnKF posterior is reused from its saved exports.
python3 supervisor_evaluation/evaluate.py images --n-images 100 \
  --data-root /scratch/work/zhangx29/data \
  --output-dir supervisor_evaluation/outputs/full
```

Uncertainty (4DVarNet aughead_obs, DINCAE, EnKF) is scored by one function in
`evaluate.py`, also run at the end of `smoke`/`full`. It reuses the saved EnKF
exports, so after changing only the scoring it can be rerun on its own:

```bash
python3 supervisor_evaluation/evaluate.py uncertainty \
  --data-root /scratch/work/zhangx29/data \
  --output-dir supervisor_evaluation/outputs/full
```

All three methods are scored on the same frames (common support of DINCAE's
t-1/t+1 context, complete 4DVarNet windows and the EnKF warmup) and the same cells
(walkable, not observed at that frame, all four channels), in physical units.
DINCAE's variance channel is log1p-transformed inside the model, so its physical
predictive is a shifted log-normal and is scored with the closed-form log-normal
CRPS; every other channel/method is Gaussian. The null model is each method's
own N(point, RMSE^2) on the same cells. Output: `raw/uncertainty_unified.json`.

The developer/package-maintainer runs this once after final model selection:

```bash
python3 supervisor_evaluation/evaluate.py prepare
```

It copies only the reported checkpoints and required fitted artifacts into
`models/`, then records SHA-256 fingerprints.  Evaluation never falls back to a
checkpoint elsewhere in `runs/`.

## Outputs

The stable supervisor-facing products are `accuracy.csv`, `uncertainty.csv`,
`calibration.csv`, `inference_time_controlled.csv`, `verification_report.json`, and the
summary figures below `figures/`. Method-native per-day JSON files are retained
below `raw/` for auditability.

The summary figures under `figures/` are thesis-ready: 6.3 in text width, PNG at
300 dpi, one colour per method and one colormap per quantity from
`compare/plotstyle.py` (the method palette passes the dataviz colour-vision
checks). `figures/captions.md` holds a generated caption for each, including the
run metadata that is kept off the figures themselves.

- `accuracy_rmse` — grouped bars of RMSE per method, pooled and per channel; the
  same numbers are in `accuracy_table.tex` with the best value per column in bold.
- `uncertainty_summary` — pooled ("All") and per-channel (a) CRPS skill vs. the
  constant-sigma null and (b) spread/RMSE; the pooled bars carry 95% bootstrap
  intervals over the seven test days. Per-channel numbers:
  `uncertainty_by_channel.csv`; interval coverage (not plotted): `calibration.csv`.
- `inference_latency` — median GPU time per frame per method (bars, log scale).

`images` adds `images/comparison_*.png`, `sample_manifest.csv`,
`selected_predictions.npz`, `spread_comparison.png`, `color_scales.json`, and a
10-by-10 contact sheet.
Frames are distributed over the seven test days, stratified over the daily
walkable-density range, and constrained to the common support of all methods.
The manifest records each selected date/frame, density level, timestamp, and
observed fraction. The output images use shared density and error color scales
across every date and method. Each image follows the EnKF map style: rows are
methods (including the single `aughead_obs` checkpoint), and columns show
partial observations, truth, reconstruction, and absolute error. The spread
figure compares DINCAE, `aughead_obs`, and EnKF on one common selected frame in
physical density units with a shared spread scale. The NPZ retains all four
predicted channels and the selected density spread maps.
After adjusting plot style, `python3 supervisor_evaluation/evaluate.py redraw
--output-dir supervisor_evaluation/outputs/full` refreshes the PNGs from the saved
predictions without rerunning inference.

Smoke mode uses one day and at most 512 frames. Full mode uses all seven test
days and automatically runs the controlled benchmark when all methods are
selected. `benchmark` uses the first 600 frames of the first test day, the same
GPU for all six methods, and three timed repeats after warm-up. It times predictor
compute with inputs prepared, excluding model loading and scoring. The reported
milliseconds per output frame measure compute throughput. The 4DVarNet models
need a complete 200-frame window before an output can be returned, so their
throughput number is not the response delay for one arriving observation.
The legacy `inference_time.csv` retains the seven-day evaluator timings, which
were collected with different timing scopes and are not used in the final speed
figure when `inference_time_controlled.csv` is available.

DINCAE uses the centered context `t-1,t,t+1`; 4DVarNet reconstructs a window.
Senseiver-A, Senseiver-G (`t-15..t`), and EnKF (`t-4..t` analyses) are causal.
