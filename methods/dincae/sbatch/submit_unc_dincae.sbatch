#!/bin/bash
#SBATCH --job-name=uncdinc
#SBATCH --partition=gpu-h200-141g-short,gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=methods/dincae/check_outputs/uncdinc_%j.out
# DINCAE's native sigma^2 -- the only method besides 4DVarNet that predicts variance,
# yet it had always been missing from table B, because its own evaluate.py only reports
# the geoscience-style calibration table, not CRPS / spread-skill / coverage.
# Two inference passes: the constant-sigma baseline's sigma is each slice's own RMSE,
# which is only known after the first pass finishes.
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || { echo "submit from the repo root" >&2; exit 1; }
cd "$THESIS_ROOT/methods/dincae"
# Defaults to the PUBLISHED configuration (single epoch-60 checkpoint of runs/dincae_ff), which is
# what uncertainty_dincae.json holds. Without this, evaluate.load_models matches
# all 16 ckpt_*.pt and refuses to average silently, so the entry point would fail.
RUN="$THESIS_ROOT/methods/dincae/runs/dincae_ff"
CKPT=$RUN/ckpt_00060.pt
srun python3 -u -m methods.dincae.checks.eval_uncertainty_dincae --run-dir "$RUN" --ckpt-glob "$CKPT" "$@"
