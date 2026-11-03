#!/bin/bash
#SBATCH --job-name=cov_smoke
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=supervisor_evaluation/outputs/logs/%x_%j.log
# Coverage smoke test: first test day up to frame 6800 (frames 400-5799 are a gap block); the rerun must agree with the reported
# run in its pooled scores. Not a result.
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
export DATA_VARIANT=all92 EVAL_VARIANT=six_allwalk_all92 DINCAE_VAR_TRANSFORM=none
O=supervisor_evaluation/outputs/dev/coverage_smoke_20261031
nvidia-smi --query-gpu=name --format=csv,noheader; hostname
python3 -u -m supervisor_evaluation.checks.coverage_run enkf --out $O --smoke-frames 6800
python3 -u -m supervisor_evaluation.checks.coverage_run check --out $O
python3 -u -m supervisor_evaluation.checks.coverage_run unc --out $O --days 1 --frames 6800
echo "[done] coverage smoke"
