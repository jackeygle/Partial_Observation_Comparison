#!/bin/bash
#SBATCH --job-name=diag_dincae_out
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=supervisor_evaluation/outputs/logs/diag_dincae_outliers_%j.log
# Route 2: DINCAE's unclipped predictions outside the physical range, per test day (user, 2026-10-15).
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
export DATA_VARIANT=all92 EVAL_VARIANT=six_allwalk_all92 DINCAE_VAR_TRANSFORM=none
nvidia-smi --query-gpu=name --format=csv,noheader
python3 -u -m supervisor_evaluation.checks.diag_dincae_outliers
