#!/bin/bash
#SBATCH --job-name=final_fold_eval
#SBATCH --partition=gpu-h200-141g-short,gpu-a100-80g,gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=supervisor_evaluation/outputs/logs/final_fold_eval_%j.log
source /etc/profile 2>/dev/null
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
python3 -u -m supervisor_evaluation.checks.run_fold_evaluation --phase "${1:-all}"
