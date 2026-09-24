#!/bin/bash
#SBATCH --job-name=uq_unified_full
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=supervisor_evaluation/outputs/logs/uq_unified_full_%j.log
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
# Seven test days, all three uncertainty methods on one protocol in physical units.
# Reuses the saved EnKF exports; rewrites uncertainty.csv, calibration.csv and figures.
python3 -u supervisor_evaluation/evaluate.py uncertainty \
  --data-root /scratch/work/zhangx29/data \
  --output-dir supervisor_evaluation/outputs/full
