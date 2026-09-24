#!/bin/bash
#SBATCH --job-name=uq_unified_smoke
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=supervisor_evaluation/outputs/logs/uq_smoke_%j.log
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
# One day, first 1400 frames, against the saved seven-day EnKF exports.
python3 -u supervisor_evaluation/evaluate.py uncertainty --days 1 --frames 1400 \
  --data-root /scratch/work/zhangx29/data \
  --output-dir supervisor_evaluation/outputs/dev/uq_smoke
