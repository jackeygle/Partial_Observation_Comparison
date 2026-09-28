#!/bin/bash
#SBATCH --job-name=enkf_final_update
#SBATCH --partition=gpu-v100-32g,gpu-v100-16g,gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=supervisor_evaluation/outputs/logs/enkf_final_update_%j.log
# Re-run the final EnKF (FINAL_CONFIG changed: noise at the model's measured size) on the
# seven test days, then the unified uncertainty scoring, then original-vs-final.
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
python3 -u supervisor_evaluation/evaluate.py full --methods enkf \
  --data-root /scratch/work/zhangx29/data --output-dir supervisor_evaluation/outputs/full
python3 -u supervisor_evaluation/evaluate.py uncertainty \
  --data-root /scratch/work/zhangx29/data --output-dir supervisor_evaluation/outputs/full
python3 -u -m methods.enkf.checks.compare_original_final
