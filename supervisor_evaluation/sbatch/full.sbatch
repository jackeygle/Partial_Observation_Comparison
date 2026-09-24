#!/bin/bash
#SBATCH --job-name=supervisor_full
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=supervisor_evaluation/outputs/logs/full_%j.log
# The complete evaluation: all six methods on the seven test days, unified
# uncertainty scoring, the controlled inference benchmark, CSVs and figures.
# Submit from the repository root:  sbatch supervisor_evaluation/sbatch/full.sbatch
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
python3 -u supervisor_evaluation/evaluate.py full \
  --data-root /scratch/work/zhangx29/data \
  --output-dir supervisor_evaluation/outputs/full
