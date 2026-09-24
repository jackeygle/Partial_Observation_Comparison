#!/bin/bash
#SBATCH --job-name=supervisor_bench
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=supervisor_evaluation/outputs/logs/controlled_benchmark_%j.log
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
python3 -u supervisor_evaluation/evaluate.py benchmark \
  --data-root /scratch/work/zhangx29/data \
  --output-dir supervisor_evaluation/outputs/full
