#!/bin/bash
#SBATCH --job-name=supervisor_smoke
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=supervisor_evaluation/outputs/logs/smoke_%j.log
# Short end-to-end check of the whole pipeline (one test day, a few hundred
# frames). The numbers are not thesis results.
# Submit from the repository root:  sbatch supervisor_evaluation/sbatch/smoke.sbatch
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
python3 -u supervisor_evaluation/evaluate.py smoke \
  --data-root /scratch/work/zhangx29/data \
  --output-dir "${SMOKE_DIR:-supervisor_evaluation/outputs/dev/smoke}"
