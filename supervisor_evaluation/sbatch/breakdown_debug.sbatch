#!/bin/bash
#SBATCH --job-name=bd_dbg
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=supervisor_evaluation/outputs/logs/breakdown_debug_%j.log
# Error breakdown of the six methods (split from senseiver_reselect.sbatch to fit gpu-debug)
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
python3 -u supervisor_evaluation/breakdown.py --data-root /scratch/work/zhangx29/data \
  --output-dir supervisor_evaluation/outputs/full
