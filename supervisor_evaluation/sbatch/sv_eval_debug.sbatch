#!/bin/bash
#SBATCH --job-name=sv_eval_dbg
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=supervisor_evaluation/outputs/logs/sv_eval_debug_%j.log
# Senseiver re-evaluation on the test days + example frames (split from senseiver_reselect.sbatch to fit gpu-debug)
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
python3 -u supervisor_evaluation/evaluate.py full --methods senseiver_a senseiver_g \
  --data-root /scratch/work/zhangx29/data --output-dir supervisor_evaluation/outputs/full
python3 -u supervisor_evaluation/evaluate.py images --n-images 100 \
  --data-root /scratch/work/zhangx29/data --output-dir supervisor_evaluation/outputs/full
