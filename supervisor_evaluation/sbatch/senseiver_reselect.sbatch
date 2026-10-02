#!/bin/bash
#SBATCH --job-name=sv_reeval
#SBATCH --partition=gpu-h200-141g-short,gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=supervisor_evaluation/outputs/logs/senseiver_reselect_%j.log
#
# After both Senseiver models were retrained and their epoch chosen on the whole
# validation split (methods/senseiver/checks/select_checkpoint.py): rescore them on the
# test days, redraw the example frames, and redo the error breakdown of all six methods.
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
python3 -u supervisor_evaluation/evaluate.py full \
  --methods senseiver_a senseiver_g \
  --data-root /scratch/work/zhangx29/data \
  --output-dir supervisor_evaluation/outputs/full
python3 -u supervisor_evaluation/evaluate.py images --n-images 100 \
  --data-root /scratch/work/zhangx29/data \
  --output-dir supervisor_evaluation/outputs/full
python3 -u supervisor_evaluation/breakdown.py --data-root /scratch/work/zhangx29/data \
  --output-dir supervisor_evaluation/outputs/full
