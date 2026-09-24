#!/bin/bash
#SBATCH --job-name=supervisor_aug_figs
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=supervisor_evaluation/outputs/logs/aughead_full_images_%j.log
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
python3 -u supervisor_evaluation/evaluate.py full \
  --methods varnet_aughead \
  --data-root /scratch/work/zhangx29/data \
  --output-dir supervisor_evaluation/outputs/full
python3 -u supervisor_evaluation/evaluate.py images --n-images 100 \
  --data-root /scratch/work/zhangx29/data \
  --output-dir supervisor_evaluation/outputs/full
