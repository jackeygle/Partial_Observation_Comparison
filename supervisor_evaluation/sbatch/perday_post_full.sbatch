#!/bin/bash
#SBATCH --job-name=perday_post_full
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=supervisor_evaluation/outputs/logs/perday_post_full_%j.log
# Refresh the thesis images and breakdown after all six methods have been scored.
set -euo pipefail
cd /scratch/work/zhangx29/Thesis_Project
source sbatch/_env.sh
python3 -u supervisor_evaluation/evaluate.py images --n-images 100   --data-root /scratch/work/zhangx29/data   --output-dir supervisor_evaluation/outputs/full
python3 -u supervisor_evaluation/breakdown.py   --data-root /scratch/work/zhangx29/data   --output-dir supervisor_evaluation/outputs/full
