#!/bin/bash
#SBATCH --job-name=frames_all92
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1,min-vram:32G
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=supervisor_evaluation/outputs/logs/frames_all92_%j.log
# Six final methods (route 2) on 5000 consecutive frames of a test day, for visual inspection
# (supervisor_evaluation/frames_block.py): predictions, animation, RMSE over time. The boards of
# every frame are drawn by frames_boards_all92.sbatch (CPU). Reported scores are untouched.
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
export DATA_VARIANT=all92 EVAL_VARIANT=six_allwalk_all92 DINCAE_VAR_TRANSFORM=none
nvidia-smi --query-gpu=name --format=csv,noheader
python3 -u -m supervisor_evaluation.frames_block --day atc-20130811 --start 30000 --frames 5000 --board-every 25
