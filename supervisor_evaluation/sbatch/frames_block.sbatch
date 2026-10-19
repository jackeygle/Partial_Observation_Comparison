#!/bin/bash
#SBATCH --job-name=frames_block
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=supervisor_evaluation/outputs/logs/frames_block_%j.log
# Five-method reconstructions of one contiguous test block, for visual inspection
# (supervisor_evaluation/frames_block.py). Writes outputs/dev/frames_<n>/; reported
# scores are untouched. Submit from the repository root:
#   sbatch supervisor_evaluation/sbatch/frames_block.sbatch
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
python3 -u -m supervisor_evaluation.frames_block --day atc-20130811 --start 30000 --frames 5000
