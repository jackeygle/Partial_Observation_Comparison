#!/bin/bash
#SBATCH --job-name=boards_all92
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=supervisor_evaluation/outputs/logs/boards_all92_%j.log
# One board per frame for the 5000-frame route-2 block, from the saved predictions.npz of
# frames_block_all92.sbatch (no inference, CPU only).
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
export DATA_VARIANT=all92 EVAL_VARIANT=six_allwalk_all92 DINCAE_VAR_TRANSFORM=none
python3 -u -m supervisor_evaluation.frames_block --frames 5000 --day atc-20130811 \
    --redraw-boards --board-every 1 --workers "${SLURM_CPUS_PER_TASK}"
