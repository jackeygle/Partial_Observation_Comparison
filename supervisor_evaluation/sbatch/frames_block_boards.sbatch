#!/bin/bash
#SBATCH --job-name=frames_boards
#SBATCH --partition=batch-csl,batch-skl,batch-milan
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=supervisor_evaluation/outputs/logs/frames_block_boards_%j.log
# One board per frame for the whole block, drawn from the saved predictions.npz of
# frames_block.sbatch (no inference, CPU only).
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
python3 -u -m supervisor_evaluation.frames_block --frames 5000 --day atc-20130811 \
    --redraw-boards --board-every 1 --workers "${SLURM_CPUS_PER_TASK}"
