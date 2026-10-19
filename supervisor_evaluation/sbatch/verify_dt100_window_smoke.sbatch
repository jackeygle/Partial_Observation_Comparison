#!/bin/bash
#SBATCH --job-name=verify_dt100_window
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1,min-vram:16g
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=supervisor_evaluation/outputs/dev/varnet_dt100_window_smoke_20261010/verify_%j.out
set -e
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh"
cd "$THESIS_ROOT"
python3 -u -m supervisor_evaluation.checks.verify_dt100_window_smoke \
    --checkpoint methods/varnet/runs/smoke_wink3_dt100_dev/varnet_last.pt \
    --output supervisor_evaluation/outputs/dev/varnet_dt100_window_smoke_20261010/grouping_check.json
