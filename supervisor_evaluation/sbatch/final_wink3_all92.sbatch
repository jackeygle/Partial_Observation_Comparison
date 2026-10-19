#!/bin/bash
#SBATCH --partition=gpu-v100-32g,gpu-v100-16g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/work/zhangx29/data_all92/logs/%x_%j.log
# Route 2 4DVarNet selection on all walkable validation cells (gap mask on) + batch-1 test
# inference, one run per job on a cheap GPU (user, 2026-10-15).
# Usage: sbatch --job-name=final_<x> supervisor_evaluation/sbatch/final_wink3_all92.sbatch <run dir name> <out name>
source /etc/profile 2>/dev/null
export DATA_VARIANT=all92
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || exit 1
set -e
cd "$THESIS_ROOT"
nvidia-smi --query-gpu=name --format=csv,noheader
OUT="$THESIS_ROOT/supervisor_evaluation/outputs/dev/$2"
mkdir -p "$OUT"
python3 -u -m supervisor_evaluation.checks.final_wink3 --run-dir "$THESIS_ROOT/methods/varnet/runs/$1" --out "$OUT"
