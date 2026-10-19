#!/bin/bash
#SBATCH --job-name=all92_bd_shard
#SBATCH --partition=gpu-debug,gpu-v100-32g
#SBATCH --gres=gpu:1,min-vram:32G
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=supervisor_evaluation/outputs/logs/%x_%j.log
# Route 2 error breakdown, one part of the test days (fits gpu-debug's 30 min; user, 2026-10-15).
# Usage: sbatch all92_breakdown_shard.sbatch <k> <n>; then all92_finish.sbatch <n>
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
export DATA_VARIANT=all92 EVAL_VARIANT=six_allwalk_all92 DINCAE_VAR_TRANSFORM=none
D=/scratch/work/zhangx29/data_all92
O=supervisor_evaluation/outputs/full_6m_all92
E=supervisor_evaluation/evaluate.py
nvidia-smi --query-gpu=name --format=csv,noheader; hostname
python3 -u supervisor_evaluation/breakdown.py --data-root $D --output-dir $O --shard $1/$2
