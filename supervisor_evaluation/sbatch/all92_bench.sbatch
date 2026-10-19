#!/bin/bash
#SBATCH --job-name=all92_bench
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=supervisor_evaluation/outputs/logs/all92_bench_%j.log
# Route 2 timing: all six methods in one job on whichever gpu-debug GPU is free (GPU recorded).
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
export DATA_VARIANT=all92 EVAL_VARIANT=six_allwalk_all92 DINCAE_VAR_TRANSFORM=none
D=/scratch/work/zhangx29/data_all92
O=supervisor_evaluation/outputs/full_6m_all92
E=supervisor_evaluation/evaluate.py
nvidia-smi --query-gpu=name --format=csv,noheader; hostname
python3 -u $E benchmark --data-root $D --output-dir $O
python3 -u $E collect --output-dir $O
