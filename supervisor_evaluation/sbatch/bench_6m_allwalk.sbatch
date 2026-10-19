#!/bin/bash
#SBATCH --job-name=bench_6m_allwalk
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=supervisor_evaluation/outputs/logs/bench_6m_allwalk_%j.log
# Timing for the six-method variant on whichever gpu-debug GPU is free; all six methods are
# timed in the same job, so they are comparable with each other (the GPU is recorded).
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
export EVAL_VARIANT=six_allwalk DINCAE_VAR_TRANSFORM=none
O=supervisor_evaluation/outputs/full_6m_allwalk
nvidia-smi --query-gpu=name --format=csv,noheader; hostname
python3 -u supervisor_evaluation/evaluate.py benchmark --data-root /scratch/work/zhangx29/data --output-dir $O
