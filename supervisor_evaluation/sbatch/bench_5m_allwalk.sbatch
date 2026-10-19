#!/bin/bash
#SBATCH --job-name=bench_5m_allwalk
#SBATCH --partition=gpu-debug
#SBATCH --nodelist=gpu9
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=supervisor_evaluation/outputs/logs/bench_5m_allwalk_%j.log
# Timing for the five-method variant on the node of the thesis benchmark (gpu9, V100-PCIE-32GB),
# so the times are measured on the same host as before.
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
export EVAL_VARIANT=five_allwalk
O=supervisor_evaluation/outputs/full_5m_allwalk_20261011
nvidia-smi --query-gpu=name --format=csv,noheader; hostname
python3 -u supervisor_evaluation/evaluate.py benchmark --data-root /scratch/work/zhangx29/data --output-dir $O
