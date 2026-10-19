#!/bin/bash
#SBATCH --job-name=all92_full
#SBATCH --partition=gpu-v100-32g,gpu-v100-16g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=supervisor_evaluation/outputs/logs/%x_%j.log
# Route 2 test evaluation of ONE method on all 15 test days (raw outputs only; the summary job
# collects). Usage: sbatch --job-name=all92_<m> all92_full_method.sbatch <method> [enkf k/n]
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
export DATA_VARIANT=all92 EVAL_VARIANT=six_allwalk_all92 DINCAE_VAR_TRANSFORM=none
D=/scratch/work/zhangx29/data_all92
O=supervisor_evaluation/outputs/full_6m_all92
E=supervisor_evaluation/evaluate.py
nvidia-smi --query-gpu=name --format=csv,noheader; hostname
if [ -n "${2:-}" ]; then export EVAL_ENKF_SHARD=$2; fi
EVAL_NO_SUMMARY=1 python3 -u $E full --methods $1 --data-root $D --output-dir $O
echo "[done] $1 ${2:-}"
