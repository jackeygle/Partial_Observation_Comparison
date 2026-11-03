#!/bin/bash
#SBATCH --job-name=cov_enkf
#SBATCH --partition=gpu-v100-32g,gpu-v100-16g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=supervisor_evaluation/outputs/logs/%x_%j.log
# Rerun of the final EnKF for the members' coverage. Usage: sbatch coverage_enkf_all92.sbatch k/n
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
export DATA_VARIANT=all92 EVAL_VARIANT=six_allwalk_all92 DINCAE_VAR_TRANSFORM=none
O=supervisor_evaluation/outputs/dev/coverage_all92_20261031
nvidia-smi --query-gpu=name --format=csv,noheader; hostname
python3 -u -m supervisor_evaluation.checks.coverage_run enkf --out $O --shard "$1"
echo "[done] coverage enkf $1"
