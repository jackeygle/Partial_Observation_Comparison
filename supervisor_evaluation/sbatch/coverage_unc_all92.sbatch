#!/bin/bash
#SBATCH --job-name=cov_unc
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=supervisor_evaluation/outputs/logs/%x_%j.log
# After the coverage_enkf jobs: check the rerun against the reported run, then score coverage
# with the reported EnKF exports (read-only) on the protocol of the CRPS.
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
export DATA_VARIANT=all92 EVAL_VARIANT=six_allwalk_all92 DINCAE_VAR_TRANSFORM=none
O=supervisor_evaluation/outputs/dev/coverage_all92_20261031
nvidia-smi --query-gpu=name --format=csv,noheader; hostname
python3 -u -m supervisor_evaluation.checks.coverage_run check --out $O
python3 -u -m supervisor_evaluation.checks.coverage_run unc --out $O
echo "[done] coverage unc"
