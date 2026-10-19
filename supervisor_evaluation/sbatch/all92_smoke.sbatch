#!/bin/bash
#SBATCH --job-name=all92_smoke
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=supervisor_evaluation/outputs/logs/all92_smoke_%j.log
# Route 2 (six methods, all 92 days, gap mask on): package the selected checkpoints, verify
# them, and run every method on the first 512-600 frames of the first test day (not a result).
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
export DATA_VARIANT=all92 EVAL_VARIANT=six_allwalk_all92 DINCAE_VAR_TRANSFORM=none
D=/scratch/work/zhangx29/data_all92
O=supervisor_evaluation/outputs/full_6m_all92
E=supervisor_evaluation/evaluate.py
nvidia-smi --query-gpu=name --format=csv,noheader; hostname
python3 -u $E prepare
python3 -u $E smoke --data-root $D --output-dir supervisor_evaluation/outputs/dev/smoke_6m_all92
echo "[done] all92 smoke"
