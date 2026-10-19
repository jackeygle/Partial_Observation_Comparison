#!/bin/bash
#SBATCH --job-name=all92_breakdown
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=supervisor_evaluation/outputs/logs/all92_breakdown_%j.log
# Route 2 error breakdown, then the tables, figures and all-walkable table (uncertainty was done
# by the first summary job, images by all92_images).
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
export DATA_VARIANT=all92 EVAL_VARIANT=six_allwalk_all92 DINCAE_VAR_TRANSFORM=none
D=/scratch/work/zhangx29/data_all92
O=supervisor_evaluation/outputs/full_6m_all92
E=supervisor_evaluation/evaluate.py
nvidia-smi --query-gpu=name --format=csv,noheader; hostname
python3 -u supervisor_evaluation/breakdown.py --data-root $D --output-dir $O
python3 -u $E collect --output-dir $O
python3 -u $E plot --output-dir $O
python3 -u -m supervisor_evaluation.checks.accuracy_all_walkable $O
echo "[done] all92 breakdown + tables"
