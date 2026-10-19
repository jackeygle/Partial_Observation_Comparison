#!/bin/bash
#SBATCH --job-name=all92_finish
#SBATCH --partition=gpu-debug,gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=supervisor_evaluation/outputs/logs/all92_finish_%j.log
# Route 2: merge the breakdown parts (with its accuracy.csv consistency check), then tables,
# figures and the all-walkable table. Usage: sbatch all92_finish.sbatch <n parts>
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
export DATA_VARIANT=all92 EVAL_VARIANT=six_allwalk_all92 DINCAE_VAR_TRANSFORM=none
D=/scratch/work/zhangx29/data_all92
O=supervisor_evaluation/outputs/full_6m_all92
E=supervisor_evaluation/evaluate.py
nvidia-smi --query-gpu=name --format=csv,noheader; hostname
python3 -u supervisor_evaluation/breakdown.py --data-root $D --output-dir $O --merge $1
python3 -u $E collect --output-dir $O
python3 -u $E plot --output-dir $O
python3 -u -m supervisor_evaluation.checks.accuracy_all_walkable $O
echo "[done] all92 finish"
