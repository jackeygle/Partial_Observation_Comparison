#!/bin/bash
#SBATCH --job-name=eval_6m_allwalk
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=supervisor_evaluation/outputs/logs/eval_6m_allwalk_%j.log
# Six-method test evaluation (EVAL_VARIANT=six_allwalk): the five models of
# outputs/full_5m_allwalk_gaussdincae_20261011, whose accuracy files are copied into the output
# directory beforehand (same checkpoints), plus the retrained aug. var. 4DVarNet. Only the
# aug. var. accuracy is computed anew; uncertainty (three probabilistic methods), images and the
# error breakdown are recomputed for all. Timing runs separately on gpu9 (bench_6m_allwalk).
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
export EVAL_VARIANT=six_allwalk DINCAE_VAR_TRANSFORM=none
D=/scratch/work/zhangx29/data
O=supervisor_evaluation/outputs/full_6m_allwalk
E=supervisor_evaluation/evaluate.py
nvidia-smi --query-gpu=name --format=csv,noheader
python3 -u $E full --methods varnet_aughead --data-root $D --output-dir $O
python3 -u $E uncertainty --data-root $D --output-dir $O
python3 -u $E images --data-root $D --output-dir $O
python3 -u supervisor_evaluation/breakdown.py --data-root $D --output-dir $O
python3 -u $E collect --output-dir $O
python3 -u $E plot --output-dir $O
python3 -u -m supervisor_evaluation.checks.accuracy_all_walkable $O
echo "[done] six-method evaluation"
