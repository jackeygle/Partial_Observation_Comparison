#!/bin/bash
#SBATCH --job-name=eval_5m_allwalk
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=supervisor_evaluation/outputs/logs/eval_5m_allwalk_%j.log
# Five-method test evaluation (EVAL_VARIANT=five_allwalk): checkpoints selected on all walkable
# validation cells (Senseiver-A 96, Senseiver-G 54, DINCAE 130), the window-normalised kt3
# 4DVarNet (epoch 80) and the unchanged EnKF, whose test run and exports are reused from
# outputs/full through symlinks (same PedPred3, bank and configuration E2). Writes only to
# outputs/full_5m_allwalk_20261011; outputs/full is not touched.
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
export EVAL_VARIANT=five_allwalk
D=/scratch/work/zhangx29/data
O=supervisor_evaluation/outputs/full_5m_allwalk_20261011
E=supervisor_evaluation/evaluate.py
nvidia-smi --query-gpu=name --format=csv,noheader
python3 -u $E full --methods senseiver_a senseiver_g dincae varnet_mse --data-root $D --output-dir $O
python3 -u $E uncertainty --data-root $D --output-dir $O
python3 -u $E images --data-root $D --output-dir $O
python3 -u supervisor_evaluation/breakdown.py --data-root $D --output-dir $O
python3 -u $E benchmark --data-root $D --output-dir $O
python3 -u $E collect --output-dir $O
python3 -u $E plot --output-dir $O
echo "[done] five-method all-walkable evaluation"
