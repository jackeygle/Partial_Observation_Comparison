#!/bin/bash
#SBATCH --job-name=eval_5m_gdincae
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=supervisor_evaluation/outputs/logs/eval_5m_gdincae_%j.log
# Five-method test evaluation with the DINCAE trained without the log1p of the velocity
# variance (DINCAE_VAR_TRANSFORM=none; all four channels Gaussian), checkpoint selected on all
# walkable validation cells. Only DINCAE is re-run: the Senseiver and 4DVarNet accuracy files
# and the timing files are copied from outputs/full_5m_allwalk_20261011 (same checkpoints;
# DINCAE's architecture and input size are unchanged), and the EnKF run and exports are the
# symlinked ones of outputs/full. Uncertainty, images and the error breakdown are recomputed.
# Writes only to outputs/full_5m_allwalk_gaussdincae_20261011.
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
export EVAL_VARIANT=five_allwalk
export DINCAE_VAR_TRANSFORM=none
D=/scratch/work/zhangx29/data
O=supervisor_evaluation/outputs/full_5m_allwalk_gaussdincae_20261011
E=supervisor_evaluation/evaluate.py
nvidia-smi --query-gpu=name --format=csv,noheader
python3 -u $E full --methods dincae --data-root $D --output-dir $O
python3 -u $E uncertainty --data-root $D --output-dir $O
python3 -u $E images --data-root $D --output-dir $O
python3 -u supervisor_evaluation/breakdown.py --data-root $D --output-dir $O
python3 -u $E collect --output-dir $O
python3 -u $E plot --output-dir $O
python3 -u -m supervisor_evaluation.checks.accuracy_all_walkable $O
echo "[done] five-method evaluation, Gaussian DINCAE"
