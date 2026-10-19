#!/bin/bash
#SBATCH --job-name=unc_allwalk_5m
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=supervisor_evaluation/outputs/logs/unc_allwalk_5m_%j.log
# Uncertainty of DINCAE and the EnKF on all walkable cells as well as on the blind cells
# (same frames), for the five-method thesis. Overwrites raw/uncertainty_unified.json and the
# uncertainty CSVs of the Gaussian-DINCAE evaluation; the blind-only versions are kept in
# _blind_only_backup/ and the blind values must come out unchanged.
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
export EVAL_VARIANT=five_allwalk
export DINCAE_VAR_TRANSFORM=none
O=supervisor_evaluation/outputs/full_5m_allwalk_gaussdincae_20261011
nvidia-smi --query-gpu=name --format=csv,noheader
python3 -u supervisor_evaluation/evaluate.py uncertainty --data-root /scratch/work/zhangx29/data --output-dir $O
echo "[done] uncertainty on all walkable and blind cells"
