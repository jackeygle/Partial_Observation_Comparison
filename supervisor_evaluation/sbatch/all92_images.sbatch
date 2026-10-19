#!/bin/bash
#SBATCH --job-name=all92_images
#SBATCH --partition=gpu-debug,gpu-v100-32g
#SBATCH --gres=gpu:1,min-vram:32G   # any GPU with >= 32 GB (user, 2026-10-15)
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=supervisor_evaluation/outputs/logs/all92_images_%j.log
# Route 2 comparison images (about 20 min) on a gpu-debug A100: a V100-16G ran out of memory
# here (Senseiver-G frame batches), V100-32G queue was long (user, 2026-10-15).
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
export DATA_VARIANT=all92 EVAL_VARIANT=six_allwalk_all92 DINCAE_VAR_TRANSFORM=none
D=/scratch/work/zhangx29/data_all92
O=supervisor_evaluation/outputs/full_6m_all92
E=supervisor_evaluation/evaluate.py
nvidia-smi --query-gpu=name --format=csv,noheader; hostname
python3 -u $E images --data-root $D --output-dir $O
echo "[done] all92 images"
