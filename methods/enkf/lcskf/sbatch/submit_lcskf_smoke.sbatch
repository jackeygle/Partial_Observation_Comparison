#!/bin/bash
#SBATCH --job-name=lrkf_smoke
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=methods/enkf/runs/slurm_%x_%j.out
#
# Fast end-to-end CUDA check before submitting the full rank sweep.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || { echo "submit from the repo root" >&2; exit 1; }
cd "$THESIS_ROOT/methods/enkf"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -u -m methods.enkf.lcskf.checks.check_kalman
python3 -u -m methods.enkf.lcskf.train \
  --rank 16 --width 32 --days 1 --max-frames 512 --epochs 1 --batch 32 \
  --eval-pairs 128 --out runs/lowrank_smoke_gpu "$@"
