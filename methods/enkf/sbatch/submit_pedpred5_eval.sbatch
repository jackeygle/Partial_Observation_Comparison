#!/bin/bash
#SBATCH --job-name=ped5_eval
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=methods/enkf/runs/slurm_%x_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || { echo "submit from the repo root" >&2; exit 1; }
cd "$THESIS_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -u -m methods.enkf.checks.eval_pedpred5 "$@"
