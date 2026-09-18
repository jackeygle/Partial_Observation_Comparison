#!/bin/bash
#SBATCH --job-name=cov5_seq
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=methods/enkf/runs/slurm_%x_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || { echo "submit from the repo root" >&2; exit 1; }
cd "$THESIS_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -u -m methods.enkf.checks.eval_pedpred5_sequential "$@"
