#!/bin/bash
#SBATCH --job-name=surrogate
#SBATCH --partition=gpu-h200-141g-short,gpu-a100-80g,gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=methods/enkf/runs/slurm_%x_%j.out
#
# One member of the PedPred3 surrogate deep ensemble (methods/enkf/lcskf/dynamics/train.py).
# Usage, from the repo root:
#   sbatch --job-name=surr_mean_s0 methods/enkf/lcskf/sbatch/submit_dynamics.sbatch --seed 0
# The deep-ensemble sigma arm was removed in 9825c81 with its checkpoints.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || { echo "submit from the repo root" >&2; exit 1; }
cd "$THESIS_ROOT/methods/enkf"
nvidia-smi --query-gpu=name --format=csv,noheader
python3 -u -m methods.enkf.lcskf.dynamics.train "$@"
