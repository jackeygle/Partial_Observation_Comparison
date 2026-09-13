#!/bin/bash
#SBATCH --job-name=varnet_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=methods/varnet/runs/slurm_%x_%j.out
# Usage: sbatch methods/varnet/sbatch/submit_eval.sbatch checks/<script>.py [args...]
#   (eval_test_days.py was removed on 2026-09-09; compare/compare5.py computes
#    the same quantities for every method in one pass)
source /etc/profile 2>/dev/null
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || { echo "submit from the repo root" >&2; exit 1; }
cd "$THESIS_ROOT/methods/varnet"
nvidia-smi --query-gpu=name --format=csv,noheader
python3 -u "$@"
