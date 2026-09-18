#!/bin/bash
#SBATCH --job-name=senseiver_eval
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=0-00:30:00
#SBATCH --output=methods/senseiver/runs/slurm_%x_%j.out
# Usage: sbatch methods/senseiver/sbatch/submit_eval.sbatch [evaluate.py args]
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || { echo "submit from the repo root" >&2; exit 1; }
cd "$THESIS_ROOT/methods/senseiver"
python3 -u -m methods.senseiver.checks.evaluate "$@"
