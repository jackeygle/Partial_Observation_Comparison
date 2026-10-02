#!/bin/bash
#SBATCH --job-name=sv_select
#SBATCH --partition=gpu-h200-141g-short,gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=methods/senseiver/runs/slurm_%x_%j.out
#
# Choose a Senseiver run's epoch on the whole validation split.
# Usage (from the repo root): sbatch methods/senseiver/sbatch/select_checkpoint.sbatch <run-dir>
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || { echo "submit from the repo root" >&2; exit 1; }
cd "$THESIS_ROOT"
python3 -u -m methods.senseiver.checks.select_checkpoint --run-dir "$@"
