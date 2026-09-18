#!/bin/bash
#SBATCH --job-name=sv_hist_age
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=methods/senseiver/check_outputs/temporal/slurm_%x_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || { echo "submit from repo root" >&2; exit 1; }
cd "$THESIS_ROOT"
python3 -u -m methods.senseiver.checks.diagnose_history_age "$@"
