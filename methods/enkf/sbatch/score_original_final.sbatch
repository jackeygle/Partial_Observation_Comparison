#!/bin/bash
#SBATCH --job-name=enkf_orig_vs_final
#SBATCH --partition=batch-csl
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=methods/enkf/check_outputs/original_vs_final/logs/score_%j.out
# Scores the original and the final EnKF identically (compare_original_final.py).
# Numpy only; needs more memory than a login node allows.
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
python3 -u -m methods.enkf.checks.compare_original_final "$@"
