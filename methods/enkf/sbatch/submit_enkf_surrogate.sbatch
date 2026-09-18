#!/bin/bash
#SBATCH --job-name=enkf_surr
#SBATCH --partition=batch-csl,batch-hsw,batch-milan
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=methods/enkf/check_outputs/%x_%j.out
#
# One arm x one day of checks/run_enkf_surrogate.py. CPU only, like the shipped EnKF.
# Usage, from the repo root:
#   sbatch --job-name=enkfv_E2_0616 methods/enkf/sbatch/submit_enkf_surrogate.sbatch --arms E2 --only atc-20130616
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || { echo "submit from the repo root" >&2; exit 1; }
cd "$THESIS_ROOT/methods/enkf"
python3 -u -m methods.enkf.checks.run_enkf_surrogate "$@"
