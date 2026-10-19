#!/bin/bash
#SBATCH --job-name=fold_diagnostics
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:25:00
#SBATCH --output=supervisor_evaluation/outputs/logs/fold_diagnostics_%j.log
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
DIAGNOSTIC_OUTPUT=${1:?pass a separate diagnostic output directory}
shift
python3 -u -m supervisor_evaluation.checks.fold_diagnostics \
    --probe --output-dir "$DIAGNOSTIC_OUTPUT" "$@"
