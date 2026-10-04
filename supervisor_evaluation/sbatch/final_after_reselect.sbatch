#!/bin/bash
#SBATCH --job-name=final_eval
#SBATCH --partition=gpu-h200-141g-short,gpu-a100-80g,gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=supervisor_evaluation/outputs/logs/final_after_reselect_%j.log
#
# The single test evaluation after the validation re-selection
# (supervisor_evaluation/RESELECTION_PROTOCOL.md). Only the EnKF configuration changed
# (candidate E2); the other five final models are unchanged. Reruns the EnKF on the test
# days, the unified uncertainty evaluation, the inference benchmark, the example frames,
# the error breakdown, and the original-vs-final EnKF comparison.
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
D=/scratch/work/zhangx29/data
O=supervisor_evaluation/outputs/full
python3 -u supervisor_evaluation/evaluate.py full --methods enkf --data-root $D --output-dir $O
python3 -u supervisor_evaluation/evaluate.py uncertainty --data-root $D --output-dir $O
python3 -u supervisor_evaluation/evaluate.py benchmark --data-root $D --output-dir $O
python3 -u supervisor_evaluation/evaluate.py images --n-images 100 --data-root $D --output-dir $O
python3 -u supervisor_evaluation/breakdown.py --data-root $D --output-dir $O
python3 -u -m methods.enkf.checks.compare_original_final
