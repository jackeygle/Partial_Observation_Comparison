#!/bin/bash
#SBATCH --job-name=sel_all92
#SBATCH --partition=gpu-v100-32g,gpu-v100-16g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=supervisor_evaluation/outputs/reselect_all_walkable_all92_gapmask/%x_%j.out
# Route 2 validation selection on all walkable cells of the 14 validation days (gap mask on:
# tracker-gap blocks excluded, output dir reselect_all_walkable_all92_gapmask).
# Usage: sbatch --job-name=sel_<method> supervisor_evaluation/sbatch/reselect_all92.sbatch <method> <k>/<n>
source /etc/profile 2>/dev/null
export DATA_VARIANT=all92 DINCAE_VAR_TRANSFORM=none
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || exit 1
set -e
cd "$THESIS_ROOT"
python3 -u -m supervisor_evaluation.checks.reselect_all_walkable "$1" --shard "$2"
