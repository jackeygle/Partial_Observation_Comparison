#!/bin/bash
#SBATCH --job-name=sel_dincae_varnone
#SBATCH --partition=gpu-v100-32g,gpu-v100-16g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=supervisor_evaluation/outputs/reselect_all_walkable_20261010/%x_%j.out
# Validation selection (all walkable cells) of the DINCAE run without the log1p of `var`.
# Usage: sbatch supervisor_evaluation/sbatch/reselect_dincae_varnone.sbatch <k>/<n>
source /etc/profile 2>/dev/null
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || exit 1
set -e
cd "$THESIS_ROOT"
export DINCAE_VAR_TRANSFORM=none
python3 -u -m supervisor_evaluation.checks.reselect_all_walkable dincae_varnone --shard "$1"
