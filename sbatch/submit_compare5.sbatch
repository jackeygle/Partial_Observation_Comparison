#!/bin/bash
#SBATCH --job-name=compare5
#SBATCH --partition=gpu-h200-141g-short,gpu-a100-80g,gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=0-02:00:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/compare/results/slurm_%x_%j.out
# Per-channel comparison across five methods, four cell-scope conventions side by side.
#
# This job script lives in the top-level sbatch/, not under any one method's sbatch/:
# it scores all five methods, and filing it under any single one would be misleading
# (compare4 used to live in senseiver_crowd/, so its output landed in Senseiver's
# check_outputs and looked like a Senseiver metric).
#
# 128G memory: the EnKF's est_*.npz is about 500MB per day, plus 12 solvers' worth of
# whole-day (T,C,H,W) intermediates.
#
# Usage:
#   sbatch sbatch/submit_compare5.sbatch                          # full run
#   sbatch sbatch/submit_compare5.sbatch --days 1 --ensembles ""  # smoke test
set -euo pipefail
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd "$THESIS_ROOT"
mkdir -p compare/results
python3 -u -m compare.compare5 "$@"
