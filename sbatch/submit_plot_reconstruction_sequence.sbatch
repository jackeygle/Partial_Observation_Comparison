#!/bin/bash
#SBATCH --job-name=plot_recon_seq
#SBATCH --partition=gpu-h200-141g-short,gpu-a100-80g,gpu-v100-32g,gpu-v100-16g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=compare/results/slurm_%x_%j.out
# N consecutive frames, all 4 methods, one PNG per frame. Model inference is
# batched once per method (minutes); the N savefig calls afterward dominate
# wall time -- budget roughly this: --time above covers ~5000 frames.
#
# Usage:
#   sbatch sbatch/submit_plot_reconstruction_sequence.sbatch --day atc-20130811 --n 5000
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || { echo "submit from the repo root" >&2; exit 1; }
cd "$THESIS_ROOT"
python3 -u -m compare.plot_reconstruction_sequence "$@"
