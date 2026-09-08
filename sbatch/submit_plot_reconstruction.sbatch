#!/bin/bash
#SBATCH --job-name=plot_recon
#SBATCH --partition=gpu-h200-141g-short,gpu-a100-80g,gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/compare/results/slurm_%x_%j.out
# One reconstructed field, side by side across all four methods (or any subset).
# Lives in the top-level sbatch/, same reasoning as submit_compare5.sbatch: this
# touches more than one method, so it isn't filed under any single one's sbatch/.
#
# Usage:
#   sbatch sbatch/submit_plot_reconstruction.sbatch --day atc-20130811
#   sbatch sbatch/submit_plot_reconstruction.sbatch --methods dincae,senseiver --day atc-20130818
set -euo pipefail
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd "$THESIS_ROOT"
mkdir -p compare/results
python3 -u -m compare.plot_reconstruction "$@"
