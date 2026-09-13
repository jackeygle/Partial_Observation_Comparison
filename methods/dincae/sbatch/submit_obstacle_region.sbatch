#!/bin/bash
#SBATCH --job-name=dincae_obstacle
#SBATCH --partition=gpu-h200-141g-short,gpu-a100-80g,gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=methods/dincae/runs/slurm_%x_%j.out

source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || { echo "submit from the repo root" >&2; exit 1; }
cd "$THESIS_ROOT/methods/dincae"
python3 -u -m methods.dincae.checks.measure_obstacle_region "$@"
