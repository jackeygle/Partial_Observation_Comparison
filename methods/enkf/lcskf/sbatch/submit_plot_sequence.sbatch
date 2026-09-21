#!/bin/bash
#SBATCH --job-name=plotseq
#SBATCH --partition=batch-csl
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=methods/enkf/runs/slurm_%x_%j.out
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || { echo "submit from the repo root" >&2; exit 1; }
srun python3 -u -m compare.plot_reconstruction_sequence "$@"
