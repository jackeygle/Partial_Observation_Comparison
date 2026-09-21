#!/bin/bash
#SBATCH --job-name=lcskf_seqcmp
#SBATCH --partition=gpu-v100-16g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=methods/enkf/runs/slurm_%x_%j.out
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || { echo "submit from the repo root" >&2; exit 1; }
cd "$THESIS_ROOT/methods/enkf"
nvidia-smi --query-gpu=name --format=csv,noheader
srun python3 -u -m methods.enkf.lcskf.checks.eval_sequential "$@"
