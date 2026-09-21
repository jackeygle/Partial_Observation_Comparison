#!/bin/bash
#SBATCH --job-name=lcskf_bg
#SBATCH --partition=gpu-v100-16g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=methods/enkf/runs/slurm_%x_%j.out
#
# Export the filter's own analyses as a training background (one round of DAgger).
#
# Deliberately on the cheapest, most plentiful GPU: batch size is 1, the grid is
# 4x36x12 and the two nets are 1.4M parameters together, so the 39k-frame loop is
# latency-bound on tiny kernels and uses under 1 GB. A big card buys nothing and
# queues longer.
# One job per day keeps each well inside the time limit:
#   for d in $(python3 -c "..."); do sbatch ... --split train --only $d; done
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || { echo "submit from the repo root" >&2; exit 1; }
cd "$THESIS_ROOT/methods/enkf"
nvidia-smi --query-gpu=name --format=csv,noheader
srun python3 -u -m methods.enkf.lcskf.checks.export_backgrounds "$@"
