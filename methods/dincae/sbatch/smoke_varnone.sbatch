#!/bin/bash
#SBATCH --job-name=smoke_dincae_varnone
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=methods/dincae/runs/slurm_%x_%j.out
# Smoke test of the DINCAE ablation without the log1p of `var`: builds its own statistics,
# trains one epoch on two days with the final recipe, and samples GPU utilisation.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh"
cd "$THESIS_ROOT"
export DINCAE_VAR_TRANSFORM=none
nvidia-smi --query-gpu=name --format=csv,noheader
[ -f methods/dincae/artifacts/state_stats_none.npz ] || python3 -u -m methods.dincae.state
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -l 10 > methods/dincae/runs/smoke_varnone_gpu_$SLURM_JOB_ID.csv &
SMI=$!
START=$(date +%s)
python3 -u -m methods.dincae.train --amp --epochs 1 --days 2 --save-every 1 --save-from 0 \
    --trajectory-mode per_day --full-field-loss --out methods/dincae/runs/smoke_varnone_dev
echo "[smoke] train wall seconds: $(( $(date +%s) - START ))"
kill $SMI
