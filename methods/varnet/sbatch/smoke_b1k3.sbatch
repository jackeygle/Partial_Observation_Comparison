#!/bin/bash
#SBATCH --job-name=smoke_b1k3
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=methods/varnet/runs/slurm_%x_%j.out
# Smoke test for submit_b1k3_chain.sbatch: same command, 2 training days, one epoch at the
# final depth K=20, to check it runs and to time one window per step. Separate run dir.
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh"
cd "$THESIS_ROOT/methods/varnet"
nvidia-smi --query-gpu=name --format=csv,noheader
START=$(date +%s)
python3 -u -m methods.varnet.train \
    --obs-every-k 1 --hidden 96 --kt 3 --init-seed 3 --data-seed 0 \
    --days 2 --dT 200 --epochs 1 --batch 1 --amp --loss supervised \
    --iter-schedule '0:20:3e-4' --clip-grad 1.0 \
    --trajectory-mode per_day --fold-axes ct \
    --outdir runs/smoke_b1k3_dev
echo "[smoke] wall seconds: $(( $(date +%s) - START ))"
