#!/bin/bash
#SBATCH --job-name=smoke_wink3_dt100
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1,min-vram:32g
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=methods/varnet/runs/slurm_%x_%j.out
# Smoke test for submit_wink3_chain.sbatch: same command, 2 training days, one epoch at the
# final depth K=20, to check it trains (no NaN) and to time it. Separate run dir.
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh"
cd "$THESIS_ROOT/methods/varnet"
nvidia-smi --query-gpu=name --format=csv,noheader
START=$(date +%s)
python3 -u -m methods.varnet.train \
    --obs-every-k 1 --hidden 96 --kt 3 --init-seed 3 --data-seed 0 \
    --days 2 --dT 100 --epochs 1 --batch 32 --amp --loss supervised \
    --iter-schedule '0:20:3e-4' --clip-grad 1.0 \
    --trajectory-mode per_day --fold-axes ct --normg-scope window \
    --outdir runs/smoke_wink3_dt100_dev
echo "[smoke] wall seconds: $(( $(date +%s) - START ))"
