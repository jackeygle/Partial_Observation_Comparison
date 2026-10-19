#!/bin/bash
#SBATCH --job-name=smoke_augwink3
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=methods/varnet/runs/slurm_%x_%j.out
# Smoke test for submit_aughead_wink3_chain.sbatch: same command, 2 training days, one epoch at
# the final depth K=20 (no NaN, timing), then batch independence of mean and variance on the
# smoke checkpoint. Separate run dir.
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh"
cd "$THESIS_ROOT/methods/varnet"
nvidia-smi --query-gpu=name --format=csv,noheader
START=$(date +%s)
python3 -u -m methods.varnet.train \
    --obs-every-k 1 --hidden 96 --kt 3 --init-seed 0 --data-seed 0 \
    --days 2 --dT 200 --epochs 1 --batch 32 --amp \
    --loss nll --nll-beta 0.0 --var-eps 1e-6 --augmented-var --obs-nll --var-head \
    --iter-schedule '0:20:3e-4' --clip-grad 1.0 \
    --trajectory-mode per_day --fold-axes ct --normg-scope window \
    --outdir runs/smoke_aughead_wink3_dev
echo "[smoke] wall seconds: $(( $(date +%s) - START ))"
cd "$THESIS_ROOT"
python3 -u -m supervisor_evaluation.checks.batch_independence_var \
    --ckpt methods/varnet/runs/smoke_aughead_wink3_dev/varnet_last.pt
