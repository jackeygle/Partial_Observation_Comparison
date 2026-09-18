#!/bin/bash
#SBATCH --job-name=smoke_h96
#SBATCH --partition=gpu-h200-141g-short,gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:40:00
#SBATCH --exclude=gpu51
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/methods/varnet/runs/slurm_%x_%j.out
# Smoke test for the hidden=96 / kt=5 retrain of the three uncertainty arms.
#
# The point is NOT accuracy -- 40 steps decide nothing. It is that each of the three
# losses BUILDS and STEPS at the new prior width before 15 chains are queued. The aug
# arm is the reason this exists: its chain crash-looped through 251 generations on
# 2026-09-02/03 leaving no logs, and that was at hidden=32. Its solver input width is
# 2*C*dT + lstm_hidden, so it is the one most likely to break on a shape change.
source /etc/profile 2>/dev/null
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project/methods/varnet

nvidia-smi --query-gpu=name --format=csv,noheader
SCRATCH=/scratch/work/zhangx29/Thesis_Project/methods/varnet/scratch/smoke_h96
rm -rf "$SCRATCH"; mkdir -p "$SCRATCH"

run () {   # run <tag> <extra flags...>
    TAG=$1; shift
    echo; echo "=================== $TAG ==================="
    python3 -u -m methods.varnet.train \
        --obs-every-k 1 --init-seed 0 --data-seed 0 \
        --hidden 96 --kt 5 \
        --days 2 --dT 200 --epochs 1 --batch 32 --amp --steps 40 \
        --iter-schedule '0:5:1e-3' --clip-grad 1.0 \
        --outdir "$SCRATCH/$TAG" "$@" \
      && echo "[smoke] $TAG OK" || { echo "[smoke] $TAG FAILED"; exit 1; }
}

run mse5 --loss supervised
run vsb0 --loss nll --nll-beta 0.0 --var-eps 1e-6
run aug0 --loss nll --nll-beta 0.0 --var-eps 1e-6 --augmented-var

echo; echo "[smoke] all three arms built and stepped at hidden=96 kt=5"
