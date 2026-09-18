#!/bin/bash
#SBATCH --job-name=smoke_augobs
#SBATCH --partition=gpu-h200-141g-short,gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --exclude=gpu51
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/methods/varnet/runs/slurm_%x_%j.out
# Obs-term NLL: exact-math checks, then aug0 vs augobs stepped side by side at FULL scale
# (hidden 96, dT 200, batch 32, AMP) and n_iter 20 -- the longest unroll, where an exploding
# e^{-s} on observed cells would show first.
source /etc/profile 2>/dev/null
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project

nvidia-smi --query-gpu=name --format=csv,noheader
python3 -u -m methods.varnet.scratch.check_obs_nll || exit 1

cd methods/varnet
SCRATCH=/scratch/work/zhangx29/Thesis_Project/methods/varnet/scratch/smoke_augobs
rm -rf "$SCRATCH"; mkdir -p "$SCRATCH"

run () {   # run <tag> <extra flags...>
    TAG=$1; shift
    echo; echo "=================== $TAG ==================="
    python3 -u -m methods.varnet.train \
        --obs-every-k 1 --init-seed 0 --data-seed 0 \
        --hidden 96 --kt 5 \
        --days 2 --dT 200 --epochs 1 --batch 32 --amp --steps 30 \
        --iter-schedule '0:20:1e-3' --clip-grad 1.0 \
        --loss nll --nll-beta 0.0 --var-eps 1e-6 --augmented-var \
        --outdir "$SCRATCH/$TAG" "$@" \
      && echo "[smoke] $TAG OK" || { echo "[smoke] $TAG FAILED"; exit 1; }
}

run aug0
run augobs --obs-nll
echo; echo "[smoke] both arms built and stepped"
