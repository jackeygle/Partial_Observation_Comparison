#!/bin/bash
#SBATCH --job-name=sparsifaug
#SBATCH --partition=gpu-h200-141g-m
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=00:40:00
#SBATCH --exclude=gpu51
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/methods/varnet/check_outputs/sparsifaug_%j.out
# The point of the augmented-state design: does iterating log sigma^2 inside the solver, against
# a cost term that actually depends on it, give the variance the per-cell discriminative power
# the read-out never had? The read-out version scores 0.036 / 0.035 / 0.002 / 0.040 -- close to
# a random ordering. The ceiling for this route is the prior residual's own R^2 against the true
# error, 0.38 (vx) / 0.35 (vy) / 0.11 (density), since that is what the new cost term makes
# sigma^2 chase.
#
# Measured on varnet_best.pt, which is the epoch-27 model: blind MSE bottomed there and has
# drifted slightly worse over the 100 epochs since, so waiting for epoch 150 would not change
# which checkpoint this is.
export AUDIT_FMT="runs/varnet_aug0_s{}"
export AUDIT_MEMBERS="0"
export AUDIT_OUT="sparsification_aug0.json"
cd /scratch/work/zhangx29/Thesis_Project
module load scicomp-python-env
srun python3 -m methods.varnet.checks.diag_sparsification
