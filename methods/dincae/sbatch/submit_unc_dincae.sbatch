#!/bin/bash
#SBATCH --job-name=uncdinc
#SBATCH --partition=gpu-h200-141g-short,gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/methods/dincae/check_outputs/uncdinc_%j.out
# DINCAE's native sigma^2 -- the only method besides 4DVarNet that predicts variance,
# yet it had always been missing from table B, because its own evaluate.py only reports
# the geoscience-style calibration table, not CRPS / spread-skill / coverage.
# Two inference passes: the constant-sigma baseline's sigma is each slice's own RMSE,
# which is only known after the first pass finishes.
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project/methods/dincae
# Defaults to the PUBLISHED configuration (single epoch-70 checkpoint), which is
# what uncertainty_dincae.json holds. Without this, evaluate.load_models matches
# all 16 ckpt_*.pt and refuses to average silently, so the entry point would fail.
CKPT=/scratch/work/zhangx29/Thesis_Project/methods/dincae/runs/dincae_full/ckpt_00070.pt
srun python3 -u -m methods.dincae.checks.eval_uncertainty_dincae --ckpt-glob "$CKPT" "$@"
