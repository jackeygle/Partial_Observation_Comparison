#!/bin/bash
#SBATCH --job-name=dincae_eval
#SBATCH --partition=gpu-h200-141g-short,gpu-a100-80g,gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/methods/dincae/runs/slurm_%x_%j.out

source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project/methods/dincae
# Defaults to the PUBLISHED configuration: the single epoch-70 checkpoint every
# reported DINCAE number uses. Without an explicit --ckpt-glob, evaluate.py
# matches all 16 ckpt_*.pt and now refuses to average them silently, so state
# the intent here rather than leaving the entry point broken.
# To reproduce the reference implementation's averaging instead:
#   sbatch sbatch/submit_eval.sbatch --average-checkpoints --ckpt-glob ''
ROOT=/scratch/work/zhangx29/Thesis_Project
CKPT="$ROOT/methods/dincae/runs/dincae_full/ckpt_00070.pt"
# --out matches the checkpoint: the published single-checkpoint result belongs in
# eval_single_00070/, NOT in eval/, which holds the 16-checkpoint average. Writing
# both to eval/ would silently overwrite one result with the other under the same
# filename.
python3 -u -m methods.dincae.checks.evaluate --ckpt-glob "$CKPT" \
        --out "$ROOT/methods/dincae/check_outputs/eval_single_00070" "$@"
