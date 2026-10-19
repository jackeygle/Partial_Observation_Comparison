#!/bin/bash
#SBATCH --job-name=batch_indep
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=supervisor_evaluation/outputs/logs/batch_indep_%j.log
# Batch 1 vs 16 on one test day, for the new (window-normalised) and the old 4DVarNet.
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
python3 -u -m supervisor_evaluation.checks.batch_independence \
  --ckpt methods/varnet/runs/varnet_mse5_h96k3_win_pdfold_s3/ckpt_00080.pt
python3 -u -m supervisor_evaluation.checks.batch_independence \
  --ckpt supervisor_evaluation/models/varnet_mse_fold_s3.pt
