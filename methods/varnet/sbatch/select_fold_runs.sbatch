#!/bin/bash
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:28:00
#SBATCH --output=methods/varnet/runs/slurm_%x_%j.out
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh"
cd "$THESIS_ROOT"
for R in runs/varnet_mse5_h96_pdfold_s3 runs/varnet_aughead_obs_h96_fold_s0; do
    echo "########## $R"
    python3 -u -m methods.varnet.checks.select_checkpoint --run-dir "$R" --split valid || echo "[fail] $R"
done
