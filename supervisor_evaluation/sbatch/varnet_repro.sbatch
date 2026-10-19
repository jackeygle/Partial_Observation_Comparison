#!/bin/bash
#SBATCH --job-name=varnet_repro
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=supervisor_evaluation/outputs/logs/varnet_repro_%j.log
# Reproducibility check of the reported 4DVarNet test accuracy: the same command as
# evaluate.py's varnet_mse stage (compare.compare5, 7 test days), with a chosen window
# batch. Usage: sbatch varnet_repro.sbatch <varnet-batch> <tag>. Output under outputs/dev/.
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
VB=$1; TAG=$2
O=supervisor_evaluation/outputs/dev/varnet_repro; mkdir -p $O
M=supervisor_evaluation/models
python3 -u -m compare.compare5 --senseiver $M/senseiver_A_pd.pt \
  --varnet-checkpoints MSE=$M/varnet_mse_fold_s3.pt --arms "" --no-with-ensemble \
  --days 0 --frames 0 --varnet-batch $VB --out $O/varnet_b${VB}_${TAG}.json
nvidia-smi --query-gpu=name --format=csv,noheader
