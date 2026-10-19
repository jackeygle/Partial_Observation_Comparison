#!/bin/bash
#SBATCH --job-name=diag_pedpred
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/work/zhangx29/data_all92/logs/diag_pedpred_%j.log
set -uo pipefail
export DATA_VARIANT=all92
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
export PYTHONPATH="$THESIS_ROOT/methods/enkf/enkf_opt:$PYTHONPATH"
cd "$THESIS_ROOT/methods/enkf"
nvidia-smi --query-gpu=name --format=csv,noheader
R=runs/pedpred3_5to5_s0_all92; C=runs/pedpred3_5to5_clip_s0_all92
python3 -u -m supervisor_evaluation.checks.diag_pedpred_density $R/epoch_14.pt $R/epoch_16.pt $R/epoch_19.pt $C/epoch_01.pt
