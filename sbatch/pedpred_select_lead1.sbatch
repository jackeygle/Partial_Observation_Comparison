#!/bin/bash
#SBATCH --job-name=pp_select
#SBATCH --partition=gpu-v100-32g,gpu-v100-16g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/work/zhangx29/data_all92/logs/pp_select_%j.log
set -uo pipefail
export DATA_VARIANT=all92
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
export PYTHONPATH="$THESIS_ROOT/methods/enkf/enkf_opt:$PYTHONPATH"
cd "$THESIS_ROOT/methods/enkf"
nvidia-smi --query-gpu=name --format=csv,noheader
python3 -u -m supervisor_evaluation.checks.pedpred_select_lead1 runs/pedpred3_lead1_selection_all92.json \
    $(ls runs/pedpred3_5to5_s0_all92/epoch_*.pt) $(ls runs/pedpred3_5to5_clip_s0_all92/epoch_*.pt)
