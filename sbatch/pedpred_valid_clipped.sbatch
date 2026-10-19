#!/bin/bash
#SBATCH --job-name=pp_valid_clip
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/work/zhangx29/data_all92/logs/pp_valid_clip_%j.log
set -uo pipefail
export DATA_VARIANT=all92
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
export PYTHONPATH="$THESIS_ROOT/methods/enkf/enkf_opt:$PYTHONPATH"
cd "$THESIS_ROOT/methods/enkf"
nvidia-smi --query-gpu=name --format=csv,noheader
python3 -u -m supervisor_evaluation.checks.pedpred_valid_clipped runs/pedpred3_5to5_s0_all92/epoch_14.pt runs/pedpred3_5to5_s0_all92/epoch_16.pt runs/pedpred3_5to5_s0_all92/epoch_18.pt runs/pedpred3_5to5_s0_all92/epoch_19.pt runs/pedpred3_5to5_s0_all92/epoch_21.pt runs/pedpred3_5to5_s0_all92/epoch_22.pt runs/pedpred3_5to5_clip_s0_all92/epoch_01.pt $(ls runs/pedpred3_5to5_clip_s0_all92 | grep -E 'epoch_0[2-9]' | sed 's|^|runs/pedpred3_5to5_clip_s0_all92/|')
