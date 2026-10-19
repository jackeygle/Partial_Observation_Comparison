#!/bin/bash
#SBATCH --job-name=bench_pedpred
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/work/zhangx29/data_all92/logs/bench_pedpred_%j.log
# PedPred3 step-speed test (batch 64/256, eager / CUDA graph) on a V100, the card of the
# production run; GPU utilisation sampled every 2 s. Writes nothing but the log.
set -uo pipefail
export DATA_VARIANT=all92
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
export PYTHONPATH="$THESIS_ROOT/methods/enkf/enkf_opt:$PYTHONPATH"
cd "$THESIS_ROOT/methods/enkf"
nvidia-smi --query-gpu=name --format=csv,noheader
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -l 2 > /scratch/work/zhangx29/data_all92/logs/bench_pedpred_util_$SLURM_JOB_ID.txt &
SMI=$!
python3 -u -m supervisor_evaluation.checks.bench_pedpred_speed --days 4
kill $SMI
