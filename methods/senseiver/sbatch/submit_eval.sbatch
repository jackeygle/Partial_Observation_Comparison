#!/bin/bash
#SBATCH --job-name=senseiver_eval
#SBATCH --partition=gpu-h200-141g-short,gpu-a100-80g,gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/methods/senseiver/runs/slurm_%x_%j.out
# 用法: sbatch sbatch/submit_eval.sbatch [evaluate.py 的参数]
set -euo pipefail
cd /scratch/work/zhangx29/Thesis_Project/methods/senseiver
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
python3 -u -m methods.senseiver.checks.evaluate "$@"
