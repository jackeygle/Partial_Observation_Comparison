#!/bin/bash
#SBATCH --job-name=varnet_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/methods/varnet/runs/slurm_%x_%j.out
# Usage: sbatch sbatch/submit_eval.sbatch checks/eval_test_days.py [args...]
source /etc/profile 2>/dev/null
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project/methods/varnet
nvidia-smi --query-gpu=name --format=csv,noheader
python3 -u "$@"
