#!/bin/bash
#SBATCH --job-name=varnet
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/4dvarnet_enkf/runs/slurm_%x_%j.out
# Usage (submit from any directory): sbatch sbatch/submit_varnet.sbatch [train_varnet.py args...]
#   e.g.: sbatch sbatch/submit_varnet.sbatch --days 8 --dT 7 --n-iter 15 --epochs 30 --loss supervised
source /etc/profile 2>/dev/null
module load scicomp-pytorch-env/2026.1
cd /scratch/work/zhangx29/Thesis_Project/4dvarnet_enkf
mkdir -p runs
nvidia-smi --query-gpu=name --format=csv,noheader
python3 -u train_varnet.py "$@"
