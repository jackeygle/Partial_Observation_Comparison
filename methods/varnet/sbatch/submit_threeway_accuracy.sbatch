#!/bin/bash
#SBATCH --job-name=threeway
#SBATCH --partition=gpu-h200-141g-m
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --output=check_outputs/threeway_%j.out
#SBATCH --exclude=gpu51
# Fair four-way test of the learnt sigma^2: raw / globally rescaled / constant / 1-D lookup,
# every fitted quantity taken from the valid split and applied unchanged to the 7 test days.
module load scicomp-python-env
cd /scratch/work/zhangx29/Thesis_Project/4dvarnet_enkf
srun python3 checks/eval_threeway_accuracy.py
