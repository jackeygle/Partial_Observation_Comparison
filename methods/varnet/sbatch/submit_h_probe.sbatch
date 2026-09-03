#!/bin/bash
#SBATCH --job-name=hprobe
#SBATCH --partition=gpu-h200-141g-m
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=check_outputs/hprobe_%j.out
#SBATCH --exclude=gpu51
# Fair four-way test of the learnt sigma^2: raw / globally rescaled / constant / 1-D lookup,
# every fitted quantity taken from the valid split and applied unchanged to the 7 test days.
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project/methods/varnet
srun python3 -m methods.varnet.checks.diag_h_predicts_xhat
