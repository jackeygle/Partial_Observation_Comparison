#!/bin/bash
#SBATCH --job-name=dincae_eval
#SBATCH --partition=gpu-h200-141g-short,gpu-a100-80g,gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/dincae_crowd/runs/slurm_%x_%j.out

module load scicomp-pytorch-env/2026.1
cd /scratch/work/zhangx29/Thesis_Project/dincae_crowd
python3 -u checks/evaluate.py "$@"
