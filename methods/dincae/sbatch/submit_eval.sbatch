#!/bin/bash
#SBATCH --job-name=dincae_eval
#SBATCH --partition=gpu-h200-141g-short,gpu-a100-80g,gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/methods/dincae/runs/slurm_%x_%j.out

source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project/methods/dincae
python3 -u -m methods.dincae.checks.evaluate "$@"
