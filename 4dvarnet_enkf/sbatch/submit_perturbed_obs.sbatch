#!/bin/bash
#SBATCH --job-name=pertobs
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --output=check_outputs/pertobs_%j.out
# Uncertainty from perturbed observations instead of a trained head: assimilate y + eps_i for
# 20 draws of eps from the observation-error distribution and take the spread. No retraining.
module load scicomp-python-env
cd /scratch/work/zhangx29/Thesis_Project/4dvarnet_enkf
srun python3 checks/diag_perturbed_obs_ensemble.py
