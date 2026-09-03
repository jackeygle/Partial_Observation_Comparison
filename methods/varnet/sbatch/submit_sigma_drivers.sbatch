#!/bin/bash
#SBATCH --job-name=sigmadrv
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=00:28:00
#SBATCH --output=check_outputs/sigmadrv_%j.out
# Which of the head's 12 pointwise inputs does sigma actually track? One-dimensional
# quantile-binned lookups against each, R^2 on held-out windows.
module load scicomp-python-env
cd /scratch/work/zhangx29/Thesis_Project/4dvarnet_enkf
srun python3 checks/diag_sigma_drivers.py
