#!/bin/bash
#SBATCH --job-name=uncrefresh
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=check_outputs/uncrefresh_%j.out
# Re-run the uncertainty evaluation now that the variance head's sigma^2 floor is gone.
# No retraining: the same 5 checkpoints, only the head's parameterisation changed
# (sigma^2 = softplus(head) instead of obs_var + softplus(head)). Regenerates the json the
# three-way deck reads, plus the variance-head diagnostic on the same window.
module load scicomp-python-env
cd /scratch/work/zhangx29/Thesis_Project/4dvarnet_enkf
srun python3 checks/eval_uncertainty.py --batch 2
srun python3 checks/diag_varhead_trace.py
