#!/bin/bash
#SBATCH --job-name=sigmaaudit
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:28:00
#SBATCH --output=check_outputs/sigmaaudit_%j.out
# Two audits of the variance head, neither needing any training:
#   1. can a (channel, distance-to-nearest-observation) lookup reproduce our sigma?
#   2. is x_hat near a stationary point of J, i.e. is the inverse-Hessian route available?
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project/methods/varnet
srun python3 -m methods.varnet.checks.diag_sigma_is_it_flow
