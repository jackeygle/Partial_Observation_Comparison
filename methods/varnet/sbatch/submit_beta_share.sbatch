#!/bin/bash
#SBATCH --job-name=betashare
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=00:28:00
#SBATCH --output=check_outputs/betashare_%j.out
# Why does beta=0 lose 42% blind-zone MSE to beta=1 when the two differ only by a detached
# constant on each point's NLL? Measures the 1/sigma^2 gradient weight and splits the error by
# empty vs occupied cells, which is what discriminates the candidate causes.
# Submit WITHOUT --export=NONE: that strips the shell so `module` itself is undefined, the load
# silently no-ops and python3 is not on PATH.
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project/methods/varnet
export AUDIT_RUNS AUDIT_WINDOWS
srun python3 -m methods.varnet.checks.diag_beta_gradient_share
