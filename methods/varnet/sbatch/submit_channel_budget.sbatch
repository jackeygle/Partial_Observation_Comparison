#!/bin/bash
#SBATCH --job-name=chanbudget
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=check_outputs/chanbudget_%j.out
# One 200-frame window: decompose J per channel at x0 AND at the finished reconstruction, and
# work one psi output value out by hand against real data. The GPU is needed only for the second
# decomposition, which runs the solver's 20 unrolled iterations.
# Writes check_outputs/eval/channel_budget.json, which checks/plot_math_detail.py reads.
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project/methods/varnet
srun python3 -m methods.varnet.checks.diag_channel_budget
