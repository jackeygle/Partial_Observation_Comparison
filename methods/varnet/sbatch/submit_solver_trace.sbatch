#!/bin/bash
#SBATCH --job-name=solvertrace
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=check_outputs/solvertrace_%j.out
# Instrument the solver's 20 unrolled steps on one window: J and its two terms, update size,
# the four gate means, and the gate arithmetic at one voxel. Writes
# check_outputs/eval/solver_trace.json, which checks/plot_math_detail.py reads.
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project/methods/varnet
srun python3 -m methods.varnet.checks.diag_solver_trace
