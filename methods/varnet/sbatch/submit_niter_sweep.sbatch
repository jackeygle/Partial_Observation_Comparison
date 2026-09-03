#!/bin/bash
#SBATCH --job-name=nitersweep
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=check_outputs/nitersweep_%j.out
# Does the reconstruction actually keep improving for all 20 unrolled steps? One window said
# no (blind MSE bottoms out around step 9 and then degrades). This checks it on the real
# 7 held-out test days by re-evaluating the SAME checkpoint at several iteration counts.
# Nothing is retrained; only the loop length at inference changes.
module load scicomp-python-env
cd /scratch/work/zhangx29/Thesis_Project/4dvarnet_enkf
for K in 5 8 10 12 15 20 25; do
  srun python3 checks/eval_test_days.py --n-iter $K --tag "_b0_k1_it$K"
done
