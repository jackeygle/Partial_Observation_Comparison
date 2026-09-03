#!/bin/bash
#SBATCH --job-name=uncmse5
#SBATCH --partition=gpu-h200-141g-m
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=check_outputs/uncmse5_%j.out
#SBATCH --exclude=gpu51
# Lakshminarayanan et al.'s Table 2 contrast, on our data: an MSE-trained ensemble whose
# uncertainty is only the empirical variance across its members, against the NLL-trained
# ensemble that learns a sigma^2. Same architecture, same seeds, same schedule, same days --
# the loss is the only difference, which is what makes it a single-variable comparison.
module load scicomp-python-env
cd /scratch/work/zhangx29/Thesis_Project/4dvarnet_enkf
srun python3 checks/eval_uncertainty.py --batch 2 \
     --run-fmt "runs/varnet_mse5_s{}" --members 0 1 2 3 4 \
     --out check_outputs/eval/uncertainty_mse5.json
