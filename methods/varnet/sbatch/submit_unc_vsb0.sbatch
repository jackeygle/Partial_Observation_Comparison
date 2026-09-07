#!/bin/bash
#SBATCH --job-name=uncvsb0
#SBATCH --partition=gpu-h200-141g-m
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/methods/varnet/check_outputs/uncvsb0_%j.out
# The 5-member deep ensemble, combined by Lakshminarayanan et al. Sec. 2.4 moment matching:
#   mu*     = mean_m mu_m
#   sigma*^2= mean_m sigma_m^2 + Var_m(mu_m)     (population variance, unbiased=False)
# Scored on 7 test days with their Sec. 3.1 metrics (NLL, RMSE) and their App. A.2 calibration
# curve. The old ml5 numbers in the deck came from a model whose VarianceHead no longer exists
# and must be replaced wholesale by these.
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project/methods/varnet
srun python3 -m methods.varnet.checks.eval_uncertainty --batch 2 \
     --run-fmt "runs/varnet_vsb0_s{}" --members 0 1 2 3 4 \
     --out check_outputs/eval/uncertainty_vsb0.json
