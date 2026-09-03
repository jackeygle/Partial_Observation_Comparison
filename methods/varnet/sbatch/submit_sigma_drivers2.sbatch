#!/bin/bash
#SBATCH --job-name=sigmadrv2
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=00:28:00
#SBATCH --output=check_outputs/sigmadrv2_%j.out
# Two questions. (1) Does the floorless-trained head (nf5_s0) key off amplitude where the
# floor-trained ones did not? (2) Calibrated as a sigma, does a lookup on |xhat| alone beat
# 548 trained parameters?
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project/methods/varnet
AUDIT_RUNS=varnet_ml5_s0,varnet_ml5_s1,varnet_ml5_s2,varnet_ml5_s3,varnet_ml5_s4 \
  AUDIT_OUT=sigma_drivers.json srun python3 -m methods.varnet.checks.diag_sigma_drivers
echo "############ floorless-trained head ############"
AUDIT_RUNS=varnet_nf5_s0 AUDIT_OUT=sigma_drivers_nf5.json \
  srun python3 -m methods.varnet.checks.diag_sigma_drivers
