#!/bin/bash
#SBATCH --job-name=enkf_val_obs
#SBATCH --partition=batch-bdw,batch-csl,batch-hsw,batch-skl
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/methods/enkf/enkf_opt/experiments/outputs/validation_obs_%j.out
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project
python3 -u -m methods.enkf.checks.export_obs_for_enkf --split valid --frames 0 \
  --outdir methods/enkf/enkf_opt/experiments/validation_obs
