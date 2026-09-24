#!/bin/bash
#SBATCH --job-name=qenkf
#SBATCH --partition=batch-bdw,batch-csl,batch-hsw,batch-skl
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/methods/enkf/enkf_opt/experiments/outputs/qenkf_%j.out
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project
python3 -u -m methods.enkf.enkf_opt.experiments.eval_structured_q "$@"
