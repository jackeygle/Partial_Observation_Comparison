#!/bin/bash
#SBATCH --job-name=qenkf_gpu_smoke
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/methods/enkf/enkf_opt/experiments/outputs/qenkf_gpu_%j.out
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project
python3 -u -m methods.enkf.enkf_opt.experiments.eval_structured_q_gpu "$@"
