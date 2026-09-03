#!/bin/bash
#SBATCH --job-name=compare4
#SBATCH --partition=gpu-h200-141g-short,gpu-a100-80g,gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=0-08:00:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/senseiver_crowd/runs/slurm_%x_%j.out
# 四方对比（Senseiver / 4DVarNet / EnKF / DINCAE），「有定义格子」口径。
# 内存要 128G：EnKF 的 est_*.npz 单日约 500MB，加上整天的 (T,C,H,W) 中间量。
# 用法: sbatch sbatch/submit_compare4.sbatch [compare4.py 的参数]
set -euo pipefail
cd /scratch/work/zhangx29/Thesis_Project/senseiver_crowd
module load scicomp-pytorch-env/2026.1
python3 -u checks/compare4.py "$@"
