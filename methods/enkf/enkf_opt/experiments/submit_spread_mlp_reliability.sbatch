#!/bin/bash
#SBATCH --job-name=spread_mlp_rel
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/methods/enkf/enkf_opt/experiments/outputs/spread_mlp_rel_%j.out
set -euo pipefail
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project

root=methods/enkf/enkf_opt/experiments/outputs/spread_mlp_smoke
for weight in 0.1 0.5 2.0; do
  python3 -u -m methods.enkf.enkf_opt.experiments.train_spread_calibrator \
    --train "$root/exports/atc-20130616.npz" "$root/exports/atc-20130623.npz" \
    --val "$root/exports/atc-20130630.npz" \
    --out "$root/model_rel${weight}" --epochs 30 --max-per-stratum 30000 \
    --reliability-weight "$weight"
done
