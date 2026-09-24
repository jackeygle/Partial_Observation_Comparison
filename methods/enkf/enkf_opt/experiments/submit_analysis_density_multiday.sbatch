#!/bin/bash
#SBATCH --job-name=analysis_density_multi
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/methods/enkf/enkf_opt/experiments/outputs/analysis_density_multi_%j.out
set -euo pipefail
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project

root=methods/enkf/enkf_opt/experiments/outputs/analysis_density_multiday
model=methods/enkf/enkf_opt/experiments/outputs/analysis_calibrator_smoke/model/model.pt
mkdir -p "$root"
common=(
  --noise-kind residual --density-noise-space log1p
  --log-density-mean-correction ensemble --scale 1.5 --temporal-rho 0.5
  --channel-scales 1.5 1 1 0.75
  --blind-channel-scales 1 1.25 1.4 0.9333333
  --cross-channel-matrix 1 .5 .5 .1 .5 1 .5 .1 .5 .5 1 .1 .1 .1 .1 1
  --frames 5000 --warmup 500
  --obs-dir methods/enkf/enkf_opt/experiments/validation_obs
  --bank methods/enkf/enkf_opt/experiments/outputs/pedpred3_train_residual_q_logdensity_8192.npz
)

for day in atc-20130707 atc-20130714 atc-20130721 atc-20130728; do
  python3 -u -m methods.enkf.enkf_opt.experiments.eval_structured_q_gpu \
    "${common[@]}" --day "$day" --out "$root/${day}_raw.json"
  python3 -u -m methods.enkf.enkf_opt.experiments.eval_structured_q_gpu \
    "${common[@]}" --day "$day" --forecast-calibrator "$model" \
    --forecast-calibration-channels 1 0 0 0 \
    --out "$root/${day}_density.json"
done
