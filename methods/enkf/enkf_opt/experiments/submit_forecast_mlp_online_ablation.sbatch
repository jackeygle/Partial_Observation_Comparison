#!/bin/bash
#SBATCH --job-name=forecast_mlp_ablate
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/methods/enkf/enkf_opt/experiments/outputs/forecast_mlp_ablate_%j.out
set -euo pipefail
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project

root=methods/enkf/enkf_opt/experiments/outputs/forecast_mlp_online_ablation
model=methods/enkf/enkf_opt/experiments/outputs/forecast_mlp_online_smoke/model/model.pt
mkdir -p "$root"
common=(
  --noise-kind residual --density-noise-space log1p
  --log-density-mean-correction ensemble --scale 1.5 --temporal-rho 0.5
  --channel-scales 1.5 1 1 0.75
  --blind-channel-scales 1 1.25 1.4 0.9333333
  --cross-channel-matrix 1 .5 .5 .1 .5 1 .5 .1 .5 .5 1 .1 .1 .1 .1 1
  --frames 5000 --warmup 500 --day atc-20130630
  --obs-dir methods/enkf/enkf_opt/experiments/validation_obs
  --bank methods/enkf/enkf_opt/experiments/outputs/pedpred3_train_residual_q_logdensity_8192.npz
  --forecast-calibrator "$model"
)

run_arm() {
  local name=$1 strength=$2 age=$3 den=$4 vx=$5 vy=$6 var=$7
  python3 -u -m methods.enkf.enkf_opt.experiments.eval_structured_q_gpu \
    "${common[@]}" \
    --forecast-calibration-strength "$strength" \
    --forecast-calibration-min-age "$age" \
    --forecast-calibration-channels "$den" "$vx" "$vy" "$var" \
    --out "$root/${name}.json"
}

run_arm alpha25_all       0.25 1  1 1 1 1
run_arm alpha50_all       0.50 1  1 1 1 1
run_arm alpha25_novar     0.25 1  1 1 1 0
run_arm alpha50_novar     0.50 1  1 1 1 0
run_arm alpha50_age11     0.50 11 1 1 1 1
run_arm alpha50_age51     0.50 51 1 1 1 1
run_arm alpha50_age11_nv  0.50 11 1 1 1 0
run_arm alpha50_age51_nv  0.50 51 1 1 1 0
