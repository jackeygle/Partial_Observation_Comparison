#!/bin/bash
#SBATCH --job-name=analysis_calib_channels
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/methods/enkf/enkf_opt/experiments/outputs/analysis_channels_%j.out
set -euo pipefail
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project

root=methods/enkf/enkf_opt/experiments/outputs/analysis_calibrator_channels
model=methods/enkf/enkf_opt/experiments/outputs/analysis_calibrator_smoke/model/model.pt
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
  local name=$1 den=$2 vx=$3 vy=$4 var=$5
  python3 -u -m methods.enkf.enkf_opt.experiments.eval_structured_q_gpu \
    "${common[@]}" --forecast-calibration-channels "$den" "$vx" "$vy" "$var" \
    --out "$root/${name}.json"
}

run_arm no_variance 1 1 1 0
run_arm density_only 1 0 0 0
run_arm density_vy 1 0 1 0
run_arm velocity_xy 0 1 1 0
run_arm density_vx 1 1 0 0
