#!/bin/bash
#SBATCH --job-name=analysis_calib_smoke
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/methods/enkf/enkf_opt/experiments/outputs/analysis_calib_%j.out
set -euo pipefail
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project

root=methods/enkf/enkf_opt/experiments/outputs/analysis_calibrator_smoke
mkdir -p "$root/snapshots"
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

for day in atc-20130616 atc-20130623 atc-20130630; do
  python3 -u -m methods.enkf.enkf_opt.experiments.eval_structured_q_gpu \
    "${common[@]}" --day "$day" \
    --analysis-snapshot-out "$root/snapshots/${day}.npz" \
    --analysis-snapshot-count 64 \
    --out "$root/${day}_raw.json"
done

python3 -u -m methods.enkf.enkf_opt.experiments.train_analysis_calibrator \
  --train "$root/snapshots/atc-20130616.npz" "$root/snapshots/atc-20130623.npz" \
  --val "$root/snapshots/atc-20130630.npz" \
  --out "$root/model" --epochs 6 --max-scale 1.5 \
  --reliability-weight 0.1

python3 -u -m methods.enkf.enkf_opt.experiments.eval_structured_q_gpu \
  "${common[@]}" --day atc-20130630 \
  --forecast-calibrator "$root/model/model.pt" \
  --out "$root/atc-20130630_online.json"
