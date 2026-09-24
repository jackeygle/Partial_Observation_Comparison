#!/bin/bash
#SBATCH --partition=gpu-v100-16g,gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/methods/enkf/enkf_opt/experiments/outputs/cross_loc_%x_%j.out

set -euo pipefail
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project

DAY="$1"
OUT_DIR="methods/enkf/enkf_opt/experiments/outputs/cross_localization_validation"
mkdir -p "$OUT_DIR"

common=(
  --noise-kind residual
  --density-noise-space log1p
  --log-density-mean-correction ensemble
  --bank methods/enkf/enkf_opt/experiments/outputs/pedpred3_train_residual_q_logdensity_8192.npz
  --scale 1.5
  --temporal-rho .5
  --channel-scales 1.5 1 1 .75
  --blind-channel-scales 1 1.25 1.4 .9333333
  --frames 100000
  --warmup 500
  --day "$DAY"
  --obs-dir methods/enkf/enkf_opt/experiments/validation_obs
)

run_variant() {
  local kind="$1"
  local mat="$2"
  local out="$OUT_DIR/${DAY}_${kind}.json"

  if [ -f "$out" ] && [ -s "$out" ]; then
    echo "[skip] $out already exists."
    return 0
  fi
  echo "[start] $DAY $kind -> $out"
  python3 -u -m methods.enkf.enkf_opt.experiments.eval_structured_q_gpu \
    "${common[@]}" \
    --cross-channel-matrix $mat \
    --out "$out"
}

run_variant "diag" "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"
run_variant "physics" "1 .5 .5 .1 .5 1 .5 .1 .5 .5 1 .1 .1 .1 .1 1"
run_variant "weak50" "1 .5 .5 .5 .5 1 .5 .5 .5 .5 1 .5 .5 .5 .5 1"
