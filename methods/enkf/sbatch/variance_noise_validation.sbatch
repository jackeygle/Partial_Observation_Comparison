#!/bin/bash
#SBATCH --job-name=enkf_var_noise
#SBATCH --partition=gpu-v100-32g,gpu-v100-16g,gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --array=0-6
#SBATCH --output=methods/enkf/check_outputs/variance_noise_validation/logs/day_%a_%j.out
# Velocity-variance noise, on the 7 VALIDATION days. The final EnKF adds variance-channel
# noise of std 0.47 (PROC_STD 0.417 x 1.5 x 0.75) while the forecast model's own one-step
# error there is 0.095; clipping at 0 then pushes the mean up. Arms (everything else = final):
#   var040 / var020 / var010 : variance channel scale 0.75 -> 0.4 / 0.2 / 0.1
#   native                   : bank kept at its own measured size, channel scales all 1
# Baseline = the final configuration, already run in bias_ema_validation/with.
# Rule, fixed before the results: lowest pooled blind CRPS, with pooled spread/RMSE in
# [0.8, 1.25] and no channel's CRPS more than 2% worse than the baseline.
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
DAYS=(atc-20130616 atc-20130623 atc-20130630 atc-20130707 atc-20130714 atc-20130721 atc-20130728)
DAY=${DAYS[$SLURM_ARRAY_TASK_ID]}
OUT=methods/enkf/check_outputs/variance_noise_validation
OBS=methods/enkf/check_outputs/bias_ema_validation/obs
M=supervisor_evaluation/models
RUN="from crowdcore import config; config.CFG['data']['root']='/scratch/work/zhangx29/data'; import runpy,sys; sys.argv=sys.argv[1:]; runpy.run_module(sys.argv[0], run_name='__main__')"
BASE="--noise-kind residual --density-noise-space log1p --log-density-mean-correction ensemble \
  --model-checkpoint $M/pedpred3_5to5_epoch38.pt --input-frames 5 --bank $M/enkf_residual_q_bank.npz \
  --scale 1.5 --temporal-rho 0.5 --blind-channel-scales 1.0 1.25 1.4 0.9333333 \
  --cross-channel-matrix 1.0 0.5 0.5 0.1 0.5 1.0 0.5 0.1 0.5 0.5 1.0 0.1 0.1 0.1 0.1 1.0 \
  --ensemble 100 --radius 7 --seed 0 --frames 1000000 --warmup 500 --day $DAY --obs-dir $OBS"
run() { mkdir -p $OUT/$1; python3 -u -c "$RUN" methods.enkf.enkf_opt.experiments.eval_structured_q_gpu $BASE "${@:2}" \
          --calibration-export $OUT/$1/$DAY.npz --out $OUT/$1/$DAY.json; }
run var040 --channel-scales 2.0 1.0 1.0 0.4
run var020 --channel-scales 2.0 1.0 1.0 0.2
run var010 --channel-scales 2.0 1.0 1.0 0.1
run native --channel-scales 1 1 1 1 --bank-native-std
