#!/bin/bash
#SBATCH --job-name=enkf_kalman_contrib
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=methods/enkf/check_outputs/kalman_contribution/job_%j.out
# How much the Kalman update changes the estimate, original vs final EnKF, one test day.
# Both configurations exactly as in their scored runs, plus a forecast (pre-update) export.
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
DAY=atc-20130811
OBS=supervisor_evaluation/outputs/full/raw/enkf_observations
OUT=methods/enkf/check_outputs/kalman_contribution
M=supervisor_evaluation/models
RUN="from crowdcore import config; config.CFG['data']['root']='/scratch/work/zhangx29/data'; import runpy,sys; sys.argv=sys.argv[1:]; runpy.run_module(sys.argv[0], run_name='__main__')"
COMMON="--ensemble 100 --radius 7 --seed 0 --frames 1000000 --warmup 500 --day $DAY --obs-dir $OBS"

python3 -u -c "$RUN" methods.enkf.enkf_opt.experiments.eval_structured_q_gpu $COMMON \
  --noise-kind gaussian --scale 0.01 \
  --model-checkpoint methods/enkf/enkf_opt/apt-ibex_train_model_28D.pth --input-frames 1 \
  --density-noise-space physical --log-density-mean-correction none \
  --temporal-rho 0 --channel-scales 1 1 1 1 --blind-channel-scales 1 1 1 1 \
  --forecast-calibration-export $OUT/original_forecast.npz \
  --calibration-export $OUT/original_analysis.npz --out $OUT/original_run.json

python3 -u -c "$RUN" methods.enkf.enkf_opt.experiments.eval_structured_q_gpu $COMMON \
  --noise-kind residual --density-noise-space log1p --log-density-mean-correction ensemble \
  --model-checkpoint $M/pedpred3_5to5_epoch38.pt --input-frames 5 --bank $M/enkf_residual_q_bank.npz \
  --scale 1.5 --temporal-rho 0.5 --channel-scales 2.0 1.0 1.0 0.75 \
  --blind-channel-scales 1.0 1.25 1.4 0.9333333 \
  --cross-channel-matrix 1.0 0.5 0.5 0.1 0.5 1.0 0.5 0.1 0.5 0.5 1.0 0.1 0.1 0.1 0.1 1.0 \
  --forecast-calibration-export $OUT/final_forecast.npz \
  --calibration-export $OUT/final_analysis.npz --out $OUT/final_run.json

for v in original final; do
  python3 -u -m methods.enkf.checks.kalman_contribution --label $v \
    --forecast $OUT/${v}_forecast.npz --analysis $OUT/${v}_analysis.npz \
    --obs $OBS/obs_$DAY.npz --out $OUT/${v}_contribution.json
done
