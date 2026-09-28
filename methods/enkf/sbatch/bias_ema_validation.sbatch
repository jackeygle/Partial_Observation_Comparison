#!/bin/bash
#SBATCH --job-name=enkf_bias_ema
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-6
#SBATCH --output=methods/enkf/check_outputs/bias_ema_validation/logs/day_%a_%j.out
# Is the bias EMA needed? The final EnKF with and without it, on the 7 VALIDATION days.
# Decision rule, fixed before the results: switch it off only if neither the blind RMSE
# nor the CRPS skill gets worse without it.
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
DAYS=(atc-20130616 atc-20130623 atc-20130630 atc-20130707 atc-20130714 atc-20130721 atc-20130728)
DAY=${DAYS[$SLURM_ARRAY_TASK_ID]}
OUT=methods/enkf/check_outputs/bias_ema_validation
M=supervisor_evaluation/models
RUN="from crowdcore import config; config.CFG['data']['root']='/scratch/work/zhangx29/data'; import runpy,sys; sys.argv=sys.argv[1:]; runpy.run_module(sys.argv[0], run_name='__main__')"
mkdir -p $OUT/obs $OUT/with $OUT/without
python3 -u -c "$RUN" methods.enkf.checks.export_obs_for_enkf --split valid --frames 0 \
  --obs-every-k 1 --only "$DAY" --outdir $OUT/obs
FINAL="--noise-kind residual --density-noise-space log1p --log-density-mean-correction ensemble \
  --model-checkpoint $M/pedpred3_5to5_epoch38.pt --input-frames 5 --bank $M/enkf_residual_q_bank.npz \
  --scale 1.5 --temporal-rho 0.5 --channel-scales 2.0 1.0 1.0 0.75 \
  --blind-channel-scales 1.0 1.25 1.4 0.9333333 \
  --cross-channel-matrix 1.0 0.5 0.5 0.1 0.5 1.0 0.5 0.1 0.5 0.5 1.0 0.1 0.1 0.1 0.1 1.0 \
  --ensemble 100 --radius 7 --seed 0 --frames 1000000 --warmup 500 --day $DAY --obs-dir $OUT/obs"
python3 -u -c "$RUN" methods.enkf.enkf_opt.experiments.eval_structured_q_gpu $FINAL \
  --calibration-export $OUT/with/$DAY.npz --out $OUT/with/$DAY.json
python3 -u -c "$RUN" methods.enkf.enkf_opt.experiments.eval_structured_q_gpu $FINAL --no-bias-ema \
  --calibration-export $OUT/without/$DAY.npz --out $OUT/without/$DAY.json
