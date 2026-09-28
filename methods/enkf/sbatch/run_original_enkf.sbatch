#!/bin/bash
#SBATCH --job-name=enkf_original
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-6
#SBATCH --output=methods/enkf/check_outputs/original_vs_final/logs/original_%a_%j.out
# The ORIGINAL EnKF configuration on the seven test days, through the same GPU filter,
# observations and frames as the final EnKF, so the two can be scored identically
# (methods/enkf/checks/compare_original_final.py).
#
# Original configuration = the vendored forecast model (apt-ibex, 1 input frame) and
# independent Gaussian process noise at 0.01 x PROC_STD -- the multiplier hard-coded in
# the original forecast step. No residual noise, no per-channel or blind-cell scaling,
# no temporal correlation, physical density space. The localisation fix (velocity
# observations assimilated) is part of the filter and applies to both versions.
#
# Submit from the repository root:  sbatch methods/enkf/sbatch/run_original_enkf.sbatch
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
DAYS=(atc-20130811 atc-20130818 atc-20130825 atc-20130901 atc-20130915 atc-20130922 atc-20130929)
DAY=${DAYS[$SLURM_ARRAY_TASK_ID]}
OUT=methods/enkf/check_outputs/original_vs_final
mkdir -p $OUT/exports $OUT/runs
python3 -u -c "from crowdcore import config; config.CFG['data']['root']='/scratch/work/zhangx29/data'; import runpy,sys; sys.argv=sys.argv[1:]; runpy.run_module(sys.argv[0], run_name='__main__')" \
  methods.enkf.enkf_opt.experiments.eval_structured_q_gpu \
  --noise-kind gaussian --scale 0.01 \
  --model-checkpoint methods/enkf/enkf_opt/apt-ibex_train_model_28D.pth --input-frames 1 \
  --density-noise-space physical --log-density-mean-correction none \
  --temporal-rho 0 --channel-scales 1 1 1 1 --blind-channel-scales 1 1 1 1 \
  --ensemble 100 --radius 7 --seed 0 \
  --frames 1000000 --warmup 500 \
  --day "$DAY" --obs-dir supervisor_evaluation/outputs/full/raw/enkf_observations \
  --calibration-export $OUT/exports/$DAY.npz \
  --out $OUT/runs/$DAY.json
