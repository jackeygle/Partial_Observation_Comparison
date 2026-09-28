#!/bin/bash
#SBATCH --job-name=enkf_bias_ema_score
#SBATCH --partition=batch-csl
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=methods/enkf/check_outputs/bias_ema_validation/logs/score_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
OUT=methods/enkf/check_outputs/bias_ema_validation
python3 -u -m methods.enkf.checks.compare_original_final --no-check \
  --original $OUT/with --final $OUT/without --labels with_bias,without_bias \
  --days atc-20130616,atc-20130623,atc-20130630,atc-20130707,atc-20130714,atc-20130721,atc-20130728 \
  --out $OUT/comparison.json
