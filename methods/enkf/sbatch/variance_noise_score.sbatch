#!/bin/bash
#SBATCH --job-name=enkf_var_noise_score
#SBATCH --partition=batch-csl
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=methods/enkf/check_outputs/variance_noise_validation/logs/score_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
OUT=methods/enkf/check_outputs/variance_noise_validation
V=atc-20130616,atc-20130623,atc-20130630,atc-20130707,atc-20130714,atc-20130721,atc-20130728
for arm in var040 var020 var010 native; do
  python3 -u -m methods.enkf.checks.compare_original_final --no-check \
    --original methods/enkf/check_outputs/bias_ema_validation/with --final $OUT/$arm \
    --labels baseline,$arm --days $V --out $OUT/${arm}_vs_baseline.json
done
