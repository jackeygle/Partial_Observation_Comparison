#!/bin/bash
#SBATCH --partition=gpu-v100-32g,gpu-v100-16g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=supervisor_evaluation/outputs/dev/varnet_wink3_200_20261010/final_%j.out
source /etc/profile 2>/dev/null
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || exit 1
set -e
cd "$THESIS_ROOT"
OUT="$THESIS_ROOT/supervisor_evaluation/outputs/dev/varnet_wink3_200_20261010"
if [ -f "$OUT/status.json" ]; then exit 0; fi
GEN=${1:-1}
if [ "$GEN" -gt 12 ]; then echo 'Evaluation relay limit reached'; exit 1; fi
successor=$(sbatch --parsable --export=NONE --dependency=afterany:$SLURM_JOB_ID \
    --job-name=varnet_wink3_final supervisor_evaluation/sbatch/final_wink3.sbatch "$((GEN+1))")
successor=${successor%%;*}
echo "[final] timeout successor $successor"
status=0
python3 -u -m supervisor_evaluation.checks.final_wink3 \
    --run-dir "$THESIS_ROOT/methods/varnet/runs/varnet_mse5_h96k3_win_pdfold_s3" \
    --out "$OUT" || status=$?
scancel "$successor" || true
exit "$status"
