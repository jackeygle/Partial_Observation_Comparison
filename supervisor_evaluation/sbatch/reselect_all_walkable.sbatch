#!/bin/bash
#SBATCH --partition=gpu-v100-32g,gpu-v100-16g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=supervisor_evaluation/outputs/reselect_all_walkable_20261010/%x_%j.out
source /etc/profile 2>/dev/null
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || exit 1
set -e
cd "$THESIS_ROOT"
METHOD=${1:?method required}
GEN=${2:-1}
OUT=supervisor_evaluation/outputs/reselect_all_walkable_20261010
if [ -f "$OUT/${METHOD}_selection.json" ]; then exit 0; fi
if [ "$GEN" -gt 12 ]; then echo 'Relay limit reached'; exit 1; fi
next=$(sbatch --parsable --export=NONE --job-name="reselect_${METHOD}" \
    --dependency=afterany:$SLURM_JOB_ID supervisor_evaluation/sbatch/reselect_all_walkable.sbatch \
    "$METHOD" "$((GEN+1))")
next=${next%%;*}
echo "[relay] successor $next"
status=0
python3 -u -m supervisor_evaluation.checks.reselect_all_walkable "$METHOD" || status=$?
scancel "$next" || true
exit "$status"
