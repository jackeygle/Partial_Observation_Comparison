#!/bin/bash
#SBATCH --job-name=rs_dincae
#SBATCH --partition=gpu-h200-141g-short,gpu-a100-80g,gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@1200
#SBATCH --output=methods/dincae/runs/slurm_%x_%j.out
# DINCAE candidate for the validation re-selection (supervisor_evaluation/RESELECTION_PROTOCOL.md):
# the recipe of runs/dincae_ff (150 epochs, fixed routes) with the reference loss on withheld
# observed values (--no-full-field-loss). Four-hour segments; each first scores the new
# checkpoints on the validation split (checks/early_stop.py) and stops the chain once the best
# checkpoint is 30 epochs old.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || { echo "submit from the repo root" >&2; exit 1; }
cd "$THESIS_ROOT/methods/dincae"
OUT=$THESIS_ROOT/methods/dincae/runs/dincae_reselect_obsloss
mkdir -p "$OUT"
python3 -u -m methods.dincae.checks.early_stop --run-dir "$OUT"
if [ $? -eq 3 ]; then echo "[chain] early stop -- no resume, no successor."; exit 0; fi
RESUME=""; [ -f "$OUT/last.pt" ] && RESUME="--resume"
chain() { (cd "$THESIS_ROOT" && sbatch methods/dincae/sbatch/submit_reselect.sbatch) || true; }
trap 'echo "[chain] close to wall clock, queuing successor"; chain' USR1
python3 -u -m methods.dincae.train --amp --epochs 150 --out "$OUT" $RESUME \
    --no-full-field-loss --trajectory-mode fixed &
wait $!
