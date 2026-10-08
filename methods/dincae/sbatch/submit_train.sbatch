#!/bin/bash
#SBATCH --job-name=dincae
#SBATCH --partition=gpu-h200-141g-short,gpu-a100-80g,gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:USR1@1200
#SBATCH --output=methods/dincae/runs/slurm_%x_%j.out
#
# Self-chaining: exits normally once --epochs is reached; if close to the wall clock
# (USR1 received 1200s early), queues a successor first and resumes with --resume
# (train.py saves last.pt, including optimizer state, every epoch).
# Usage: sbatch methods/dincae/sbatch/submit_train.sbatch [extra train.py args]
#   Env vars: EPOCHS (default 200)  OUT (default runs/dincae_ff)

set -uo pipefail
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || { echo "submit from the repo root" >&2; exit 1; }
DIR="$THESIS_ROOT/methods/dincae"
cd "$DIR"

EPOCHS=${EPOCHS:-200}
OUT=${OUT:-$DIR/runs/dincae_ff}
# EARLY STOPPING, as in sbatch/submit_reselect.sbatch: score this run's checkpoints on the
# validation split and stop the chain once the best one is --patience (30) epochs older than
# the newest. Selection stays post-hoc over the saved snapshots, so this only avoids training
# past a tail that is already beyond its optimum (runs/dincae_ff selected epoch 60 of 150).
# Default on; EARLY_STOP=0 reproduces the original fixed-budget recipe.
EARLY_STOP=${EARLY_STOP:-1}
if [ "$EARLY_STOP" = 1 ] && [ -f "$OUT/last.pt" ]; then
    python3 -u -m methods.dincae.checks.early_stop --run-dir "$OUT"
    if [ $? -eq 3 ]; then echo "[chain] early stop -- no resume, no successor."; exit 0; fi
fi

RESUME=""
[ -f "$OUT/last.pt" ] && RESUME="--resume"

chain() { (cd "$THESIS_ROOT" && EPOCHS="$EPOCHS" OUT="$OUT" EARLY_STOP="$EARLY_STOP" sbatch methods/dincae/sbatch/submit_train.sbatch "$@") || true; }
trap 'echo "[chain] close to wall clock, queuing successor"; chain "$@"' USR1

python3 -u -m methods.dincae.train --amp --epochs "$EPOCHS" --out "$OUT" $RESUME "$@" &
wait $!
