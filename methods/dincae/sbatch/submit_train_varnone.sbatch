#!/bin/bash
#SBATCH --job-name=dincae_varnone
#SBATCH --partition=gpu-h200-141g-m,gpu-h200-141g-short,gpu-h100-80g,gpu-h100-80g-short,gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@1200
#SBATCH --output=methods/dincae/runs/slurm_%x_%j.out
#
# DINCAE without the log1p of the velocity variance (DINCAE_VAR_TRANSFORM=none); everything
# else is the recipe of runs/dincae_ff_pd (150 epochs, batch 64, lr 1e-3, l2 1e-4, clip 5,
# enc 32/64/96, loss weights 0.3/0.7, mean pooling, stride 4, per-day routes, full-field
# loss, bf16). A fixed budget of 150 epochs, no early stopping, and every epoch saved, so the
# validation selection can consider all 150. Self-chaining as submit_train.sbatch.
# Usage (repo root): sbatch methods/dincae/sbatch/submit_train_varnone.sbatch
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || { echo "submit from the repo root" >&2; exit 1; }
export DINCAE_VAR_TRANSFORM=none
DIR="$THESIS_ROOT/methods/dincae"
cd "$DIR"
OUT="$DIR/runs/dincae_ff_pd_varnone_all150"
GEN=${1:-1}
if [ "$GEN" -gt 12 ]; then echo "[chain] generation limit reached"; exit 1; fi
DONE=$(python3 -c "import torch,os; p='$OUT/last.pt'; print(1 if os.path.exists(p) and int(torch.load(p,map_location='cpu',weights_only=False)['epoch'])+1>=150 else 0)" 2>/dev/null)
if [ "$DONE" = "1" ]; then echo "[chain] 150 epochs reached -- done."; exit 0; fi
RESUME=""
[ -f "$OUT/last.pt" ] && RESUME="--resume"
chain() { (cd "$THESIS_ROOT" && sbatch methods/dincae/sbatch/submit_train_varnone.sbatch $((GEN+1))) || true; }
trap 'echo "[chain] close to wall clock, queuing successor"; chain' USR1
nvidia-smi --query-gpu=name --format=csv,noheader
[ -f artifacts/state_stats_none.npz ] || (cd "$THESIS_ROOT" && python3 -u -m methods.dincae.state)
python3 -u -m methods.dincae.train --amp --epochs 150 --save-every 1 --save-from 0 \
    --trajectory-mode per_day --full-field-loss --out "$OUT" $RESUME &
wait $!
