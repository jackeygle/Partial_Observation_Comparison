#!/bin/bash
# ROUTE 2 (DATA_VARIANT=all92) copy of submit_aughead_wink3_chain.sbatch: same recipe, all 92 ATC days; 160G for
# the doubled set of training windows held in host memory.
#SBATCH --partition=gpu-h200-141g-m,gpu-h200-141g-short,gpu-h100-80g,gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=04:00:00   # 4-h segments backfill sooner (user, 2026-10-14)
#SBATCH --output=methods/varnet/runs/slurm_%x_%j.out
# 4DVarNet aug. var. (augmented variance, observation NLL, variance head) on the solver of the
# final 4DVarNet: per-window gradient normalisation and a 3x3x3 prior kernel (six-method thesis).
#
# Usage (from the repo root):
#   sbatch --job-name=varnet_augwink3_s3 methods/varnet/sbatch/submit_aughead_wink3_all92_chain.sbatch 3
#   (args: init seed, generation [internal])
#
# Anchor: runs/varnet_aughead_obs_h96_fold_s0 (submit_aughead_chain.sbatch, per-day routes,
# ct fold, init seed 0, loss nll / beta 0 / var-eps 1e-6 / --augmented-var --obs-nll --var-head).
# The same two changes as submit_wink3_chain.sbatch made to the MSE model, nothing else:
#   --normg-scope window  The anchor scales the first cost gradient by its RMS over the whole
#              batch tensor plus 1e-12. The cost is a batch mean, so each window's gradient
#              carries 1/B, and at this problem's gradient sizes the 1e-12 does not cancel it:
#              the same weights gave blind RMSE 0.237 at B=16 and 0.293 at B=1. With 'window'
#              each window is solved exactly as if alone (checked: max difference 6e-8), so
#              training, selection, evaluation, figures and timing all see the same solver,
#              whatever their batch size. Batch stays 32, so training is as fast as before.
#   --kt 3     Prior temporal kernel 3x3x3 instead of 5x3x3. Set, not selected on validation.
# Everything else is the anchor's recipe: hidden 96, ConvLSTM 64, dT 200, batch 32, schedule, clip 1.0,
# bf16, 150-epoch budget, data seed 0, per-day routes, the NLL loss and variance head; init seed 3,
# the seed of the final MSE model (the anchor used 0), so the two differ only in the variance parts.
# Early stopping and selection as for the MSE model: all walkable validation cells, K=20 snapshots.
source /etc/profile 2>/dev/null
export DATA_VARIANT=all92   # route 2: all 92 days (63/14/15); successors are submitted with --export=NONE
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || { echo "submit from the repo root" >&2; exit 1; }
set -e
cd "$THESIS_ROOT/methods/varnet"

# Generation cap: see submit_mse5_chain.sbatch.
GEN=${2:-1}
MAX_GEN=24   # 4-h segments: twice as many as with 8 h
if [ "$GEN" -gt "$MAX_GEN" ]; then
    echo "[chain] generation $GEN exceeds $MAX_GEN -- refusing to continue; check the last log."
    exit 1
fi

S=${1:?usage: sbatch --job-name=varnet_augwink3_sS methods/varnet/sbatch/submit_aughead_wink3_all92_chain.sbatch <seed>}
OUTDIR=runs/varnet_aughead_obs_h96k3_win_pdfold_s${S}_all92
release_final() {
    if [ -f "$OUTDIR/final_job_id" ]; then
        final_job=$(cat "$OUTDIR/final_job_id")
        scontrol release "$final_job"
        echo "[chain] training finished; released validation/test job $final_job"
    fi
}
EPOCHS=150
SCHED='0:5:1e-3,25:10:7e-4,50:15:5e-4,75:20:3e-4'

DONE=$(python3 -c "import torch,os; p='$OUTDIR/varnet_last.pt'; print(1 if (os.path.exists(p) and int(torch.load(p,map_location='cpu').get('epoch',-1))+1>=$EPOCHS) else 0)" 2>/dev/null)
if [ "$DONE" = "1" ]; then
    early_stop_status=0
    python3 -u -m methods.varnet.checks.early_stop --run-dir "$OUTDIR" --scope all_walkable --batch 1 || early_stop_status=$?
    if [ "$early_stop_status" -ne 0 ] && [ "$early_stop_status" -ne 3 ]; then exit "$early_stop_status"; fi
    echo "[chain] target $EPOCHS epochs reached -- stopping chain."
    release_final
    exit 0
fi

# Early stopping (its batch no longer matters for this solver).
early_stop_status=0
python3 -u -m methods.varnet.checks.early_stop --run-dir "$OUTDIR" --scope all_walkable --batch 1 || early_stop_status=$?
if [ "$early_stop_status" -eq 3 ]; then
    echo "[chain] early stop -- no resume, no successor."
    release_final
    exit 0
fi
if [ "$early_stop_status" -ne 0 ]; then exit "$early_stop_status"; fi

CHAIN_PARTS=gpu-h200-141g-m,gpu-h200-141g-short,gpu-h100-80g,gpu-a100-80g   # bf16: no V100 (user, 2026-10-14: priority low, widen from H200 only)
cd "$THESIS_ROOT"
successor_job=$(sbatch --parsable --export=NONE --partition="$CHAIN_PARTS" \
       --gres=gpu:1 --time=04:00:00 --job-name="$SLURM_JOB_NAME" \
       --dependency=afterany:$SLURM_JOB_ID \
       methods/varnet/sbatch/submit_aughead_wink3_all92_chain.sbatch "$S" "$((GEN+1))")
successor_job=${successor_job%%;*}
cd "$THESIS_ROOT/methods/varnet"
echo "[chain] submitted successor $successor_job (afterany:$SLURM_JOB_ID); 200 frames, H200/H100/A100, 4 hours"

nvidia-smi --query-gpu=name --format=csv,noheader
training_status=0
python3 -u -m methods.varnet.train --resume \
    --obs-every-k 1 \
    --hidden 96 --kt 3 \
    --init-seed $S --data-seed 0 \
    --days 63 --dT 200 --epochs $EPOCHS --batch 32 --amp \
    --loss nll --nll-beta 0.0 --var-eps 1e-6 --augmented-var --obs-nll --var-head \
    --iter-schedule "$SCHED" --clip-grad 1.0 \
    --trajectory-mode per_day --fold-axes ct --normg-scope window \
    --outdir "$OUTDIR" || training_status=$?
if [ "$training_status" -ne 0 ]; then
    scancel "$successor_job" || true
    echo "[chain] training exited with $training_status; cancelled successor $successor_job"
    exit "$training_status"
fi
