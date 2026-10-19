#!/bin/bash
#SBATCH --partition=gpu-h200-141g-m
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=methods/varnet/runs/slurm_%x_%j.out
# 4DVarNet with one window per batch and a 3x3x3 prior kernel (five-method thesis).
#
# Usage (from the repo root):
#   sbatch --job-name=varnet_b1k3_s3 methods/varnet/sbatch/submit_b1k3_chain.sbatch 3
#   (args: init seed, generation [internal])
#
# Anchor: runs/varnet_mse5_h96_pdfold_s3 (submit_mse5_chain.sbatch, MODE=per_day, ct fold).
# Two changes, nothing else:
#   --batch 1  The solver normalises the first cost gradient by its RMS over the whole batch
#              tensor (as 4dvarnet-core's solver.py does), so a window's reconstruction depends
#              on its batch-mates. The anchor trained with 32 windows per batch but was scored
#              with 16 (accuracy), 8 (selection) and 1 (figures, timing); one window alone gave
#              blind RMSE 0.293 against 0.237 at 16. With one window per batch in training,
#              selection, early stopping and every evaluation, training and inference match,
#              as they do in 4dvarnet-core (train and test share its batch_size).
#   --kt 3     Prior temporal kernel 3x3x3 instead of 5x3x3. Set, not selected on validation.
# Everything else is the anchor's recipe: hidden 96, ConvLSTM 64, dT 200, schedule, clip 1.0,
# bf16, 150-epoch budget, init seed 3, data seed 0, per-day routes, supervised MSE.
source /etc/profile 2>/dev/null
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || { echo "submit from the repo root" >&2; exit 1; }
cd "$THESIS_ROOT/methods/varnet"

# Generation cap: see submit_mse5_chain.sbatch.
GEN=${2:-1}
MAX_GEN=40
if [ "$GEN" -gt "$MAX_GEN" ]; then
    echo "[chain] generation $GEN exceeds $MAX_GEN -- refusing to continue; check the last log."
    exit 1
fi

S=${1:?usage: sbatch --job-name=varnet_b1k3_sS methods/varnet/sbatch/submit_b1k3_chain.sbatch <seed>}
OUTDIR=runs/varnet_mse5_h96k3_b1_pdfold_s${S}
EPOCHS=150
SCHED='0:5:1e-3,25:10:7e-4,50:15:5e-4,75:20:3e-4'

DONE=$(python3 -c "import torch,os; p='$OUTDIR/varnet_last.pt'; print(1 if (os.path.exists(p) and int(torch.load(p,map_location='cpu').get('epoch',-1))+1>=$EPOCHS) else 0)" 2>/dev/null)
if [ "$DONE" = "1" ]; then
    python3 -u -m methods.varnet.checks.early_stop --run-dir $OUTDIR --batch 1
    echo "[chain] target $EPOCHS epochs reached -- stopping chain."
    exit 0
fi

# Early stopping, scored with one window per batch like everything else for this run.
python3 -u -m methods.varnet.checks.early_stop --run-dir $OUTDIR --batch 1
if [ $? -eq 3 ]; then echo "[chain] early stop -- no resume, no successor."; exit 0; fi

CHAIN_PARTS=${CHAIN_PARTS:-gpu-h200-141g-m,gpu-h200-141g-short,gpu-h100-80g,gpu-h100-80g-short,gpu-a100-80g,gpu-v100-32g,gpu-v100-16g}
cd "$THESIS_ROOT"
sbatch --export=NONE --partition="$CHAIN_PARTS" --job-name=$SLURM_JOB_NAME \
       --dependency=afterany:$SLURM_JOB_ID \
       methods/varnet/sbatch/submit_b1k3_chain.sbatch $S $((GEN+1))
cd "$THESIS_ROOT/methods/varnet"
echo "[chain] submitted successor (afterany:$SLURM_JOB_ID)"

nvidia-smi --query-gpu=name --format=csv,noheader
python3 -u -m methods.varnet.train --resume \
    --obs-every-k 1 \
    --hidden 96 --kt 3 \
    --init-seed $S --data-seed 0 \
    --days 32 --dT 200 --epochs $EPOCHS --batch 1 --amp \
    --loss supervised \
    --iter-schedule "$SCHED" --clip-grad 1.0 \
    --trajectory-mode per_day --fold-axes ct \
    --outdir $OUTDIR
