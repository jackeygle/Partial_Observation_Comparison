#!/bin/bash
#SBATCH --partition=gpu-h200-141g-m
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=methods/varnet/runs/slurm_%x_%j.out
# Prior-width candidates for the validation re-selection
# (supervisor_evaluation/RESELECTION_PROTOCOL.md). Recipe of submit_mse5_chain.sbatch,
# seed 0, with only --hidden/--kt changed, fixed routes as in runs/varnet_mse5_h96_s0,
# and early stopping: each segment first scores the new checkpoints on the validation
# split (checks/early_stop.py) and neither resumes nor queues a successor once the best
# checkpoint is 20 epochs old.
#
# Usage: sbatch --job-name=rs_vn_h32k3 methods/varnet/sbatch/submit_reselect_chain.sbatch 32 3
source /etc/profile 2>/dev/null
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || { echo "submit from the repo root" >&2; exit 1; }
cd "$THESIS_ROOT/methods/varnet"

H=${1:?hidden}; KT=${2:?kt}; GEN=${3:-1}
[ "$GEN" -gt 12 ] && { echo "[chain] generation cap reached"; exit 1; }
OUTDIR=runs/varnet_reselect_h${H}k${KT}_s0
EPOCHS=150
SCHED='0:5:1e-3,25:10:7e-4,50:15:5e-4,75:20:3e-4'
mkdir -p $OUTDIR

DONE=$(python3 -c "import torch,os; p='$OUTDIR/varnet_last.pt'; print(1 if (os.path.exists(p) and int(torch.load(p,map_location='cpu').get('epoch',-1))+1>=$EPOCHS) else 0)" 2>/dev/null)
if [ "$DONE" = "1" ]; then
    python3 -u -m methods.varnet.checks.early_stop --run-dir $OUTDIR
    echo "[chain] $EPOCHS epochs reached -- stopping chain."; exit 0
fi
python3 -u -m methods.varnet.checks.early_stop --run-dir $OUTDIR
if [ $? -eq 3 ]; then echo "[chain] early stop -- no resume, no successor."; exit 0; fi

cd "$THESIS_ROOT"
sbatch --export=NONE --job-name=$SLURM_JOB_NAME --dependency=afterany:$SLURM_JOB_ID \
       methods/varnet/sbatch/submit_reselect_chain.sbatch $H $KT $((GEN+1))
cd "$THESIS_ROOT/methods/varnet"
nvidia-smi --query-gpu=name --format=csv,noheader
python3 -u -m methods.varnet.train --resume \
    --obs-every-k 1 --trajectory-mode fixed \
    --hidden $H --kt $KT \
    --init-seed 0 --data-seed 0 \
    --days 32 --dT 200 --epochs $EPOCHS --batch 32 --amp \
    --loss supervised \
    --iter-schedule "$SCHED" --clip-grad 1.0 \
    --outdir $OUTDIR
