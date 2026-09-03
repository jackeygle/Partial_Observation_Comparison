#!/bin/bash
#SBATCH --partition=gpu-h200-141g-m
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/4dvarnet_enkf/runs/slurm_%x_%j.out
# A2 — a LARGER prior than the baseline, solver unchanged.
#
# Usage:  sbatch --job-name=varnet_a2_k1 sbatch/submit_a2_chain.sbatch 1
#         sbatch --job-name=varnet_a2_k4 sbatch/submit_a2_chain.sbatch 4
#
# ONE variable against runs/varnet_b0_k{K}: the prior's capacity.
#
#   GENN     9,420 -> 32,076     --hidden 64 --kt 5
#            psi 3,488 -> 11,584 (kt 3 -> 5 also reaches two frames further in time),
#            phi 1,188 -> 4,420
#   solver   2,042,122 -> unchanged (no --lstm-hidden here on purpose)
#   total    2,051,542 -> 2,074,198
#
# Where that sits against the paper: 0.64x its Lorenz-96 GENN (~50,000) and 2.1x the
# Lorenz-63 one (~15,000).
#
# Result on the held-out test set: full-state RMSE 0.1666 (k=1) / 0.1883 (k=4) — only ~2%
# better than b0's 0.1702 / 0.1891 for 3.4x the prior parameters, and 2.1x slower at
# inference (7.86 vs 3.66 ms/frame). That trade is why b0, not this, is the model of record.
source /etc/profile 2>/dev/null
module load scicomp-pytorch-env/2026.1
cd /scratch/work/zhangx29/Thesis_Project/4dvarnet_enkf

K=${1:?usage: sbatch --job-name=varnet_a2_kN sbatch/submit_a2_chain.sbatch <obs_every_k>}
OUTDIR=runs/varnet_a2_k${K}
EPOCHS=100
SCHED='0:5:1e-3,25:10:7e-4,50:15:5e-4,75:20:3e-4'

# (1) already done?
DONE=$(python3 -c "import torch,os; p='$OUTDIR/varnet_last.pt'; print(1 if (os.path.exists(p) and int(torch.load(p,map_location='cpu').get('epoch',-1))+1>=$EPOCHS) else 0)" 2>/dev/null)
if [ "$DONE" = "1" ]; then
    echo "[chain] target $EPOCHS epochs reached — stopping chain."
    exit 0
fi

# (2) submit successor (runs whether this job finishes or hits the time limit).
# --export=NONE is REQUIRED, not tidiness: without it each generation inherits the previous
# one's environment and then runs `module load` again, so PATH and LD_LIBRARY_PATH gain one
# more copy of the module's paths per generation. After ~6 generations the environment
# exceeded ARG_MAX and every command failed with "Argument list too long" — that is how the
# first C1 attempt died at epoch 59.
sbatch --export=NONE --job-name=$SLURM_JOB_NAME --dependency=afterany:$SLURM_JOB_ID \
       sbatch/submit_a2_chain.sbatch $K
echo "[chain] submitted successor (afterany:$SLURM_JOB_ID)"

# (3) train — no --lstm-hidden on purpose (solver stays at the config default, identical to
nvidia-smi --query-gpu=name --format=csv,noheader
python3 -u train_varnet.py --resume \
    --obs-every-k $K \
    --hidden 64 --kt 5 \
    --days 32 --dT 200 --epochs $EPOCHS --batch 32 --amp \
    --loss supervised --iter-schedule "$SCHED" --clip-grad 1.0 \
    --outdir $OUTDIR
