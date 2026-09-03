#!/bin/bash
#SBATCH --partition=gpu-h200-141g-m
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/4dvarnet_enkf/runs/slurm_%x_%j.out
# A4 — the widest prior tried, as a capacity control for A2.
#
# Usage:  sbatch --job-name=varnet_a4_k1 sbatch/submit_a4_chain.sbatch 1
#         sbatch --job-name=varnet_a4_k4 sbatch/submit_a4_chain.sbatch 4
#
# ONE variable against runs/varnet_a2_k{K}: hidden 64 -> 96.
#
#   GENN     32,076 -> 54,220    --hidden 96 --kt 5
#   solver   2,042,122 -> unchanged
#
# Purpose: check whether the prior's accuracy is still improving with width at this scale,
# so that b0's 2% deficit against A2 can be read as "the prior is saturated here" rather
# than "we did not make it big enough". Neither A2 nor A4 buys enough to displace b0.
source /etc/profile 2>/dev/null
module load scicomp-pytorch-env/2026.1
cd /scratch/work/zhangx29/Thesis_Project/4dvarnet_enkf

K=${1:?usage: sbatch --job-name=varnet_a4_kN sbatch/submit_a4_chain.sbatch <obs_every_k>}
OUTDIR=runs/varnet_a4_k${K}
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
       sbatch/submit_a4_chain.sbatch $K
echo "[chain] submitted successor (afterany:$SLURM_JOB_ID)"

# (3) train — no --lstm-hidden on purpose (solver stays at the config default, identical to
# A1 and to the baseline); the prior is plain-phi at a matched parameter budget.
nvidia-smi --query-gpu=name --format=csv,noheader
python3 -u train_varnet.py --resume \
    --obs-every-k $K \
    --hidden 96 --kt 5 \
    --days 32 --dT 200 --epochs $EPOCHS --batch 32 --amp \
    --loss supervised --iter-schedule "$SCHED" --clip-grad 1.0 \
    --outdir $OUTDIR
