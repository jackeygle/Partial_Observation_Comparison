#!/bin/bash
#SBATCH --partition=gpu-h200-141g-m
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/methods/varnet/runs/slurm_%x_%j.out
#
# Usage:  sbatch --job-name=varnet_b0_k1 sbatch/submit_b0_chain.sbatch 1
#
# This is now the MODEL OF RECORD: the config.yaml defaults, and what the presentation shows.
#
#   psi     Conv3d(4 -> 32, 3x3x3), centre tap masked      3,488
#   phi     Conv3d(32,32,1) + ReLU + Conv3d(32,4,1)        1,188      } per branch
#   GENN    two branches + a ConvTranspose Up               9,420
#   solver  ConvLSTM                                    2,042,122
#   total                                              2,051,542
#
# Against the paper: 0.63x its Lorenz-63 GENN (~15,000) and 0.19x the Lorenz-96 one
# (~50,000) — smaller than either, and smaller than every configuration tried here. That is
# the point: Sec. 3.2 asks only for a zero-centre psi and kernel size 1 throughout phi, and
# this is literally that, with nothing added.
#
# Result on the held-out test set: full-state RMSE 0.1702 (k=1) / 0.1891 (k=4), i.e. within
# 2% of the 3.4x-larger prior in runs/varnet_a2_k*, and 2.1x faster at inference.
#
# 150 epochs, not 100, and the curriculum is left exactly as the other runs have it
# (5 iterations to epoch 25, then 10, 15, and 20 from epoch 75). Two reasons to keep it
# rather than stretch it over 150: epochs 0-99 stay directly comparable to the existing
# 100-epoch runs at matched epochs, and the extra 50 epochs all land on n_iter=20 — the
# configuration the model is actually deployed at — taking it from 25 epochs to 75.
#
#   Each job: (1) if the checkpoint already reached the target -> stop the chain;
#             (2) otherwise submit a successor (afterany) that will --resume when this job
#                 finishes OR is killed at the time limit;
#             (3) train with --resume.
source /etc/profile 2>/dev/null
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project/methods/varnet

K=${1:?usage: sbatch --job-name=varnet_b0_kN sbatch/submit_b0_chain.sbatch <obs_every_k>}
OUTDIR=runs/varnet_b0_k${K}
EPOCHS=150
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
       sbatch/submit_b0_chain.sbatch $K
echo "[chain] submitted successor (afterany:$SLURM_JOB_ID)"

# (3) train — no --lstm-hidden on purpose (solver stays at the config default, identical to
nvidia-smi --query-gpu=name --format=csv,noheader
python3 -u -m methods.varnet.train --resume \
    --obs-every-k $K \
    --days 32 --dT 200 --epochs $EPOCHS --batch 32 --amp \
    --loss supervised --iter-schedule "$SCHED" --clip-grad 1.0 \
    --outdir $OUTDIR
