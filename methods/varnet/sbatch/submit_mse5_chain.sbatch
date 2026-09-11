#!/bin/bash
#SBATCH --partition=gpu-h200-141g-m
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/methods/varnet/runs/slurm_%x_%j.out
# MSE5 — the MSE arm of the three-way uncertainty comparison.
#
# Usage:  sbatch --job-name=varnet_mse5_s0 sbatch/submit_mse5_chain.sbatch 0 0
#         (args: init seed, beta [ignored], generation [internal])
#
# Lines 8-35 of this file used to hold submit_vs_chain.sbatch's rationale verbatim -- the
# file was copied from that one and the header was never rewritten, so it argued about a
# sigma^2 read-out that this run does not have. Removed 2026-09-09.
source /etc/profile 2>/dev/null
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project/methods/varnet

# The MSE arm of the three-way comparison, and Lakshminarayanan et al.'s own Table 2 baseline:
# M networks trained on the plain squared loss, whose uncertainty is the EMPIRICAL VARIANCE
# across their predictions rather than a learnt sigma^2. --loss supervised sets predict_var
# False, so these have no variance read-out at all -- that is the point of the comparison.
#
# Architecture, schedule, data and seeds match runs/varnet_vsb0_s* exactly; the ONLY difference
# is the loss. Without that the ensemble spread would mix a seed effect with an architecture
# effect, which is why the existing MSE models (a2/a4/b0, hidden 64/96/32, kt 5/5/3) cannot be
# used as an ensemble even though there are three of them.
#
# The paper's Table 2 finding is that learning the variance beats this by a wide margin. Whether
# that holds here is the question these runs answer.

# GENERATION CAP. The successor is submitted BEFORE training starts, so that a job killed by
# the time limit still has one queued. The cost is that a job which crashes in the first second
# also queues one, and that successor crashes too: on 2026-08-31 a one-line bug in a banner
# print spun this chain through ~250 generations in minutes. The cap turns a code bug into a
# handful of wasted jobs instead of a scheduler flood. 150 epochs at 3h/segment needs ~6.
GEN=${3:-1}
MAX_GEN=12
if [ "$GEN" -gt "$MAX_GEN" ]; then
    echo "[chain] generation $GEN exceeds $MAX_GEN — refusing to continue. Something is"
    echo "        failing immediately rather than running out of time; check the last log."
    exit 1
fi

S=${1:?usage: sbatch --job-name=varnet_mse5_sM sbatch/submit_mse5_chain.sbatch <seed> 0}
BETA=${2:?missing beta}
TAG=mse5   # BETA is ignored here; kept so the chain's argument list is unchanged
OUTDIR=runs/varnet_${TAG}_h96_s${S}
EPOCHS=150
SCHED='0:5:1e-3,25:10:7e-4,50:15:5e-4,75:20:3e-4'

DONE=$(python3 -c "import torch,os; p='$OUTDIR/varnet_last.pt'; print(1 if (os.path.exists(p) and int(torch.load(p,map_location='cpu').get('epoch',-1))+1>=$EPOCHS) else 0)" 2>/dev/null)
if [ "$DONE" = "1" ]; then
    echo "[chain] target $EPOCHS epochs reached — stopping chain."
    exit 0
fi

# --export=NONE is REQUIRED: without it each generation re-runs `module load` on an inherited
# environment, PATH grows by one copy per generation, and after ~6 generations every command
# fails with "Argument list too long".
# --exclude=gpu51: that node is draining and its driver returns CUDA error 802, so every job
# placed there falls back to CPU. TRANSIENT -- remove this once the node is healthy. The real
# protection is train_varnet.py's CUDA guard, which now aborts in seconds instead of running for
# hours on CPU; this exclude just avoids burning chain generations on a known-bad node.
sbatch --export=NONE --exclude=gpu51 --job-name=$SLURM_JOB_NAME --dependency=afterany:$SLURM_JOB_ID \
       sbatch/submit_mse5_chain.sbatch $S $BETA $((GEN+1))
echo "[chain] submitted successor (afterany:$SLURM_JOB_ID)"

# PRIOR WIDTH. hidden 32 -> 96, kt 3 -> 5: the configuration runs/varnet_a4_k1 was trained
# with, promoted to the project baseline on 2026-09-09. It was the best of the three supervised
# capacity points (walkable RMSE 0.230 vs a2 and b0's 0.248), and the three uncertainty arms had
# been built on b0's 32/3 instead, so the headline model and the uncertainty comparison did not
# share a backbone. They do now. GENN goes 9.4k -> 54.2k parameters; the solver is untouched,
# which is why per-epoch cost barely moves (97% of the model is the ConvLSTM).
#
# The run directory carries the h96 tag so these do not resume from -- or overwrite -- the
# hidden=32 checkpoints of the same name. --resume matches on path, and loading a 32-channel
# prior into a 96-channel one is a shape error, not a silent wrong answer, but only if the
# paths are distinct.
nvidia-smi --query-gpu=name --format=csv,noheader
python3 -u -m methods.varnet.train --resume \
    --obs-every-k 1 \
    --hidden 96 --kt 5 \
    --init-seed $S --data-seed 0 \
    --days 32 --dT 200 --epochs $EPOCHS --batch 32 --amp \
    --loss supervised \
    --iter-schedule "$SCHED" --clip-grad 1.0 \
    --outdir $OUTDIR
