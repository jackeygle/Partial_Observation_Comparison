#!/bin/bash
#SBATCH --partition=gpu-h200-141g-m
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/4dvarnet_enkf/runs/slurm_%x_%j.out
# AUG — log sigma^2 is ITERATED as part of the solver state, not read off a hidden layer.
#
# Usage:  sbatch --job-name=varnet_aug0_s0 sbatch/submit_aug_chain.sbatch 0 0
#         (args: init seed, nll beta [ignored], generation [internal])
#
# STATUS 2026-09-03: this configuration DOES NOT TRAIN. 251 jobs were submitted between
# 2026-09-02T20:08 and 2026-09-03T12:47; 250 of them FAILED after 2-5 minutes each and left
# NO log files behind, so there is nothing to diagnose from. The generation cap below did not
# stop it. Before submitting this chain again, run ONE job with no chaining and make sure its
# log actually appears -- see "RUN IT WITHOUT THE CHAIN FIRST" below.
#
# The mechanism itself is sound: scratch/smoke_aug.py passes (job 20049770), gates_in goes
# 96 -> 128, sigma^2 stays finite, and the gradient does reach var_cost.alpha_reg (6.15e-04).
# So the failure is at full scale (dT=200, n_iter up to 20, batch 32, AMP) or in the training
# script's no-read-out branch, not in the design.
#
# RUN IT WITHOUT THE CHAIN FIRST:
#     sbatch --job-name=varnet_aug0_s0 --time=00:30:00 sbatch/submit_aug_chain.sbatch 0 0 99
#   Generation 99 is above MAX_GEN, so no successor is queued and the failure is visible in
#   one log instead of two hundred and fifty missing ones.
#
# The design rationale, the R^2 audit it came from, and the ceiling it can reach are all
# in the AUGMENTED STATE block below. (A block of submit_vr_chain.sbatch's rationale
# used to sit here; it argued the OPPOSITE -- that sigma^2 should stay out of the
# iterated state -- because this file was copied from that one. Removed 2026-09-03.)
source /etc/profile 2>/dev/null
module load scicomp-pytorch-env/2026.1
cd /scratch/work/zhangx29/Thesis_Project/4dvarnet_enkf

# AUGMENTED STATE: log sigma^2 is iterated as part of the state, not read off a hidden layer.
#
# The problem this addresses. In the paper's feed-forward net the mean IS the final layer's
# output, so a sigma^2 read out of the same layer shares everything that determines the mean.
# Here the final layer emits an ITERATION INCREMENT and the prediction is x0 minus 20 of them,
# so a read-out on the last hidden state is conditioned on a lossy summary of the solve rather
# than on the reconstruction. Feeding it x_hat helped a little and nowhere near enough, and a
# 24-bin lookup on |x_hat| still beat it -- so the limitation was never model capacity.
#
# What changes. The prior term of J becomes a Gaussian log-likelihood of the prior residual,
#     mean[ (x - Phi(x))^2 e^{-s} + s ]     with s = log sigma^2,
# whose stationary point in s is sigma^2 = (x - Phi(x))^2. That gives sigma^2 a gradient FROM
# THE COST, so the learned optimiser can descend [x, s] jointly, 20 steps, the same way it does
# the field. autograd supplies both gradients from one call.
#
# The ceiling, stated up front. sigma^2 now chases the prior residual, and the same-channel
# prior residual predicts the TRUE error with R^2 0.38 (vx), 0.35 (vy), 0.11 (density)
# (check_outputs/eval/sigma_drivers.json). So this route cannot exceed roughly that, density
# will likely barely move, and the current read-out's rank agreement is about 0.03 -- a tenfold
# gain would still be a partial answer, not a solved problem.
#
# ONE SEED FIRST. This is a real architecture change (C*dT -> 2*C*dT on the optimiser's channel
# axis, 2.7M -> ~4M parameters) and it may simply not train. Train five only if the rank
# agreement actually moves.

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

S=${1:?usage: sbatch --job-name=varnet_aug0_sM sbatch/submit_aug_chain.sbatch <seed> 0}
BETA=${2:?missing beta}
TAG=aug0   # BETA ignored; the loss is Eq.1 as written
OUTDIR=runs/varnet_${TAG}_s${S}
EPOCHS=150
SCHED='0:5:1e-3,25:10:7e-4,50:15:5e-4,75:20:3e-4'

EPOCH_NOW=$(python3 -c "import torch,os; p='$OUTDIR/varnet_last.pt'; print(int(torch.load(p,map_location='cpu').get('epoch',-1)) if os.path.exists(p) else -1)" 2>/dev/null || echo -1)
if [ "$EPOCH_NOW" = "" ]; then EPOCH_NOW=-1; fi
if [ "$EPOCH_NOW" -ge "$((EPOCHS-1))" ]; then
    echo "[chain] target $EPOCHS epochs reached — stopping chain."
    exit 0
fi

# PROGRESS GATE. The generation counter above is not enough: on 2026-09-02/03 this chain ran
# 251 generations anyway. A crash-loop cannot advance the epoch counter, so gate the successor
# on actual progress instead of trusting an argument that has to survive every resubmission.
# The stamp holds the epoch the PREVIOUS generation saw; if this generation sees the same one,
# the previous segment produced nothing and chaining again would just repeat it.
STAMP=$OUTDIR/.chain_last_epoch
if [ -f "$STAMP" ] && [ "$EPOCH_NOW" -le "$(cat "$STAMP")" ]; then
    echo "[chain] previous generation made NO progress (epoch still $EPOCH_NOW) — refusing to"
    echo "        chain. Fix the crash first; run one job with generation 99 to see its log."
    exit 1
fi
mkdir -p "$OUTDIR" && echo "$EPOCH_NOW" > "$STAMP"

# --export=NONE is REQUIRED: without it each generation re-runs `module load` on an inherited
# environment, PATH grows by one copy per generation, and after ~6 generations every command
# fails with "Argument list too long".
# --exclude=gpu51: that node is draining and its driver returns CUDA error 802, so every job
# placed there falls back to CPU. TRANSIENT -- remove this once the node is healthy. The real
# protection is train_varnet.py's CUDA guard, which now aborts in seconds instead of running for
# hours on CPU; this exclude just avoids burning chain generations on a known-bad node.
sbatch --export=NONE --exclude=gpu51 --job-name=$SLURM_JOB_NAME --dependency=afterany:$SLURM_JOB_ID \
       sbatch/submit_aug_chain.sbatch $S $BETA $((GEN+1))
echo "[chain] submitted successor (afterany:$SLURM_JOB_ID)"

nvidia-smi --query-gpu=name --format=csv,noheader
python3 -u train_varnet.py --resume \
    --obs-every-k 1 \
    --init-seed $S --data-seed 0 \
    --days 32 --dT 200 --epochs $EPOCHS --batch 32 --amp \
    --loss nll --nll-beta 0.0 --var-eps 1e-6 --augmented-var \
    --iter-schedule "$SCHED" --clip-grad 1.0 \
    --outdir $OUTDIR
