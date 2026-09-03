#!/bin/bash
#SBATCH --partition=gpu-h200-141g-m
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/4dvarnet_enkf/runs/slurm_%x_%j.out
# VR — sigma^2 from the solver's SECOND OUTPUT LAYER instead of a separate module.
#
# Usage:  sbatch --job-name=varnet_vrb1_s0 sbatch/submit_vr_chain.sbatch 0 1.0
#         sbatch --job-name=varnet_vrb0_s0 sbatch/submit_vr_chain.sbatch 0 0.0
#         (args: init seed, nll beta)
#
# Why. Lakshminarayanan Sec. 2.2.1 says "a network that outputs two values in the final layer,
# corresponding to the predicted mean and variance". Our previous design bolted a separate
# pointwise module onto x_hat instead, and the audit showed what that cost: that head keyed off
# distance-to-observation, a feature which predicts the true error with R^2 of roughly 0.01-0.03,
# while ignoring |x_hat|, which predicts it with 0.21-0.45. See checks/diag_sigma_drivers.py.
#
# The read-out sees h, the LSTM hidden state after the last of the 20 descent steps: it carries
# every gradient of J taken along the way, a 3x3 spatial context per step, and all-to-all
# mixing across the 200 frames. A pointwise head on x_hat sees one cell.
#
# sigma^2 is deliberately NOT put into the iterated state. It has no term in J -- the
# observation term has no sigma^2 component and Phi maps state channels to state channels -- so
# an augmented state would leave it unconstrained by the cost. Constraining it properly needs a
# prior on the covariance itself; Beauchamp et al. (AIES 2025) use an SPDE prior for that, which
# is a different piece of work.
#
# Controlled against runs/varnet_nf5_s0: identical config, identical seeds, identical 150
# epochs. The ONLY difference is where sigma^2 is read from. Any change is attributable.
#
# beta: 1.0 reproduces nf5's loss exactly, so it isolates the read-out. 0.0 is Eq.1 verbatim,
# which is what we would rather present -- but the NLL is unbounded below as sigma^2 -> 0, so
# --var-eps 1e-6 is doing real work there and this run may still misbehave. Watch epoch 1-10.
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
