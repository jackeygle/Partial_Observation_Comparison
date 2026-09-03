#!/bin/bash
#SBATCH --partition=gpu-h200-141g-m
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/methods/varnet/runs/slurm_%x_%j.out
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
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project/methods/varnet

# --var-h-only PINS the old design. This script produced runs/varnet_vrb{0,1}_s0, whose
# out_var is (C*T, hidden_ch, 1, 1) -- sigma^2 off the LSTM hidden state alone. var_sees_state
# later became the default, so without this flag a --resume here builds a wider out_var than the
# checkpoint holds and dies on load_state_dict. These runs are the baseline the wider read-out
# is measured against, so they must keep reproducing the narrow one.
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

S=${1:?usage: sbatch --job-name=varnet_vrbN_sM sbatch/submit_vr_chain.sbatch <seed> <beta>}
BETA=${2:?missing beta}
TAG=$(python3 -c "print('vrb%d' % round(float('$BETA')))")
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
       sbatch/submit_vr_chain.sbatch $S $BETA $((GEN+1))
echo "[chain] submitted successor (afterany:$SLURM_JOB_ID)"

nvidia-smi --query-gpu=name --format=csv,noheader
python3 -u -m methods.varnet.train --resume \
    --obs-every-k 1 \
    --init-seed $S --data-seed 0 \
    --days 32 --dT 200 --epochs $EPOCHS --batch 32 --amp \
    --loss nll --nll-beta $BETA --var-eps 1e-6 --var-h-only \
    --iter-schedule "$SCHED" --clip-grad 1.0 \
    --outdir $OUTDIR
