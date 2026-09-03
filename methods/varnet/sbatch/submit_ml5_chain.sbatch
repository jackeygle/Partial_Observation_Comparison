#!/bin/bash
# RETIRED 2026-08-25. This script CANNOT reproduce the checkpoints it originally made.
# It trained runs/ml5_s*/ back when the model had (a) a separate VarianceHead module and
# (b) a per-channel sigma^2 floor of 0.25^2 x MAD. Both were deleted from the source, so
# running this now silently trains the new architecture instead. Kept only as a record of
# the flags those checkpoints were made with. The live chain is submit_vr_chain.sbatch.

#SBATCH --partition=gpu-h200-141g-m
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/methods/varnet/runs/slurm_%x_%j.out
# ML5 — one member of a deep ensemble (Lakshminarayanan et al. 2017, arXiv:1612.01474).
#
# Usage:  sbatch --job-name=varnet_ml5_s0 sbatch/submit_ml5_chain.sbatch 0
#         ...                        s1                                  1   ... up to s4
#         (second argument is --init-seed; k is fixed at 1, see below)
#
# What differs from runs/varnet_b0_k1, which is the point of comparison:
#
#   loss      supervised (Eq.14, plain MSE)  ->  nll (their Eq.1, Gaussian NLL)
#   +         a VarianceHead (548 parameters, 0.03% of the model) predicting sigma^2
#   init      --init-seed varies per member; --data-seed stays 0 for ALL members
#   solver    unchanged.  prior unchanged.  J(x) unchanged.  curriculum unchanged.
#
# Everything else is byte-for-byte b0's configuration, so the RMSE difference measures the
# cost of the NLL objective and nothing else.
#
# Why NLL rather than the cheaper option: their Table 2 / Fig.7 show that an ensemble of
# MSE-trained nets scored by the empirical variance of its members is badly overconfident (80%
# interval covering 20% of observations), and that a single NLL-trained net beats a TEN-member
# MSE ensemble on NLL by 4x (Boston 2.55 vs 10.61). Learning the variance is what matters;
# ensembling adds a little on top (2.55 -> 2.41).
#
# Why --data-seed is held fixed: their Sec. 2.4 varies only the initialisation ("We used the
# entire training dataset to train each network"; bagging is reported to hurt). Our old single
# --seed also drove the robot routes and the observation noise, so members would each have
# seen a different observation realisation, which would confound the ensemble's disagreement
# with observation noise.
#
# NOTE these runs were trained when VarianceHead still had a per-channel sigma^2 floor. The
# floor has since been removed from the model; sbatch/submit_nf5_chain.sbatch is the retrained
# version without it. These checkpoints still load, with the stale buffer dropped.
#
# --nll-beta 1.0, not the paper's Eq.1 as written (beta=0). The first attempt at beta=0 was
# killed at epoch 42: sigma^2 collapsed onto the floor that then existed, so the loss became a
# fixed weighted MSE with weights 1/(2*floor) = [154, 5, 67, 11905]; vx, which carries 58.7% of
# the error, had the smallest weight and came out +203% worse than b0 (+52% full-state RMSE). See
# runs/FAILED_ml5_beta0.json for the curves and losses.py:nll_loss for the mechanism. beta=1
# rescales the mean's gradient to MSE's without changing what sigma^2 is fitted to.
#
# Not done, deliberately: adversarial training (their Sec. 2.3). It is optional, and their
# Table 2 states it "does not significantly help on these benchmarks" for regression.
#
# Cost: ~16 GPU-hours per member at 150 epochs (b0 took 5x 3h + 11min). Submit s0 FIRST and
# look at its first epochs before launching s1..s4 -- a broken head or a diverging NLL shows
# up within an hour, and there is no reason to spend 63 more GPU-hours to discover it.
#
#   Each job: (1) if the checkpoint already reached the target -> stop the chain;
#             (2) otherwise submit a successor (afterany) that will --resume when this job
#                 finishes OR is killed at the time limit;
#             (3) train with --resume.
source /etc/profile 2>/dev/null
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project/methods/varnet

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

S=${1:?usage: sbatch --job-name=varnet_ml5_sN sbatch/submit_ml5_chain.sbatch <init_seed>}
# k=1 only for now: the EnKF reliability diagram we compare against was computed at k=1, and
# doing both k would double an already 79-GPU-hour experiment. k=4 is a follow-up.
K=1
OUTDIR=runs/varnet_ml5_s${S}
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
# exceeded ARG_MAX and every command failed with "Argument list too long".
sbatch --export=NONE --job-name=$SLURM_JOB_NAME --dependency=afterany:$SLURM_JOB_ID \
       sbatch/submit_ml5_chain.sbatch $S
echo "[chain] submitted successor (afterany:$SLURM_JOB_ID)"

# (3) train. --init-seed varies, --data-seed pinned to 0 across all members.
nvidia-smi --query-gpu=name --format=csv,noheader
python3 -u -m methods.varnet.train --resume \
    --obs-every-k $K \
    --init-seed $S --data-seed 0 \
    --days 32 --dT 200 --epochs $EPOCHS --batch 32 --amp \
    --loss nll --nll-beta 1.0 --iter-schedule "$SCHED" --clip-grad 1.0 \
    --outdir $OUTDIR
