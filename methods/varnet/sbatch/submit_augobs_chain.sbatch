#!/bin/bash
#SBATCH --partition=gpu-h200-141g-m
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=methods/varnet/runs/slurm_%x_%j.out
# AUGOBS — aug0 with the observation term of J ALSO a Gaussian NLL, sharing the iterated
# log sigma^2 with the prior term (observed cells only). Everything else is identical to
# submit_aug_chain.sbatch (hidden 96, kt 5, dT 200, schedule, clip 1.0, beta 0), so the only
# difference from runs/varnet_aug0_h96_s<seed> is --obs-nll.
#
# Usage:  sbatch --job-name=varnet_augobs_s0 methods/varnet/sbatch/submit_augobs_chain.sbatch 0 0
#         (args: init seed, nll beta [ignored], generation [internal])
#
# Run ONE segment first, no successor, and check the log before chaining:
#     sbatch --export=ALL,NOCHAIN=1 --job-name=varnet_augobs_s0 \
#            methods/varnet/sbatch/submit_augobs_chain.sbatch 0 0
source /etc/profile 2>/dev/null
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || { echo "submit from the repo root" >&2; exit 1; }
cd "$THESIS_ROOT/methods/varnet"


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

S=${1:?usage: sbatch --job-name=varnet_augobs_sM methods/varnet/sbatch/submit_augobs_chain.sbatch <seed> 0}
BETA=${2:?missing beta}
TAG=augobs   # BETA ignored; the loss is Eq.1 as written
OUTDIR=runs/varnet_${TAG}_h96_s${S}
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
if [ "${NOCHAIN:-0}" = "1" ]; then
    echo "[chain] NOCHAIN=1 -- running only this segment, no successor queued"
else
    cd "$THESIS_ROOT"   # resubmit from the repo root: #SBATCH --output and _env.sh resolve against it
    sbatch --export=NONE --exclude=gpu51 --job-name=$SLURM_JOB_NAME \
           --dependency=afterany:$SLURM_JOB_ID \
           methods/varnet/sbatch/submit_augobs_chain.sbatch $S $BETA $((GEN+1))
    cd "$THESIS_ROOT/methods/varnet"
    echo "[chain] submitted successor (afterany:$SLURM_JOB_ID)"
fi

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
    --loss nll --nll-beta 0.0 --var-eps 1e-6 --augmented-var --obs-nll \
    --iter-schedule "$SCHED" --clip-grad 1.0 \
    --outdir $OUTDIR
