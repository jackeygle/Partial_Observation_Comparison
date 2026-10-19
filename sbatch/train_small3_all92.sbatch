#!/bin/bash
#SBATCH --job-name=small3_all92
#SBATCH --partition=gpu-h200-141g-m,gpu-h200-141g-short,gpu-h100-80g,gpu-h100-80g-short,gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=/scratch/work/zhangx29/data_all92/logs/small3_all92_%j.log
# ROUTE 2 (DATA_VARIANT=all92): Senseiver-A, Senseiver-G and DINCAE trained at the same time on
# ONE GPU (packing test 20913435: about 3x the throughput of running them one after another,
# identical Senseiver results). Each process uses exactly the final recipe of its model; only the
# data differs (all 92 days, 63 training days). Every trainer resumes from its last.pt (written
# each epoch). Relay: 10 min before the 8-h limit the job queues this script again (generation
# argument, at most 8) if any trainer is still running, then stops the trainers; a trainer that
# has already finished starts at its final epoch and exits at once.
set -uo pipefail
export DATA_VARIANT=all92
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh"
cd "$THESIS_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
SV=methods/senseiver/runs; DI=methods/dincae/runs
COMMON_SV="--amp --seed 123 --obs-seed 0 --trajectory-mode per_day --days 0 --stride 4 --valid-days 3 --valid-stride 20 --batch 64 --lr 1e-3 --save-every-epoch"
resume() { [ -f "$1/last.pt" ] && echo "--resume"; }
GEN=${1:-1}
if [ "$GEN" -gt 8 ]; then echo "[relay] generation $GEN > 8, stopping"; exit 1; fi
relay() {
    alive=0; for p in $PA $PG $PD; do kill -0 $p 2>/dev/null && alive=1; done
    if [ "$alive" = 1 ]; then
        (cd "$THESIS_ROOT" && sbatch --export=NONE sbatch/train_small3_all92.sbatch $((GEN+1)))
        echo "[relay] queued generation $((GEN+1))"
    fi
    kill -TERM $PA $PG $PD 2>/dev/null; wait; exit 0
}
trap relay USR1
python3 -u -m methods.senseiver.train $COMMON_SV --epochs 100 --latent-mode abstract --readout decoder \
    --time-window 1 --out $SV/senseiver_A_pd_all92 $(resume $SV/senseiver_A_pd_all92) \
    >> $SV/senseiver_A_pd_all92.log 2>&1 &
PA=$!
python3 -u -m methods.senseiver.train $COMMON_SV --epochs 60 --latent-mode grid --readout direct \
    --time-window 16 --out $SV/sv_G_k16_pd_all92 $(resume $SV/sv_G_k16_pd_all92) \
    >> $SV/sv_G_k16_pd_all92.log 2>&1 &
PG=$!
DINCAE_VAR_TRANSFORM=none python3 -u -m methods.dincae.train --amp --epochs 150 --save-every 1 \
    --save-from 0 --trajectory-mode per_day --full-field-loss \
    --out $DI/dincae_ff_pd_varnone_all150_all92 $(resume $DI/dincae_ff_pd_varnone_all150_all92) \
    >> $DI/dincae_ff_pd_varnone_all150_all92.log 2>&1 &
PD=$!
status=0
for p in $PA $PG $PD; do wait $p || status=$?; done
echo "[done] exit status $status"
exit $status
