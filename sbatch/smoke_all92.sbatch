#!/bin/bash
#SBATCH --job-name=smoke_all92
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/work/zhangx29/data_all92/logs/smoke_all92_%j.log
# ROUTE 2 smoke test: each of the six trainings with its route-2 command on 2 training days
# (one Wednesday, one Sunday) and a tiny budget -- data read, no NaN, checkpoints written.
# Outputs only under */runs/_smoke_all92/.
set -uo pipefail
export DATA_VARIANT=all92
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
nvidia-smi --query-gpu=name --format=csv,noheader
S=_smoke_all92
step() { echo "=== $1 $(date +%T)"; }
step varnet_mse
(cd methods/varnet && python3 -u -m methods.varnet.train --obs-every-k 1 --hidden 96 --kt 3 --init-seed 3 --data-seed 0 \
  --days 2 --dT 200 --epochs 1 --batch 32 --amp --loss supervised --iter-schedule '0:20:3e-4' --clip-grad 1.0 \
  --trajectory-mode per_day --fold-axes ct --normg-scope window --outdir runs/$S/varnet_mse 2>&1 | grep -v "^\[obs\]" | tail -n 6)
step varnet_aug
(cd methods/varnet && python3 -u -m methods.varnet.train --obs-every-k 1 --hidden 96 --kt 3 --init-seed 3 --data-seed 0 \
  --days 2 --dT 200 --epochs 1 --batch 32 --amp --loss nll --nll-beta 0.0 --var-eps 1e-6 --augmented-var --obs-nll --var-head \
  --iter-schedule '0:20:3e-4' --clip-grad 1.0 --trajectory-mode per_day --fold-axes ct --normg-scope window \
  --outdir runs/$S/varnet_aug 2>&1 | grep -v "^\[obs\]" | tail -n 6)
COMMON_SV="--amp --seed 123 --obs-seed 0 --trajectory-mode per_day --days 2 --stride 4 --valid-days 3 --valid-stride 20 --batch 64 --lr 1e-3 --steps 100 --epochs 1"
step senseiver_A
python3 -u -m methods.senseiver.train $COMMON_SV --latent-mode abstract --readout decoder --time-window 1 --out methods/senseiver/runs/$S/A 2>&1 | grep -v "^\[obs\]" | tail -n 4
step senseiver_G
python3 -u -m methods.senseiver.train $COMMON_SV --latent-mode grid --readout direct --time-window 16 --out methods/senseiver/runs/$S/G 2>&1 | grep -v "^\[obs\]" | tail -n 4
step dincae
DINCAE_VAR_TRANSFORM=none python3 -u -m methods.dincae.train --amp --epochs 1 --save-every 1 --save-from 0 --trajectory-mode per_day \
  --full-field-loss --days 2 --out methods/dincae/runs/$S/dincae 2>&1 | tail -n 3
step pedpred3
(cd methods/enkf && python3 -u -m methods.enkf.lcskf.dynamics.train --input-frames 5 --output-frames 5 --seed 0 --epochs 1 --batch 64 \
  --lr 1e-3 --eval-batch 1024 --days 2 --out runs/$S/pedpred3 2>&1 | tail -n 4)
step done
