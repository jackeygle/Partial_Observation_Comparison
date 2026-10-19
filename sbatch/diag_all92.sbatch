#!/bin/bash
#SBATCH --job-name=diag_all92
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/work/zhangx29/data_all92/logs/diag_all92_%j.log
# Route 2: (1) DINCAE smoke checkpoints of route 1 and route 2, each scored on one Wednesday and
# one Sunday validation day; (2) route-2 DINCAE trained a little longer (8 days, 3 epochs) to see
# whether its dev error settles; (3) PedPred3 one epoch on 2 days with the enkf_opt import path.
set -uo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
export DINCAE_VAR_TRANSFORM=none
nvidia-smi --query-gpu=name --format=csv,noheader
echo "=== (1a) route-1 smoke ckpt, Sunday stats"
python3 -u -m supervisor_evaluation.checks.diag_dincae_all92 --ckpt methods/dincae/runs/smoke_varnone_dev/last.pt
echo "=== (1b) route-2 smoke ckpt, all92 stats"
DATA_VARIANT=all92 python3 -u -m supervisor_evaluation.checks.diag_dincae_all92 --ckpt methods/dincae/runs/_smoke_all92/dincae/last.pt
echo "=== (2) route-2 DINCAE, 8 days, 3 epochs"
DATA_VARIANT=all92 python3 -u -m methods.dincae.train --amp --epochs 3 --save-every 1 --save-from 0 --trajectory-mode per_day \
  --full-field-loss --days 8 --out methods/dincae/runs/_smoke_all92/dincae_8d 2>&1 | grep -v "cache\]" | tail -n 4
DATA_VARIANT=all92 python3 -u -m supervisor_evaluation.checks.diag_dincae_all92 --ckpt methods/dincae/runs/_smoke_all92/dincae_8d/last.pt
echo "=== (3) PedPred3"
export PYTHONPATH="$THESIS_ROOT/methods/enkf/enkf_opt:$PYTHONPATH"
(cd methods/enkf && DATA_VARIANT=all92 python3 -u -m methods.enkf.lcskf.dynamics.train --input-frames 5 --output-frames 5 \
  --seed 0 --epochs 1 --batch 64 --lr 1e-3 --eval-batch 1024 --days 2 --out runs/_smoke_all92/pedpred3 2>&1 | tail -n 4)
echo "=== done"
