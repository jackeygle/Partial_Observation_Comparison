#!/bin/bash
#SBATCH --job-name=gpu_packing
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=supervisor_evaluation/outputs/dev/gpu_packing/packing_%j.log
# Throughput test: Senseiver-A, Senseiver-G and DINCAE (the final recipes, small fixed
# workloads, Sunday data) each alone on one A100, then all three at once on the same GPU.
# GPU utilisation sampled every 2 s. Outputs only under outputs/dev/gpu_packing.
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
O=supervisor_evaluation/outputs/dev/gpu_packing/$SLURM_JOB_ID; mkdir -p $O
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 2 > $O/gpu_samples.csv &
SMI=$!
COMMON_SV="--amp --seed 123 --trajectory-mode per_day --days 4 --valid-days 1 --steps 600 --epochs 1 --batch 64"
run_A() { timeout 480 python3 -u -m methods.senseiver.train $COMMON_SV --latent-mode abstract --readout decoder --time-window 1 --out $O/$1_svA > $O/$1_svA.log 2>&1; }
run_G() { timeout 480 python3 -u -m methods.senseiver.train $COMMON_SV --latent-mode grid --readout direct --time-window 16 --out $O/$1_svG > $O/$1_svG.log 2>&1; }
run_D() { DINCAE_VAR_TRANSFORM=none timeout 480 python3 -u -m methods.dincae.train --amp --trajectory-mode per_day --full-field-loss --days 16 --epochs 2 --save-every 1000 --out $O/$1_dincae > $O/$1_dincae.log 2>&1; }
mark() { echo "$(date +%s.%N) $1" | tee -a $O/marks.txt; }
mark solo_A_start; run_A solo; mark solo_A_end
mark solo_G_start; run_G solo; mark solo_G_end
mark solo_D_start; run_D solo; mark solo_D_end
mark packed_start
( run_A packed; mark packed_A_end ) &
( run_G packed; mark packed_G_end ) &
( run_D packed; mark packed_D_end ) &
wait %2 %3 %4 2>/dev/null; wait $(jobs -p | grep -v $SMI) 2>/dev/null
mark packed_end
kill $SMI
echo "[done]"
