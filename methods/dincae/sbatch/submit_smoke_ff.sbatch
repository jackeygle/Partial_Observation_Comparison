#!/bin/bash
#SBATCH --job-name=dincff_smoke
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/methods/dincae/runs/slurm_%x_%j.out
# Smoke test for --full-field-loss before spending 16-27 GPU-h on the real run.
# No cache: the ff targets need their own 13 GB and there is no point writing it for a smoke.
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project/methods/dincae
srun python3 -u -m methods.dincae.train --full-field-loss --amp \
     --days 2 --epochs 2 --max-frames 3000 --cache-dir "" \
     --out runs/smoke_ff
