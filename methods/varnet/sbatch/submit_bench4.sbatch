#!/bin/bash
#SBATCH --job-name=bench4
#SBATCH --partition=batch-csl,batch-hsw
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/methods/varnet/check_outputs/bench4_%j.out
# Inference speed for all four methods, measured interleaved and repeated within one job on
# the same node over the same frames (see bench_speed.py's header for why).
# Adds DINCAE and Senseiver -- previously only EnKF and 4DVarNet were covered, so the table
# in the meeting deck was incomplete.
# 4 cores, matching the existing bench_speed_vsb0 run so the two results are comparable.
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project/methods/varnet
srun python3 -u -m compare.bench_speed --frames 1000 --repeats 5 \
     --ens-fmt "runs/varnet_vsb0_s{}" --ens-members 0 1 2 3 4 \
     --dincae-ckpt ckpt_00070.pt \
     --out check_outputs/eval/bench_speed_four.json
