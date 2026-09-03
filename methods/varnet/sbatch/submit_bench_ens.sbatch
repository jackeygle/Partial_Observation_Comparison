#!/bin/bash
#SBATCH --job-name=benchens
#SBATCH --partition=batch-csl,batch-hsw
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=check_outputs/benchens_%j.out
# Inference-time benchmark for the meeting: EnKF vs one 4DVarNet member vs the 5-member deep
# ensemble, all on the SAME node in ONE job, interleaved and repeated (see bench_speed.py's
# header for why). 4 cores, matching the earlier bench_speed_b0 run so the two are comparable.
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project/methods/varnet
srun python3 -m compare.bench_speed --frames 1000 --repeats 5 \
     --ens-fmt "runs/varnet_vsb0_s{}" --ens-members 0 1 2 3 4 \
     --out check_outputs/eval/bench_speed_vsb0.json
