#!/bin/bash
#SBATCH --job-name=gaincensus
#SBATCH --partition=batch-csl,batch-hsw
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/methods/enkf/check_outputs/gaincensus_%j.out
# Measure the Kalman gain instead of estimating it. diag_correction_paths.py showed the gain
# contributes nothing; this says which term in P_yy is responsible -- the collapsed ensemble,
# the observation noise, or the hard-coded regularization=1e-3 guard. Read-only census hook,
# off by default in enkf_opt, so the shipped path is untouched.
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project
python3 -u -m methods.enkf.checks.diag_gain_census "$@"
