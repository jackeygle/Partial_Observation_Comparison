#!/bin/bash
#SBATCH --job-name=corrpaths
#SBATCH --partition=batch-csl,batch-hsw
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/methods/enkf/check_outputs/corrpaths_%j.out
# Separate the filter's two correction paths: the Kalman gain (textbook, scaled by the collapsed
# ensemble covariance) and the undocumented EMA bias term that forecast() subtracts at full
# strength regardless of the gain. Nothing in the repo or the original paper documents the
# second one, and every claim made so far about "the EnKF's assimilation" has been about both
# of them combined. Four arms, one shared RNG stream -- see the module docstring.
#
# CPU only; matches the configuration of record (ensemble 100, radius 7, inflation 1.02,
# proc_noise_scale 0.01).
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project
python3 -u -m methods.enkf.checks.diag_correction_paths "$@"
