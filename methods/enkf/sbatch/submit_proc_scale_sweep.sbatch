#!/bin/bash
#SBATCH --job-name=procscale
#SBATCH --partition=batch-csl,batch-hsw
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/methods/enkf/check_outputs/procscale_%j.out
# Sweep the process-noise injection multiplier that forecast() hard-codes at 0.01.
#
# README.md currently says the ensemble collapse is "not a tuning problem ... no amount of
# extra process noise or inflation fixes it", inferred from diag_enkf_spread_growth.py finding
# the surrogate contractive with the noise switched OFF. But a contractive map with per-step
# injection settles at an equilibrium proportional to the injection, and the shipped run shows
# exactly that: the spread hits its floor in ~5 frames, sits there for the remaining ~39,800,
# and that floor equals 1.0-1.9x the injected level. So the scale may well be a one-constant
# fix. Whether the spread's *spatial structure* improves is a separate question this also
# measures (AUSE / Spearman / within-frame CV) -- scaling a spatially uniform injection should
# not fix a spatially uniform sigma field.
#
# Diagnosis, not retuning: the reported EnKF numbers stay at the shipped 0.01.
#
# CPU only -- a Kalman analysis gets nothing from a GPU, and this is the configuration of
# record (ensemble 100, radius 7, inflation 1.02, enkf_opt whose default gain path is
# bit-identical to enkf_lab).
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project
python3 -u -m methods.enkf.checks.diag_proc_scale_sweep "$@"
