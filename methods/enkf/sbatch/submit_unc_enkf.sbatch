#!/bin/bash
#SBATCH --job-name=uncenkf
#SBATCH --partition=batch-csl,batch-hsw
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=methods/enkf/check_outputs/uncenkf_%j.out
# Re-score the EnKF's ensemble spread, this time WITH the constant-sigma null model that the
# 4DVarNet side has had all along. Without it the EnKF row in table B has an empty verdict
# column while every other row is judged against a baseline -- and that column is the only
# thing separating "sigma-hat knows how big the error is on average" from "sigma-hat knows
# WHERE the error is".
#
# Two passes: the constant sigma IS that split's own RMSE, so it cannot be scored in the pass
# that measures it, and Acc streams (~4.8e8 points) so the residuals cannot be kept around.
# Hence the doubled walltime versus the single-pass version.
#
# Pure numpy over the exported npz -- no model inference, so no GPU.
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || { echo "submit from the repo root" >&2; exit 1; }
cd "$THESIS_ROOT/methods/enkf"
srun python3 -u -m methods.enkf.checks.eval_uncertainty_enkf --k 1
