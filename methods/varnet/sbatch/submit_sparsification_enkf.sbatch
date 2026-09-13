#!/bin/bash
#SBATCH --job-name=sparsenkf
#SBATCH --partition=batch-csl,batch-hsw
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=methods/varnet/check_outputs/sparsenkf_%j.out
# Pure numpy on the EnKF's own exported arrays -- no model inference, so no GPU.
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || { echo "submit from the repo root" >&2; exit 1; }
cd "$THESIS_ROOT/methods/varnet"
srun python3 -m methods.enkf.checks.diag_sparsification_enkf
