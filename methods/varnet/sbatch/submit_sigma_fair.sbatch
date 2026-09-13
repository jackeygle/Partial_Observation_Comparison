#!/bin/bash
#SBATCH --job-name=sigfair
#SBATCH --partition=gpu-h200-141g-m
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --output=methods/varnet/check_outputs/sigfair_%j.out
#SBATCH --exclude=gpu51
# Fair four-way test of the learnt sigma^2: raw / globally rescaled / constant / 1-D lookup,
# every fitted quantity taken from the valid split and applied unchanged to the 7 test days.
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || { echo "submit from the repo root" >&2; exit 1; }
cd "$THESIS_ROOT/methods/varnet"
srun python3 -m methods.varnet.checks.eval_sigma_fair
