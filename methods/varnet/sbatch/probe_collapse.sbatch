#!/bin/bash
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=methods/varnet/runs/slurm_%x_%j.out
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh"
cd "$THESIS_ROOT"
python3 -u -m methods.varnet.checks.probe_solver_collapse "$@"
