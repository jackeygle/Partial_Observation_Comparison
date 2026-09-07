#!/bin/bash
#SBATCH --job-name=uncml5
#SBATCH --partition=gpu-h200-141g-m
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --exclude=gpu51
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/methods/varnet/check_outputs/uncml5_%j.out
# ml5 = beta-NLL (--nll-beta 1.0), default prior, M=5. Re-scored to pick up the blind-cell
# constant-sigma baseline added to eval_uncertainty.py, so table B can be judged entirely on
# blind cells -- the same cell set the accuracy table uses.
#
# eval_uncertainty.py's defaults already point at ml5 (--run-fmt runs/varnet_ml5_s{},
# members 0-4, out uncertainty_ml5.json); spelled out here anyway so the file says what it runs.
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project/methods/varnet
srun python3 -u -m methods.varnet.checks.eval_uncertainty --batch 2 \
     --run-fmt "runs/varnet_ml5_s{}" --members 0 1 2 3 4 \
     --out check_outputs/eval/uncertainty_ml5.json
