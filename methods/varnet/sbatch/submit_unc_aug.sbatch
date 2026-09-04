#!/bin/bash
#SBATCH --job-name=uncaug
#SBATCH --partition=gpu-h200-141g-m
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --exclude=gpu51
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/methods/varnet/check_outputs/uncaug_%j.out
# The other half of the augmented-state result. Its per-cell RANKING is measured
# (sparsification_aug0.json: Spearman 0.435/0.449/0.416/0.440 against the read-out's
# 0.036/0.035/0.002/0.040) but nothing yet says whether the variance is on the right SCALE.
# Those two are independent -- an ordering can be right while every interval is the wrong width
# -- so a claim about this design needs both.
#
# One member, so there is no Sec. 2.4 epistemic term here; sigma^2 is the solver's own iterated
# variance alone.
cd /scratch/work/zhangx29/Thesis_Project
module load scicomp-python-env
srun python3 -m methods.varnet.checks.eval_uncertainty --batch 2 \
     --run-fmt "runs/varnet_aug0_s{}" --members 0 \
     --out check_outputs/eval/uncertainty_aug0.json
