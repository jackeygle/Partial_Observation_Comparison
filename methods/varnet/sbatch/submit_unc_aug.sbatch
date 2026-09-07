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
# M=5 as of 2026-09-06 (s0-s4 all reached 150 epochs). The earlier M=1 run had NO Sec. 2.4
# epistemic term at all -- json's all_epistemic_only reported sigma=0, the var_eps floor -- so
# its spread/skill 0.447 was the aleatoric half alone and was NOT comparable to vsb0/ml5 at M=5.
# The aleatoric-only number survives this overwrite: eval_uncertainty.py still reports the
# all_aleatoric_only split, which is exactly what the M=1 run measured.
cd /scratch/work/zhangx29/Thesis_Project
module load scicomp-python-env
srun python3 -m methods.varnet.checks.eval_uncertainty --batch 2 \
     --run-fmt "runs/varnet_aug0_s{}" --members 0 1 2 3 4 \
     --out check_outputs/eval/uncertainty_aug0.json
