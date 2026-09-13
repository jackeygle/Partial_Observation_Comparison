#!/bin/bash
#SBATCH --job-name=uncaug
#SBATCH --partition=gpu-h200-141g-short,gpu-a100-80g,gpu-v100-32g,gpu-v100-16g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=methods/varnet/check_outputs/uncaug_%j.out
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
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || { echo "submit from the repo root" >&2; exit 1; }
cd "$THESIS_ROOT/methods/varnet"
# hidden=96/kt=5 runs (2026-09-13). Each member is scored at the checkpoint
# checks/select_checkpoint.py chose on the VALIDATION split (select_valid.json),
# the same one compare5 scores -- resolved by model_io.reported_ckpt.
# --batch only trades memory for speed; results do not depend on it. Inference
# runs in eval mode (no second-order graph), ~2 GB per solve, so a V100 is enough.
srun python3 -u -m methods.varnet.checks.eval_uncertainty --batch 8 \
     --run-fmt "runs/varnet_aug0_h96_s{}" --members 0 1 2 3 4 \
     --out check_outputs/eval/uncertainty_aug0.json
