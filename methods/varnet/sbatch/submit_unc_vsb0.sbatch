#!/bin/bash
#SBATCH --job-name=uncvsb0
#SBATCH --partition=gpu-h200-141g-short,gpu-a100-80g,gpu-v100-32g,gpu-v100-16g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=methods/varnet/check_outputs/uncvsb0_%j.out
# The 5-member deep ensemble, combined by Lakshminarayanan et al. Sec. 2.4 moment matching:
#   mu*     = mean_m mu_m
#   sigma*^2= mean_m sigma_m^2 + Var_m(mu_m)     (population variance, unbiased=False)
# Scored on 7 test days with their Sec. 3.1 metrics (NLL, RMSE) and their App. A.2 calibration
# curve. The old ml5 numbers in the deck came from a model whose VarianceHead no longer exists
# and must be replaced wholesale by these.
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || { echo "submit from the repo root" >&2; exit 1; }
cd "$THESIS_ROOT/methods/varnet"
# hidden=96/kt=5 runs (2026-09-13). Each member is scored at the checkpoint
# checks/select_checkpoint.py chose on the VALIDATION split (select_valid.json),
# the same one compare5 scores -- resolved by model_io.reported_ckpt.
# --batch only trades memory for speed; results do not depend on it. Inference
# runs in eval mode (no second-order graph), ~2 GB per solve, so a V100 is enough.
srun python3 -u -m methods.varnet.checks.eval_uncertainty --batch 8 \
     --run-fmt "runs/varnet_vsb0_h96_s{}" --members 0 1 2 3 4 \
     --out check_outputs/eval/uncertainty_vsb0.json
