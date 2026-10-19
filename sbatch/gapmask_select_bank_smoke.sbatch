#!/bin/bash
#SBATCH --job-name=gapmask_main
#SBATCH --partition=gpu-v100-32g,gpu-v100-16g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/work/zhangx29/data_all92/logs/gapmask_main_%j.log
# Route 2, gap mask on (crowdcore/frame_reliability.py), user-approved 2026-10-14:
#   1. PedPred3 lead-1 re-selection on scorable validation frames (new json; old one kept)
#   2. EnKF residual bank from the selected PedPred3, scorable training frames only
#   3. smoke test of the masked GPU scorers on 2013-08-07 (not a result)
set -euo pipefail
export DATA_VARIANT=all92
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
export PYTHONPATH="$THESIS_ROOT/methods/enkf/enkf_opt:$PYTHONPATH"
nvidia-smi --query-gpu=name --format=csv,noheader

cd "$THESIS_ROOT/methods/enkf"
SEL=runs/pedpred3_lead1_selection_all92_gapmask.json
python3 -u -m supervisor_evaluation.checks.pedpred_select_lead1 $SEL \
    $(ls runs/pedpred3_5to5_s0_all92/epoch_*.pt) $(ls runs/pedpred3_5to5_clip_s0_all92/epoch_*.pt)
BEST=$(python3 -c "
import json, math
rows = json.load(open('$SEL'))
print(min((r for r in rows if math.isfinite(r['lead1_loss_clipped'])), key=lambda r: r['lead1_loss_clipped'])['ckpt'])")
echo "[pedpred3 selected] $BEST"

cd "$THESIS_ROOT"
python3 -u -m methods.enkf.enkf_opt.experiments.build_residual_q_bank \
    --model-checkpoint "$THESIS_ROOT/methods/enkf/$BEST" \
    --out "$THESIS_ROOT/methods/enkf/enkf_opt/experiments/outputs/pedpred3_5to5_train_residual_q_8192_all92.npz"

DINCAE_VAR_TRANSFORM=none python3 -u -m supervisor_evaluation.checks.smoke_gapmask \
    supervisor_evaluation/outputs/dev/gapmask_smoke_20261014 \
    methods/senseiver/runs/senseiver_A_pd_all92/epoch_099.pt \
    methods/varnet/runs/varnet_mse5_h96k3_win_pdfold_s3_all92/ckpt_00080.pt \
    methods/dincae/runs/dincae_ff_pd_varnone_all150_all92
