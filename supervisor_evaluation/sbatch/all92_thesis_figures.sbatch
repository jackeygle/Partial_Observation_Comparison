#!/bin/bash
#SBATCH --job-name=all92_figs
#SBATCH --partition=gpu-debug,gpu-v100-32g
#SBATCH --gres=gpu:1,min-vram:32G
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=supervisor_evaluation/outputs/logs/all92_figs_%j.log
# Route 2 thesis figures (Thesis_paper20260219): comparison images again with the thesis's two
# fixed example frames added, the example-frame cache, then every figure script with
# THESIS_VARIANT=6methods_all92 (-> thesis_figures/out_6methods_all92/). data_processing is not
# redrawn: it needs the raw CSVs, which are not kept locally, and does not depend on the route.
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
export DATA_VARIANT=all92 EVAL_VARIANT=six_allwalk_all92 DINCAE_VAR_TRANSFORM=none THESIS_VARIANT=6methods_all92
nvidia-smi --query-gpu=name --format=csv,noheader; hostname
EVAL_EXTRA_FRAMES=atc-20130811:32778,atc-20130811:13551 python3 -u supervisor_evaluation/evaluate.py images \
    --data-root /scratch/work/zhangx29/data_all92 --output-dir supervisor_evaluation/outputs/full_6m_all92
python3 -u -m thesis_figures.prepare_frame
for f in pipeline study_area observation time_windows senseiver dincae varnet genn enkf comparison \
         uncertainty_maps attention_block pedpred breakdown five_methods_results; do
    python3 -u -m thesis_figures.$f
done
echo "[done] all92 thesis figures"
