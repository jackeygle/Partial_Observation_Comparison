#!/bin/bash
#SBATCH --job-name=select3_all92
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/work/zhangx29/data_all92/logs/select3_all92_%j.log
# Route 2 selection on all walkable validation cells, gap mask on (user, 2026-10-15), one GPU:
#   Senseiver-G (60 epochs) in the background, and the two 4DVarNet runs one after the other
#   (final_wink3: validation selection + batch-1 test inference).
source /etc/profile 2>/dev/null
export DATA_VARIANT=all92
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || exit 1
cd "$THESIS_ROOT"
nvidia-smi --query-gpu=name --format=csv,noheader
DEV=supervisor_evaluation/outputs/dev
python3 -u -m supervisor_evaluation.checks.reselect_all_walkable senseiver_g --shard 0/1 \
    > supervisor_evaluation/outputs/reselect_all_walkable_all92_gapmask/sel_svG_$SLURM_JOB_ID.out 2>&1 &
SVG=$!
status=0
for run in varnet_mse5_h96k3_win_pdfold_s3_all92:varnet_wink3_all92_gapmask_20261015 \
           varnet_aughead_obs_h96k3_win_pdfold_s3_all92:varnet_augwink3_all92_gapmask_20261015; do
    python3 -u -m supervisor_evaluation.checks.final_wink3 --run-dir methods/varnet/runs/${run%%:*} \
        --out $DEV/${run##*:} > $DEV/${run##*:}/final_$SLURM_JOB_ID.out 2>&1 || status=$?
    echo "[select3] ${run%%:*} exit $status"
done
wait $SVG || status=$?
echo "[select3] done, status $status"
exit $status
