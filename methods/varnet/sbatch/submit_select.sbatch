#!/bin/bash
#SBATCH --job-name=varnet_select
#SBATCH --partition=gpu-h200-141g-short,gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --exclude=gpu51
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/methods/varnet/runs/slurm_%x_%j.out
# Choose the reported epoch for every uncertainty-arm run, on the VALIDATION split.
#
#   sbatch sbatch/submit_select.sbatch                  # all 15 runs, all 7 valid days
#   sbatch sbatch/submit_select.sbatch --days 3         # quicker, for a first look
#
# Must run before compare.compare5, which now refuses to report a 4DVarNet number
# without each run's select_valid.json -- see checks/select_checkpoint.py for why
# varnet_best.pt cannot be used for this.
source /etc/profile 2>/dev/null
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project/methods/varnet

FAILED=""
for ARM in mse5 vsb0 aug0; do
    for S in 0 1 2 3 4; do
        RUN=runs/varnet_${ARM}_h96_s${S}
        echo; echo "==================== $RUN ===================="
        if [ ! -d "$RUN" ]; then echo "[skip] no such run directory"; FAILED="$FAILED $RUN(missing)"; continue; fi
        python3 -u -m methods.varnet.checks.select_checkpoint --run-dir "$RUN" "$@" \
            || FAILED="$FAILED $RUN"
    done
done

echo
if [ -n "$FAILED" ]; then
    echo "[select] FAILED for:$FAILED"
    echo "[select] compare5 will refuse to run until every arm has select_valid.json."
    exit 1
fi
echo "[select] all 15 runs selected — compare.compare5 can run now"
