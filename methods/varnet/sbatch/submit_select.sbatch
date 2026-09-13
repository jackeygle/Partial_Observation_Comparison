#!/bin/bash
#SBATCH --job-name=varnet_select
#SBATCH --partition=gpu-h200-141g-short,gpu-a100-80g,gpu-v100-32g,gpu-v100-16g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --exclude=gpu51
#SBATCH --output=methods/varnet/runs/slurm_%x_%j.out
# Choose the reported epoch for every uncertainty-arm run, on the VALIDATION split.
#
#   sbatch methods/varnet/sbatch/submit_select.sbatch                  # all 15 runs, all 7 valid days
#   sbatch methods/varnet/sbatch/submit_select.sbatch --days 3         # quicker, for a first look
#
# To split the 15 runs across several parallel jobs (each run is independent and
# ~11 min serially on one GPU; 15 in one job is ~2.75h with no parallelism at all):
#   sbatch --export=ALL,RUNS="mse5_h96_s4" methods/varnet/sbatch/submit_select.sbatch
#   sbatch --export=ALL,RUNS="vsb0_h96_s0 vsb0_h96_s1 vsb0_h96_s2 vsb0_h96_s3 vsb0_h96_s4" methods/varnet/sbatch/submit_select.sbatch
#   sbatch --export=ALL,RUNS="aug0_h96_s0 aug0_h96_s1 aug0_h96_s2 aug0_h96_s3 aug0_h96_s4" methods/varnet/sbatch/submit_select.sbatch
# RUNS is space-separated "<arm>_h96_s<seed>" names (must match the run directory
# suffix exactly -- runs/varnet_<name>/ -- or select_checkpoint.py fails fast with
# "no ckpt_*.pt", it does not silently score the wrong run); unset/empty = all 15.
#
# Must run before compare.compare5, which now refuses to report a 4DVarNet number
# without each run's select_valid.json -- see checks/select_checkpoint.py for why
# varnet_best.pt cannot be used for this.
source /etc/profile 2>/dev/null
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || { echo "submit from the repo root" >&2; exit 1; }
cd "$THESIS_ROOT/methods/varnet"

if [ -z "${RUNS:-}" ]; then
    RUNS=""
    for ARM in mse5 vsb0 aug0; do for S in 0 1 2 3 4; do RUNS="$RUNS ${ARM}_h96_s${S}"; done; done
fi

FAILED=""
for NAME in $RUNS; do
    RUN=runs/varnet_${NAME}
    echo; echo "==================== $RUN ===================="
    if [ ! -d "$RUN" ]; then echo "[skip] no such run directory"; FAILED="$FAILED $RUN(missing)"; continue; fi
    python3 -u -m methods.varnet.checks.select_checkpoint --run-dir "$RUN" "$@" \
        || FAILED="$FAILED $RUN"
done

echo
if [ -n "$FAILED" ]; then
    echo "[select] FAILED for:$FAILED"
    echo "[select] compare5 will refuse to run until every arm has select_valid.json."
    exit 1
fi
echo "[select] all 15 runs selected — compare.compare5 can run now"
