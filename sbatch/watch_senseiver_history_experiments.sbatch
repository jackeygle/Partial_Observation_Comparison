#!/bin/bash
#SBATCH --job-name=watch_sv_hist
#SBATCH --partition=batch-bdw
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=1-00:00:00
#SBATCH --output=methods/senseiver/runs/slurm_%x_%j.out

# Wait for the two 60-epoch history experiments, then submit matched held-out
# evaluations.  This job uses no GPU while polling; inference itself runs on
# gpu-debug through submit_compare5.sbatch.
set -euo pipefail

ROOT=${SLURM_SUBMIT_DIR:-/scratch/work/zhangx29/Thesis_Project}
cd "$ROOT"

BASELINE=methods/senseiver/runs/capacity/base32_k16_s123
HISTORY=methods/senseiver/runs/history_loss/w05_k16_s123
FRAMEWISE=methods/senseiver/runs/framewise/k16_s123
EVAL_DIR=methods/senseiver/check_outputs/history_refinement/eval
mkdir -p "$EVAL_DIR"

complete() {
    local run_dir=$1
    local metrics="$run_dir/metrics.jsonl"
    [[ -f "$run_dir/best.pt" && -f "$metrics" ]] || return 1
    [[ $(wc -l < "$metrics") -ge 60 ]]
}

submit_eval() {
    local name=$1 ckpt=$2 output=$3
    sbatch --parsable \
        --job-name="$name" \
        --partition=gpu-debug \
        --time=00:30:00 \
        --mem=64G \
        --output="$EVAL_DIR/slurm_%x_%j.out" \
        sbatch/submit_compare5.sbatch \
        --senseiver "$ckpt" --arms '' --no-with-ensemble --out "$output"
}

if [[ ! -e "$EVAL_DIR/.submitted_baseline" ]]; then
    submit_eval ev_sv_base "$BASELINE/best.pt" "$EVAL_DIR/baseline.json"
    touch "$EVAL_DIR/.submitted_baseline"
fi

while [[ ! -e "$EVAL_DIR/.submitted_history" || ! -e "$EVAL_DIR/.submitted_framewise" ]]; do
    if [[ ! -e "$EVAL_DIR/.submitted_history" ]] && complete "$HISTORY"; then
        submit_eval ev_sv_hloss "$HISTORY/best.pt" "$EVAL_DIR/history_w05.json"
        touch "$EVAL_DIR/.submitted_history"
    fi
    if [[ ! -e "$EVAL_DIR/.submitted_framewise" ]] && complete "$FRAMEWISE"; then
        submit_eval ev_sv_frame "$FRAMEWISE/best.pt" "$EVAL_DIR/framewise.json"
        touch "$EVAL_DIR/.submitted_framewise"
    fi
    printf '[watch] %s history=%s/60 framewise=%s/60\n' \
        "$(date --iso-8601=seconds)" \
        "$(wc -l < "$HISTORY/metrics.jsonl" 2>/dev/null || echo 0)" \
        "$(wc -l < "$FRAMEWISE/metrics.jsonl" 2>/dev/null || echo 0)"
    sleep 120
done

echo "[watch] all matched evaluations submitted; waiting for result files"
while [[ ! -s "$EVAL_DIR/baseline.json" || ! -s "$EVAL_DIR/history_w05.json" || ! -s "$EVAL_DIR/framewise.json" ]]; do
    sleep 120
done
python3 methods/senseiver/checks/summarize_history_refinement.py --eval-dir "$EVAL_DIR"
echo "[watch] final summary written"
