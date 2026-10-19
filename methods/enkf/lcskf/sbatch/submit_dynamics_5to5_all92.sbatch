#!/bin/bash
# ROUTE 2 (DATA_VARIANT=all92) copy of submit_dynamics_5to5.sbatch: same PedPred3 recipe, all 92
# ATC days (63 training days). The relay resubmits THIS file, which sets the variant itself.
#SBATCH --job-name=pedpred5_all92
# Any card with >=32 GB works: the whole training split (~8.7 GB) lives on the device. Listing
# several partitions lets Slurm start on whichever has a free slot first. It has to be in
# the header, not on the command line, because the relay below resubmits THIS file.
#SBATCH --partition=gpu-h200-141g-short,gpu-h200-141g-m,gpu-h100-80g-short,gpu-h100-80g,gpu-a100-80g,gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=methods/enkf/runs/slurm_%x_%j.out
#
# Fresh PedPred3 dynamics model: five true frames -> five true future frames.
# The Python defaults remain backward compatible, so all dimensions are explicit here.
set -uo pipefail
export DATA_VARIANT=all92
# PedPred3 lives in methods/enkf/enkf_opt/pedpred since the 2026-09-24 move (enkf_lab is an empty
# shell); the EnKF scripts put enkf_opt on sys.path themselves, the trainer does not.
export PYTHONPATH="${SLURM_SUBMIT_DIR:-/scratch/work/zhangx29/Thesis_Project}/methods/enkf/enkf_opt:${PYTHONPATH:-}"
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || { echo "submit from the repo root" >&2; exit 1; }
cd "$THESIS_ROOT/methods/enkf"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

script="$THESIS_ROOT/methods/enkf/lcskf/sbatch/submit_dynamics_5to5_all92.sbatch"
original_args=("$@")
child=""
relayed=0

# Early stop (user, 2026-10-13): stop once the validation loss has not reached a new minimum for
# PATIENCE epochs; best.pt keeps the lowest-validation-loss weights. Checked at the start of
# every segment, before a relay, and every 2 minutes while training.
PATIENCE=20
OUT=""; prev=""
for a in "$@"; do [ "$prev" = "--out" ] && OUT="$a"; prev="$a"; done
flat() {
  python3 - "$OUT/metrics.jsonl" "$PATIENCE" <<'PY'
import json, math, sys
try:
    rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
except FileNotFoundError:
    sys.exit(1)
rows = [r for r in rows if math.isfinite(r.get("valid_loss", float("nan")))]
if not rows:
    sys.exit(1)
best = min(rows, key=lambda r: r["valid_loss"])["epoch"]
last = max(r["epoch"] for r in rows)
sys.exit(0 if last - best >= int(sys.argv[2]) else 1)
PY
}
if flat; then echo "[early-stop] no new best validation loss for $PATIENCE epochs: not resuming"; exit 0; fi

relay() {
  if [[ "$relayed" -eq 0 ]] && ! flat; then
    relayed=1
    echo "[relay] time limit approaching; submitting PedPred3 continuation with --resume" >&2
    (cd "$THESIS_ROOT" && sbatch --chdir="$THESIS_ROOT" "$script" "${original_args[@]}" --resume)
  fi
  if [[ -n "$child" ]]; then
    kill -TERM "$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
  fi
  exit 0
}
trap relay USR1

python3 -u -m methods.enkf.lcskf.dynamics.train \
  --input-frames 5 \
  --output-frames 5 \
  "$@" & child=$!
( while kill -0 "$child" 2>/dev/null; do
    sleep 120
    if flat; then
      echo "[early-stop] no new best validation loss for $PATIENCE epochs: stopping"
      kill -TERM "$child" 2>/dev/null; break
    fi
  done ) &
wait "$child"
