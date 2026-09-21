#!/bin/bash
#SBATCH --job-name=pedpred5_s0
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=methods/enkf/runs/slurm_%x_%j.out
#
# Fresh PedPred3 dynamics model: five true frames -> five true future frames.
# The Python defaults remain backward compatible, so all dimensions are explicit here.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || { echo "submit from the repo root" >&2; exit 1; }
cd "$THESIS_ROOT/methods/enkf"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

script="$THESIS_ROOT/methods/enkf/lcskf/sbatch/submit_dynamics_5to5.sbatch"
original_args=("$@")
child=""
relayed=0

relay() {
  if [[ "$relayed" -eq 0 ]]; then
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
wait "$child"
