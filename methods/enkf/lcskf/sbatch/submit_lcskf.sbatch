#!/bin/bash
#SBATCH --job-name=lr_unetkf
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=methods/enkf/runs/slurm_%x_%j.out
#
# Train:
#   sbatch methods/enkf/lcskf/sbatch/submit_lcskf.sbatch train --rank 16 --seed 0
# Smoke evaluation after training:
#   sbatch methods/enkf/lcskf/sbatch/submit_lcskf.sbatch eval \
#     --checkpoint methods/enkf/runs/lowrank_r16_s0/best.pt --only atc-20130616 --frames 1000
set -uo pipefail
# Slurm executes a copied script from its spool, so BASH_SOURCE cannot locate the
# repository.  Relay submissions below explicitly set this directory to repo root.
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || { echo "submit from the repo root" >&2; exit 1; }
cd "$THESIS_ROOT/methods/enkf"
nvidia-smi --query-gpu=name --format=csv,noheader
script="$THESIS_ROOT/methods/enkf/lcskf/sbatch/submit_lcskf.sbatch"
original_args=("$@")
mode="${1:-}"
shift || true

# Five minutes before Slurm's three-hour limit, stop the current Python process and
# submit the same training command with --resume. train.py checkpoints after every
# completed epoch, so at most the in-progress epoch is repeated. Evaluation jobs are
# never relayed.
child=""
relayed=0
relay() {
  if [[ "$mode" == "train" && "$relayed" -eq 0 ]]; then
    relayed=1
    echo "[relay] time limit approaching; submitting continuation with --resume" >&2
    # sbatch records the caller's current directory as SLURM_SUBMIT_DIR.  The
    # script has already cd'ed into methods/enkf, while _env.sh is sourced
    # relative to the repository root on the next run, so submit from there.
    (cd "$THESIS_ROOT" && sbatch --chdir="$THESIS_ROOT" "$script" "${original_args[@]}" --resume)
  fi
  if [[ -n "$child" ]]; then
    kill -TERM "$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
  fi
  exit 0
}
trap relay USR1

case "$mode" in
  train) python3 -u -m methods.enkf.lcskf.train "$@" & child=$! ;;
  eval)  python3 -u -m methods.enkf.lcskf.filter "$@" & child=$! ;;
  diagnose) python3 -u -m methods.enkf.lcskf.checks.diagnose_covariance "$@" & child=$! ;;
  blind-sweep) python3 -u -m methods.enkf.lcskf.checks.sweep_blind_inflation "$@" & child=$! ;;
  pedpred5-sequential) python3 -u -m methods.enkf.lcskf.checks.eval_sequential "$@" & child=$! ;;
  visualize) python3 -u -m methods.enkf.lcskf.checks.visualize_covariance "$@" & child=$! ;;
  compare) python3 -u -m methods.enkf.lcskf.checks.compare_joint_uncertainty "$@" & child=$! ;;
  check) python3 -u -m methods.enkf.lcskf.checks.check_kalman "$@" & child=$! ;;
  *) echo "usage: $0 {train|eval|diagnose|blind-sweep|pedpred5-sequential|visualize|compare|check} [arguments...]" >&2; exit 2 ;;
esac
wait "$child"
