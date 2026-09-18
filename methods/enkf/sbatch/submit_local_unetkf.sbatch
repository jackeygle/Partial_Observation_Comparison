#!/bin/bash
#SBATCH --job-name=local_unetkf
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=methods/enkf/runs/slurm_%x_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/sbatch/_env.sh" || { echo "submit from the repo root" >&2; exit 1; }
cd "$THESIS_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mode="${1:-}"
shift || true
case "$mode" in
  audit)
    python3 -u -m methods.enkf.checks.audit_pedpred_bias "$@"
    ;;
  static)
    python3 -u -m methods.enkf.checks.build_static_local_covariance "$@"
    ;;
  eval-static)
    python3 -u -m methods.enkf.checks.eval_static_local_covariance "$@"
    ;;
  train)
    python3 -u -m methods.enkf.local_unetkf.train "$@"
    ;;
  eval-learned)
    python3 -u -m methods.enkf.checks.eval_local_unetkf "$@"
    ;;
  eval-sigma-gate)
    python3 -u -m methods.enkf.checks.eval_sigma_diagonal_gate "$@"
    ;;
  check)
    python3 -u -m methods.enkf.checks.check_local_covariance_targets "$@"
    ;;
  *)
    echo "usage: $0 {audit|static|eval-static|train|eval-learned|eval-sigma-gate|check} [arguments...]" >&2
    exit 2
    ;;
esac
