#!/bin/bash
#SBATCH --job-name=dyn150_gate
#SBATCH --partition=batch-csl
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=methods/enkf/runs/slurm_%x_%j.out
#
# Gate: succeeds only once the 150-epoch dynamics run has really finished, and pins the epoch
# with the lowest FULL-validation loss under a fixed name. Downstream jobs depend on this, not
# on a job id, because the dynamics run resubmits itself every three hours.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}/methods/enkf"
M=${1:?usage: sbatch gate_dynamics_done.sbatch <run dir relative to methods/enkf>}
last_epoch() { tail -1 "$M/metrics.jsonl" | python3 -c "import sys,json; print(json.loads(sys.stdin.read())['epoch'])"; }
idle=0
while [ "$(last_epoch)" -lt 149 ]; do
  sleep 120
  if [ "$(squeue -u "$USER" -h -o '%j' | grep -c 'dyn5\|pedpred5')" -eq 0 ]; then idle=$((idle+1)); else idle=0; fi
  if [ "$idle" -ge 5 ]; then echo "no dynamics job queued or running, but only at epoch $(last_epoch)/149" >&2; exit 1; fi
done
M="$M" python3 - <<'PY'
import json, os
M = os.environ["M"]
rows = [json.loads(l) for l in open(f"{M}/metrics.jsonl")]
# A diverged epoch reports valid_loss NaN, and NaN compares in a way that can win a min().
# One run already went that way at epoch 30, so drop them before choosing.
finite = [r for r in rows if r["valid_loss"] == r["valid_loss"]]
if not finite:
    raise SystemExit("every epoch has a non-finite validation loss")
best = min(finite, key=lambda r: r["valid_loss"])
json.dump({"epochs_run": len(rows), "epochs_finite": len(finite), "best_epoch": best["epoch"], "best_valid_loss": best["valid_loss"],
           "chosen_on": "the full validation split (269,708 pairs), not a subset"},
          open(f"{M}/dyn150_pick.json", "w"), indent=1)
print(json.dumps({k: best[k] for k in ("epoch", "valid_loss")}))
PY
cp -p "$M/best.pt" "$M/dyn150_best.pt" && echo "pinned $M/dyn150_best.pt"
