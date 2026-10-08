#!/bin/bash
# Retrain the four fixed-route final models on per-day robot routes
# =================================================================
#
# Why: of the six final models, only the augmented 4DVarNet (varnet_aughead_obs_s0) was trained
# with trajectory_mode=per_day -- 32 distinct route realizations, one per training day. The other
# four observation-input models were trained with `fixed`: ONE realization replayed on all 32
# days. Every method is tested on the same unseen per-day routes, so the test side is fair, but
# the proposed model had 32x the training-route diversity of the baselines it is compared with,
# which confounds the headline "the augmentation helps" comparison in its own favour. These four
# runs remove that confound by putting every method on the same routes.
# (The EnKF needs no run: its PedPred3 surrogate is trained on full ground-truth states and
# never sees an observation, so it has no training routes at all.)
#
# Recipes are reproduced from each final checkpoint's own saved args; only trajectory_mode (and,
# for Senseiver, the observation base seed) differs. Every run writes to a NEW directory, so the
# existing fixed-route runs and checkpoints are untouched and stay reproducible.
#
# TWO THINGS THAT WOULD HAVE MADE THIS WASTED COMPUTE, both fixed before submitting:
#   1. Senseiver's per_day seed was the day's INDEX in the split (base 123 -> seeds 123..154),
#      while 4DVarNet/DINCAE use crowdcore.observation_model.day_seed = base + the date's
#      ordinal (~735,0xx). A per_day Senseiver would have trained on routes no other method
#      ever saw. methods/senseiver/dataset.py now delegates to day_seed, and --obs-seed 0 puts
#      it on the same base seed. `fixed` is unchanged, so existing checkpoints keep their data.
#   2. DINCAE caches encoded days to disk; its cache key already includes the day's seed
#      (..._sd0_...), so per_day builds its own ~13 GB set instead of silently reusing the
#      fixed-route observations. Check free space before starting.
#
# Usage:
#   bash sbatch/submit_perday_retrain.sh                      # dry run: print all four
#   bash sbatch/submit_perday_retrain.sh --submit             # submit all four
#   bash sbatch/submit_perday_retrain.sh --submit --only 3,4  # submit only those steps
#
# --only exists so the two cheap Senseiver runs (steps 3-4, ~7 GPU-h) can go first: they are
# the ones whose data path changed (the day_seed delegation), so their first log lines confirm
# the new seeding on real training data before 37 GPU-h of 4DVarNet and DINCAE ride on it.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT=$PWD
RUN=0; ONLY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --submit) RUN=1 ;;
        --only)   ONLY=",${2//[[:space:]]/},"; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done
STEP=0

# Queue width. Each method's own sbatch file hard-codes a narrow --partition (H200-short plus
# one V100 flavour); with 372 GPU jobs queued site-wide that is what the wait is made of, not
# the GPUs. Every partition below has nodes with far more than the 96G of host RAM these jobs
# ask for, so the only thing that gates eligibility is the WALL CLOCK: gpu-h200-141g-m -- 14
# nodes x 8 H200, the largest pool here -- caps at 8h, so a job that asks for the script's
# default 24h can never land there. Asking for what the run actually needs (measured: Senseiver
# A ~4h, G ~3h, a 4DVarNet segment 3h) both unlocks that partition and makes the job
# backfill-eligible. Skipped on purpose: gpu-amd (ROCm, the module torch is CUDA),
# gpu-grace-h200-141g (aarch64), gpu-b300-288g-short (Blackwell, newest, 2 nodes),
# gpu-debug (30 min) and the interactive partitions.
# Segment length: 8h, which is the LONGEST request that still fits gpu-h200-141g-m's 8h cap,
# i.e. the shortest wait for the largest pool. Going shorter buys no further partitions (the
# 8h cap is the only binding limit) and costs wall clock, because methods/senseiver and
# methods/dincae queue their successor from a USR1 trap 20 min before the wall clock -- so each
# extra segment pays a full queue wait SERIALLY (372 GPU jobs were queued site-wide when this
# was measured). methods/varnet is the exception: it submits its successor with
# --dependency=afterany BEFORE training starts, so its 3h segments overlap their own queue wait.
# Measured restart overhead is ~290 s per segment (data bank rebuild), k=1 and k=16 alike.
PARTS=gpu-h200-141g-m,gpu-h200-141g-short,gpu-h100-80g,gpu-h100-80g-short,gpu-a100-80g,gpu-v100-32g,gpu-v100-16g

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
skip() { STEP=$((STEP+1)); [ -n "$ONLY" ] && [ "${ONLY#*,$STEP,}" = "$ONLY" ]; }
go() {
    printf '  %s\n' "$*"
    if [ "$RUN" = 1 ]; then eval "$@"; else echo "    (dry run)"; fi
}

if skip; then echo "  (skipped by --only)"; else
say "[1/4] 4DVarNet MSE baseline -- 150 epochs, ~15.6 GPU-h, self-chaining 3h segments"
# Only seed 3 is retrained: supervisor_evaluation/evaluate.py takes varnet_mse from
# varnet_mse5_h96_s3/ckpt_00080.pt alone -- the five-seed MSE ensemble is not part of the
# final comparison (the MSE arm has no variance read-out and sits out the uncertainty table).
go "sbatch --partition=$PARTS --job-name=varnet_mse5_pd_s3 \
    methods/varnet/sbatch/submit_mse5_chain.sbatch 3 0 1 per_day runs/varnet_mse5_h96_pd_s3"

fi

if skip; then echo "  (skipped by --only)"; else
say "[2/4] DINCAE -- 150 epochs, full-field loss, ~21 GPU-h = 3 segments (+ ~13 GB of cache)"
go "EPOCHS=150 OUT=$ROOT/methods/dincae/runs/dincae_ff_pd \
    sbatch --partition=$PARTS --time=08:00:00 --job-name=dincae_pd \
    methods/dincae/sbatch/submit_train.sbatch --trajectory-mode per_day"

fi

if skip; then echo "  (skipped by --only)"; else
say "[3/4] Senseiver A (abstract latent, k=1) -- 100 epochs, ~2.5 GPU-h in one segment"
go "EPOCHS=100 OUT=$ROOT/methods/senseiver/runs/senseiver_A_pd \
    CHAIN_PARTITIONS=$PARTS CHAIN_TIME=08:00:00 \
    sbatch --partition=$PARTS --time=08:00:00 --job-name=sv_A_pd \
    methods/senseiver/sbatch/submit_train.sbatch \
    --days 32 --save-every-epoch --trajectory-mode per_day --obs-seed 0"

fi

if skip; then echo "  (skipped by --only)"; else
say "[4/4] Senseiver G (grid latent, k=16) -- 60 epochs, ~16.6 GPU-h = 3 chained segments"
go "EPOCHS=60 OUT=$ROOT/methods/senseiver/runs/sv_G_k16_pd \
    CHAIN_PARTITIONS=$PARTS CHAIN_TIME=08:00:00 \
    sbatch --partition=$PARTS --time=08:00:00 --job-name=sv_G_pd \
    methods/senseiver/sbatch/submit_train.sbatch \
    --latent-mode grid --readout direct --time-window 16 --save-every-epoch \
    --trajectory-mode per_day --obs-seed 0"

fi

cat <<'NOTE'

After the four runs finish, the comparison is NOT updated yet. Still to do, in order:
  1. Epoch selection on the validation split, per method, on the same score as the existing
     models (blind walkable pooled RMSE, all seven validation days, four channels, clipped):
       methods/varnet/sbatch/submit_select.sbatch        (4DVarNet)
       methods/senseiver/sbatch/select_checkpoint.sbatch (both Senseivers)
       methods/dincae  -- sel_dincae_* recipe in methods/dincae/sbatch/
  2. Point supervisor_evaluation/evaluate.py's FINAL_CONFIG at the new checkpoints.
  3. Re-fit the uncertainty calibration constant alpha IN THE REPORTING CONVENTION (it was
     re-fit once before when the convention changed and moved from 0.5 to 2 -- do not carry a
     constant across conventions).
  4. Re-run the evaluation once on the seven test days, then regenerate tables and figures.
  5. Update section 4.6: with all five trained models on per_day the asymmetry is gone, but the
     EnKF still has no training routes, so that sentence stays -- just not as an exception.
NOTE
echo
