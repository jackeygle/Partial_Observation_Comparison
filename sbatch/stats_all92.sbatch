#!/bin/bash
#SBATCH --job-name=stats_all92
#SBATCH --partition=batch-skl,batch-csl,batch-milan
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/work/zhangx29/data_all92/logs/stats_all92_%j.log
# ROUTE 2, stage 2: everything fitted on the training days, rebuilt on the 63 all-92 training
# days -- the walkable mask (visited union, cached in data_all92/grid_cache) and DINCAE's
# normalisation statistics (methods/dincae/artifacts/state_stats_none_all92.npz) -- and a
# comparison with the Sunday-only versions. Senseiver and PedPred3 compute theirs at training.
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
export DINCAE_VAR_TRANSFORM=none
DATA_VARIANT=all92 python3 -u - <<'PY'
from crowdcore import config, observation_model as om
for s in ("train", "valid", "test"):
    f = om.split_files(s); print(f"[split] {s}: {len(f)} days")
import h5py
for s in ("train", "valid", "test"):
    for fp in om.split_files(s):
        with h5py.File(fp, "r") as h:
            n = h["grid"].shape[0]
        if n < 30000: print(f"[warn] short day {fp}: {n} frames")
PY
DATA_VARIANT=all92 python3 -u -c "
from crowdcore import navigation as nav; import numpy as np
m = nav.build_valid_mask_from_config().astype(bool); np.save('/scratch/work/zhangx29/data_all92/walkable_all92.npy', m)
print('[mask] all92 walkable cells', int(m.sum()))"
python3 -u -c "
from crowdcore import navigation as nav; import numpy as np
s = nav.build_valid_mask_from_config().astype(bool); a = np.load('/scratch/work/zhangx29/data_all92/walkable_all92.npy')
print('[mask] sunday', int(s.sum()), 'all92', int(a.sum()), 'only_sunday', int((s & ~a).sum()), 'only_all92', int((a & ~s).sum()))"
DATA_VARIANT=all92 python3 -u -m methods.dincae.state
ls -la methods/dincae/artifacts/
echo "[done]"
