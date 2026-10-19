#!/bin/bash
#SBATCH --job-name=regrid_wed
#SBATCH --partition=batch-skl,batch-csl,batch-milan
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --array=1-46
#SBATCH --output=/scratch/work/zhangx29/data_all92/logs/regrid_wed_%A_%a.log
# Route 2, stage 1: one Wednesday per array task, raw ATC CSV -> trajectory H5
# (data_all92/ATC/Wednesdays/) -> grid_cache (data_all92/grid_cache/), with the same two
# scripts and arguments that regenerated the Sunday 2012-10-28 identically (regrid_check).
# Writes only under data_all92/; an existing output is never overwritten.
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
R=/scratch/work/zhangx29/data_all92
DAY=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $R/wednesdays.txt)
DAY=${DAY#atc-}; DAY=${DAY%.h5}; DAY=${DAY%.csv}
TRAJ=$R/ATC/Wednesdays/atc-$DAY.h5
GRID=$R/grid_cache/atc-${DAY}_corridor_1.0s.h5
echo "[day] $DAY"
if [ ! -e "$TRAJ" ]; then
    python3 -u crowdcore/data/csv_to_h5.py --csv /scratch/work/zhangx29/ATC/atc-$DAY.csv --out $TRAJ.part
    mv $TRAJ.part $TRAJ
fi
if [ ! -e "$GRID" ]; then
    python3 -u crowdcore/data/h5_to_grid.py --traj-h5 $TRAJ --out $GRID.part
    mv $GRID.part $GRID
fi
python3 -c "import h5py; f=h5py.File('$GRID','r'); g=f['grid']; t=f['time'][:]; print('[ok] $DAY', g.shape, g.dtype, float(t[0]), float(t[-1]))"
