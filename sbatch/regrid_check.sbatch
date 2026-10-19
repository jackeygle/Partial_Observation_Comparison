#!/bin/bash
#SBATCH --job-name=regrid_check
#SBATCH --partition=batch-skl,batch-csl,batch-milan
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/work/zhangx29/data_all92/dev_check/regrid_check_%j.log
# Route 2, stage 1 check: regenerate one Sunday from its raw CSV with crowdcore/data and compare
# with the trajectory H5 and grid_cache file the Sunday pipeline uses. Writes only to dev_check/.
source "${SLURM_SUBMIT_DIR}/sbatch/_env.sh"
cd "$THESIS_ROOT"
D=/scratch/work/zhangx29/data_all92/dev_check; DAY=20121028
/usr/bin/time -v python3 -u crowdcore/data/csv_to_h5.py --csv /scratch/work/zhangx29/ATC/atc-$DAY.csv --out $D/atc-$DAY.h5 2>&1 | grep -E "wrote|Elapsed|Maximum resident|Error|Traceback" 
/usr/bin/time -v python3 -u crowdcore/data/h5_to_grid.py --traj-h5 $D/atc-$DAY.h5 --out $D/atc-${DAY}_corridor_1.0s.h5 2>&1 | grep -E "wrote|Elapsed|Maximum resident|Error|Traceback"
python3 -u -m supervisor_evaluation.checks.compare_regridded --new-traj $D/atc-$DAY.h5 --old-traj /scratch/work/zhangx29/data/ATC/Sundays/atc-$DAY.h5 \
    --new-grid $D/atc-${DAY}_corridor_1.0s.h5 --old-grid /scratch/work/zhangx29/data/grid_cache/atc-${DAY}_corridor_1.0s.h5
