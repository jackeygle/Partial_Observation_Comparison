#!/bin/bash
#SBATCH --job-name=sparsenkf
#SBATCH --partition=batch-csl,batch-hsw
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=check_outputs/sparsenkf_%j.out
# Pure numpy on the EnKF's own exported arrays -- no model inference, so no GPU.
module load scicomp-python-env
cd /scratch/work/zhangx29/Thesis_Project/4dvarnet_enkf
srun python3 checks/diag_sparsification_enkf.py
