#!/bin/bash
#SBATCH --job-name=smoke_ckptevery
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/methods/varnet/runs/slurm_%x_%j.out
# Verify the --ckpt-every snapshots land and that --resume still works, BEFORE the 15
# retrain chains start. A bug here would crash-loop all of them.
source /etc/profile 2>/dev/null
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project/methods/varnet
D=scratch/smoke_ckptevery; rm -rf "$D"; mkdir -p "$D"

echo "### segment 1: epochs 0-3, snapshot every 2"
python3 -u -m methods.varnet.train --resume --obs-every-k 1 --hidden 96 --kt 5 \
    --days 2 --dT 200 --epochs 4 --batch 32 --amp --steps 6 --ckpt-every 2 \
    --iter-schedule '0:5:1e-3,2:10:7e-4' --clip-grad 1.0 --outdir "$D" || exit 1
echo "--- files after segment 1:"; ls -1 "$D"

echo "### segment 2: --resume to epoch 6 (proves resume still finds varnet_last.pt)"
python3 -u -m methods.varnet.train --resume --obs-every-k 1 --hidden 96 --kt 5 \
    --days 2 --dT 200 --epochs 6 --batch 32 --amp --steps 6 --ckpt-every 2 \
    --iter-schedule '0:5:1e-3,2:10:7e-4' --clip-grad 1.0 --outdir "$D" || exit 1
echo "--- files after segment 2:"; ls -1 "$D"

echo "### segment 3: --ckpt-every 0 must not divide by zero"
python3 -u -m methods.varnet.train --obs-every-k 1 --hidden 96 --kt 5 \
    --days 2 --dT 200 --epochs 1 --batch 32 --amp --steps 4 --ckpt-every 0 \
    --iter-schedule '0:5:1e-3' --outdir "$D/zero" || exit 1
echo "--- files (must have NO ckpt_*.pt):"; ls -1 "$D/zero"
echo "[smoke] ckpt-every OK"
