#!/bin/bash
#SBATCH --job-name=smoke_ckpt2
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/methods/varnet/runs/slurm_%x_%j.out
# Closes the gap left by smoke_ckptevery.sbatch: that run passed --steps, which ends the
# epoch loop after N steps, so only epoch 0 ever ran and two branches went untested --
# `epoch % ckpt_every == 0` at epoch > 0, and the `epoch == epochs-1` forced save that is
# the reason ckpt_00149.pt will exist at the end of a 150-epoch chain.
#
# No --steps here. --n-eval 32 instead of the default 1024 is what keeps it inside the
# 25-minute debug limit: the per-epoch cost is dominated by scoring the eval subset, not
# by the 6 training steps.
source /etc/profile 2>/dev/null
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project/methods/varnet
D=scratch/smoke_ckptevery2; rm -rf "$D"; mkdir -p "$D"

# epochs 0..4, snapshot every 2 -> expect ckpt_00000, ckpt_00002, ckpt_00004(also last epoch)
python3 -u -m methods.varnet.train --obs-every-k 1 --hidden 96 --kt 5 \
    --days 1 --dT 200 --epochs 5 --batch 32 --amp --n-eval 32 --ckpt-every 2 \
    --iter-schedule '0:5:1e-3' --clip-grad 1.0 --outdir "$D" || exit 1
echo "--- snapshots written:"; ls -1 "$D"/ckpt_*.pt
GOT=$(ls -1 "$D"/ckpt_*.pt | xargs -n1 basename | tr '\n' ' ')
WANT="ckpt_00000.pt ckpt_00002.pt ckpt_00004.pt "
if [ "$GOT" = "$WANT" ]; then echo "[smoke] PASS — got exactly: $GOT"
else echo "[smoke] FAIL — got '$GOT', wanted '$WANT'"; exit 1; fi

# epochs 0..3 with every=5: nothing divides, so ONLY the forced last-epoch save should exist
D2="$D/lastonly"
python3 -u -m methods.varnet.train --obs-every-k 1 --hidden 96 --kt 5 \
    --days 1 --dT 200 --epochs 4 --batch 32 --amp --n-eval 32 --ckpt-every 5 \
    --iter-schedule '0:5:1e-3' --clip-grad 1.0 --outdir "$D2" || exit 1
GOT2=$(ls -1 "$D2"/ckpt_*.pt | xargs -n1 basename | tr '\n' ' ')
if [ "$GOT2" = "ckpt_00000.pt ckpt_00003.pt " ]; then echo "[smoke] PASS — forced last-epoch save works: $GOT2"
else echo "[smoke] FAIL — got '$GOT2', wanted 'ckpt_00000.pt ckpt_00003.pt '"; exit 1; fi
echo "[smoke] ckpt-every fully covered"
