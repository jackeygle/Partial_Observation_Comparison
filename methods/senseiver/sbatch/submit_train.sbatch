#!/bin/bash
#SBATCH --job-name=senseiver
#SBATCH --partition=gpu-h200-141g-short,gpu-a100-80g,gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:USR1@1200
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/senseiver_crowd/runs/slurm_%x_%j.out
#
# 自链接：跑满 --epochs 就正常退出；快到墙钟时(提前 1200s 收到 USR1)先排一个后继
# 作业并 --resume 接着跑(train.py 每个 epoch 存 last.pt，含 optimizer 状态)。
# 用法: sbatch sbatch/submit_train.sbatch [额外的 train.py 参数]
#   环境变量: EPOCHS(默认 100)  OUT(默认 runs/senseiver_A)
set -uo pipefail
DIR=/scratch/work/zhangx29/Thesis_Project/senseiver_crowd
cd "$DIR"
module load scicomp-pytorch-env/2026.1

EPOCHS=${EPOCHS:-100}
OUT=${OUT:-$DIR/runs/senseiver_A}
RESUME=""
[ -f "$OUT/last.pt" ] && RESUME="--resume"

chain() { EPOCHS="$EPOCHS" OUT="$OUT" sbatch "$0" "$@" || true; }
trap 'echo "[chain] 接近墙钟，排后继作业"; chain "$@"' USR1

python3 -u train.py --amp --epochs "$EPOCHS" --out "$OUT" $RESUME "$@" &
wait $!
