#!/bin/bash
#SBATCH --job-name=dincae
#SBATCH --partition=gpu-h200-141g-short,gpu-a100-80g,gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:USR1@1200
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/methods/dincae/runs/slurm_%x_%j.out
#
# 自链接：跑完 --epochs 就正常退出；若快到墙钟(提前 1200 s 收到 USR1)，先排一个
# 后继作业并 --resume 接着跑（train.py 每个 epoch 都存 last.pt，含 optimizer 状态）。
# 用法: sbatch sbatch/submit_train.sbatch [额外的 train.py 参数]
#   环境变量: EPOCHS(默认 200)  OUT(默认 runs/dincae_full)

set -uo pipefail
DIR=/scratch/work/zhangx29/Thesis_Project/methods/dincae
cd "$DIR"
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh

EPOCHS=${EPOCHS:-200}
OUT=${OUT:-$DIR/runs/dincae_full}
RESUME=""
[ -f "$OUT/last.pt" ] && RESUME="--resume"

chain() { EPOCHS="$EPOCHS" OUT="$OUT" sbatch "$0" "$@" || true; }
trap 'echo "[chain] 接近墙钟，排后继作业"; chain "$@"' USR1

python3 -u -m methods.dincae.train --amp --epochs "$EPOCHS" --out "$OUT" $RESUME "$@" &
wait $!
