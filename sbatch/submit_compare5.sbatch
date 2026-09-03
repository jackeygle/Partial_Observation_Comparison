#!/bin/bash
#SBATCH --job-name=compare5
#SBATCH --partition=gpu-h200-141g-short,gpu-a100-80g,gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=0-02:00:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/compare/results/slurm_%x_%j.out
# 五方逐通道对比，四套格子口径并列。
#
# 这个作业脚本放在顶层 sbatch/ 而不是某个方法的 sbatch/ 下：它评的是全部五个方法，
# 归到任何一方名下都会误导（compare4 曾经住在 senseiver_crowd/ 里，它的输出也就
# 落在 senseiver 的 check_outputs 里，看着像是 Senseiver 的指标）。
#
# 内存 128G：EnKF 的 est_*.npz 单日约 500MB，加上 12 个 solver 各自整天的
# (T,C,H,W) 中间量。
#
# 用法:
#   sbatch sbatch/submit_compare5.sbatch                        # 全量
#   sbatch sbatch/submit_compare5.sbatch --days 1 --ensembles "" # 冒烟
set -euo pipefail
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd "$THESIS_ROOT"
mkdir -p compare/results
python3 -u -m compare.compare5 "$@"
