# sbatch/_env.sh — 所有作业脚本的唯一环境入口。用法: source <repo>/sbatch/_env.sh
#
# 重构前每个脚本自己 `cd <方法目录>` 然后靠 sys.path.insert 拼路径（全仓库 136 处），
# 而 losses.py / dataset.py / model.py 各有多份同名，谁先进 sys.path 谁赢 ——
# dincae/checks/evaluate.py 里那段解释 insert 与 append 顺序的注释就是这么来的。
#
# 现在只有一条规则：cwd 是仓库根，PYTHONPATH 是仓库根，脚本用 `python3 -m` 跑。
# 这样 sys.path[0] 永远是仓库根，方法自己的目录不会进搜索路径，同名模块不可能互相顶掉。

THESIS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export THESIS_ROOT
export PYTHONPATH="$THESIS_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$THESIS_ROOT"

module load scicomp-pytorch-env/2026.1

# 用 -m 跑，不要 `python3 methods/varnet/train.py`：后者会把 methods/varnet/ 塞到
# sys.path[0]，同名模块的问题就回来了。
#   python3 -m methods.varnet.train --days 32 ...
#   python3 -m compare.compare4
#   python3 -m methods.enkf.checks.verify_enkf_opt --frames 12
