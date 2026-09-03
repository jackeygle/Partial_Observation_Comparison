# sbatch/_env.sh — 所有作业脚本的唯一环境入口
#
#   source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
#   cd "$THESIS_ROOT/methods/varnet"          # 各脚本自己 cd 到它的方法目录
#   python3 -u -m methods.varnet.train "$@"
#
# 重构前每个 .py 自己把项目根拼进 sys.path（全仓库 136 处），而 losses.py / dataset.py /
# model.py 各有多份同名，谁先进 sys.path 谁赢 —— dincae/checks/evaluate.py 里那段解释
# insert 与 append 顺序的注释就是这么来的。
#
# 现在两件事分开：
#
#   PYTHONPATH=仓库根     import 只走包路径（crowdcore.*、methods.<方法>.*）
#   PYTHONSAFEPATH=1      cwd 不进 sys.path
#
# 第二条是关键。用 `python3 -m` 时 Python 默认把 cwd 插到 sys.path[0]，而作业的 cwd 是
# 方法目录 —— 那等于把 methods/varnet/ 放回搜索路径，同名模块互相顶掉的问题就回来了。
# PYTHONSAFEPATH（Python 3.11+，这里是 3.12）关掉那个自动插入。实测：cwd 不在 sys.path
# 里，三份 losses 各自独立，裸 `import losses` 直接 ModuleNotFoundError。
#
# **本脚本刻意不 cd。** 各方法的脚本里有 118 处相对路径（runs/、check_outputs/、
# sbatch/ 的自我重投），全都相对于方法目录。让调用方自己 cd，那些路径一处都不用改；
# 强行把 cwd 挪到仓库根就要重写它们，而那是纯粹的风险没有收益。

THESIS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export THESIS_ROOT
export PYTHONPATH="$THESIS_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1

module load scicomp-pytorch-env/2026.1
