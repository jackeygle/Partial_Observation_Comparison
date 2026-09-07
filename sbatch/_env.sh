# sbatch/_env.sh — the single environment entry point for every job script
#
#   source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
#   cd "$THESIS_ROOT/methods/varnet"          # each script cd's into its own method dir
#   python3 -u -m methods.varnet.train "$@"
#
# Before the refactor, every .py stitched the project root onto sys.path itself
# (136 places across the repo), and losses.py / dataset.py / model.py each had
# several same-named copies -- whichever hit sys.path first won. That is where the
# comment in dincae/checks/evaluate.py explaining insert-vs-append order came from.
#
# Now two things are kept separate:
#
#   PYTHONPATH=<repo root>   imports go only through package paths
#                            (crowdcore.*, methods.<method>.*)
#   PYTHONSAFEPATH=1         cwd does not get added to sys.path
#
# The second is the important one. With `python3 -m`, Python by default inserts cwd
# at sys.path[0], and a job's cwd is its method directory -- which puts
# methods/varnet/ back on the search path, bringing back the problem of same-named
# modules shadowing each other. PYTHONSAFEPATH (Python 3.11+, 3.12 here) turns off
# that automatic insertion. Verified: with cwd off sys.path, the three copies of
# losses.py stay independent, and a bare `import losses` fails immediately with
# ModuleNotFoundError.
#
# **This script deliberately does not cd.** Each method's scripts have 118 relative
# paths (runs/, check_outputs/, sbatch/'s self-resubmission), all relative to the
# method directory. Letting the caller cd itself means none of those paths need to
# change; forcing cwd to the repo root here would mean rewriting every one of them,
# for pure risk and no benefit.

THESIS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export THESIS_ROOT
export PYTHONPATH="$THESIS_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1

module load scicomp-pytorch-env/2026.1
