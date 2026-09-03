"""
rewrite_sbatch.py — Phase 3：把 34 个 sbatch 脚本改成新的调用方式（一次性工具）

改三件事：

  1. `module load ...`  ->  `source <root>/sbatch/_env.sh`
     _env.sh 负责 module load、PYTHONPATH、PYTHONSAFEPATH，只有一处定义。
     **`cd <方法目录>` 保留**（路径更新到新位置）：脚本里 118 处相对路径都相对于
     方法目录，保留 cd 就一处都不用改。PYTHONSAFEPATH=1 已经把 cwd 挡在 sys.path
     外，所以待在方法目录是安全的。

  2. `python3 -u <脚本路径>.py`  ->  `python3 -u -m <包路径>`
     必须用 -m。`python3 methods/varnet/train.py` 会把 methods/varnet/ 塞到
     sys.path[0]，三份同名 losses.py 互相顶掉的问题就回来了。

  3. `#SBATCH --output=` 里的旧目录 -> 新目录。

脚本里的相对路径（--outdir runs/varnet_b0_k1、sbatch/submit_x_chain.sbatch 的自我重投
之类）一处都不用改，因为 cwd 仍然是方法目录。这是刻意的：第一版让 _env.sh 把 cwd 挪到
仓库根，那会让 118 处相对路径全部指错 —— 纯风险没有收益。

用法:
    python3 refactor_baseline/rewrite_sbatch.py            # dry-run
    python3 refactor_baseline/rewrite_sbatch.py --apply
"""
from __future__ import annotations

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD = "/scratch/work/zhangx29/Thesis_Project"

#: 旧方法目录 -> 新方法目录
DIRMAP = {
    "4dvarnet_enkf": "methods/varnet",
    "dincae_crowd": "methods/dincae",
    "senseiver_crowd": "methods/senseiver",
}
#: 旧脚本相对路径 -> 新包路径。checks/ 里跨方法的那些已经搬到 compare/。
MOVED_TO_COMPARE = {"eval_threeway_accuracy", "compare_channels", "plot_comparison",
                    "plot_reconstruction_enkf", "bench_speed", "plot_speed",
                    "plot_frameworks", "compare3", "compare4", "plot_compare3"}
MOVED_TO_ENKF = {"bench_enkf_opt", "bench_enkf_split", "diag_enkf_spread_growth",
                 "diag_sparsification_enkf", "eval_uncertainty_enkf", "export_obs_for_enkf",
                 "run_enkf_baseline", "score_enkf", "verify_enkf_gain_mode",
                 "verify_enkf_opt", "plot_velocity_enkf"}
#: 顶层脚本改名
RENAMED = {"train_varnet": "train"}


def module_for(old_dir: str, script_rel: str) -> str:
    """('4dvarnet_enkf', 'checks/eval_threeway_accuracy.py') -> 'compare.eval_threeway_accuracy'"""
    parts = script_rel[:-3].split("/")            # 去掉 .py
    stem = parts[-1]
    if stem in MOVED_TO_COMPARE:
        return f"compare.{stem}"
    if stem in MOVED_TO_ENKF:
        return f"methods.enkf.checks.{stem}"
    newdir = DIRMAP[old_dir].replace("/", ".")
    stem = RENAMED.get(stem, stem)
    parts = parts[:-1] + [stem]
    return f"{newdir}." + ".".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    n, manual = 0, []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__"}]
        for fn in sorted(filenames):
            if not fn.endswith(".sbatch"):
                continue
            p = os.path.join(dirpath, fn)
            s = open(p).read()
            orig = s
            rel = os.path.relpath(p, ROOT)

            # 找出这个脚本原来属于哪个方法
            owner = next((d for d in DIRMAP if f"{OLD}/{d}" in s), None)

            # 1) 旧目录 -> 新目录。两种形态都要管：带尾斜杠的（--output=.../runs/...）
            # 和行尾没有斜杠的（`cd .../4dvarnet_enkf`）。第一版只替了前者，于是
            # cd 行一个都没改到 —— 而那正是最要紧的一行。
            for old_d, new_d in DIRMAP.items():
                s = s.replace(f"{OLD}/{old_d}/", f"{OLD}/{new_d}/")
                s = re.sub(rf"{re.escape(OLD)}/{re.escape(old_d)}(?=$|[\s\"'])",
                           f"{OLD}/{new_d}", s, flags=re.M)

            # 2) python3 [-u] <path>.py  ->  python3 -u -m <pkg>
            def fix_run(m):
                pre, script, post = m.group(1), m.group(2), m.group(3)
                if owner is None:
                    return m.group(0)
                return f"{pre}-m {module_for(owner, script)}{post}"

            s = re.sub(r"(python3 (?:-u )?)([\w/]+\.py)(.*)$", fix_run, s, flags=re.M)

            # 3) `module load ...` -> `source _env.sh`，**cd 保留**
            #
            # _env.sh 刻意不 cd：各脚本里有 118 处相对路径（runs/、check_outputs/、
            # sbatch/ 的自我重投）全都相对于方法目录。保留 cd 就一处都不用改；
            # PYTHONSAFEPATH=1 已经把 cwd 挡在 sys.path 外，所以留在方法目录是安全的。
            if owner:
                s = re.sub(r"^module load [^\n]*\n",
                           f"source {OLD}/sbatch/_env.sh\n", s, count=1, flags=re.M)

            if s != orig:
                n += 1
                print(f"{rel}")
                for a, b in zip(orig.split("\n"), s.split("\n")):
                    if a != b:
                        print(f"  - {a}\n  + {b}")
                if args.apply:
                    open(p, "w").write(s)

            # cwd 仍是方法目录，相对路径参数不需要动 —— 只在没有 cd 的脚本里才要看
            if owner and f"cd {OLD}/{DIRMAP[owner]}" not in s:
                manual.append((rel, "没有 cd 到方法目录，相对路径要确认"))

    print(f"\n{n} 个 sbatch 改写" + ("（已写入）" if args.apply else "（dry-run）"))
    if manual:
        print(f"\ncwd 从方法目录变成了仓库根，以下相对路径参数需人工确认（{len(manual)} 处）:")
        for rel, arg in manual:
            print(f"    {rel}: {arg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
