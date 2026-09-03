"""
drop_syspath.py — Phase 2 的第二步：去掉不再需要的 sys.path 前言（一次性工具）

重构前每个脚本都要自己把项目根拼进 sys.path 才能 import 兄弟模块。现在它们是真包，
import 走 `crowdcore.*` / `methods.*.*`，这些前言不但没用，还会把方法自己的目录塞进
搜索路径 —— 那正是 losses.py 三份同名互相顶掉的机制。

**保留**参数里提到 enkf_lab / enkf_opt 的插入：那两个 vendor 目录内部有 `pedpred -> .`
自链接，靠把该目录放进 sys.path 让包内的 `from pedpred.X import Y` 解析。删掉就等于
改 vendor 副本的行为。

用法:
    python3 refactor_baseline/drop_syspath.py            # dry-run
    python3 refactor_baseline/drop_syspath.py --apply
"""
from __future__ import annotations

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {"__pycache__", ".git", "enkf_lab", "enkf_opt", "refactor_baseline",
             "runs", "check_outputs", "cache", "artifacts"}

RE_SYSPATH = re.compile(r"^\s*sys\.path\.(insert|append)\(")
RE_GUARD = re.compile(r"^\s*if\s+\w+\s+not\s+in\s+sys\.path\s*:\s*$")

# 白名单式删除：只删**认得出**是"把项目根拼进搜索路径"的那几种写法，
# 其他一切形式保留并报告，让人去看。
#
# 第一版用的是黑名单（行里提到 enkf_lab/enkf_opt 就保留），差点删掉
# methods/enkf/checks/verify_enkf_opt.py:54 的 `sys.path.insert(0, root)` ——
# 那里的 root 是个变量，行里没有 "enkf" 字样，而那个脚本正是重构的验收测试。
# 黑名单在这种任务上是错的方向：删错不报错，只会静默 import 到别的模块。
DROP_ARGS = (
    "os.path.dirname(os.path.dirname(os.path.abspath(__file__)))",
    "os.path.dirname(os.path.abspath(__file__))",
    "ROOT", "R", "_V4D", "V4D", "root_dir", "PROJ",
    # 上一步 rewrite_imports.py 把写死的旧路径换成了这个形式；包化之后同样不需要了
    "paths.method(paths.VARNET)",
    'os.path.join(ROOT, "checks")', "os.path.join(ROOT, 'checks')",
    'os.path.join(R, "checks")', "os.path.join(R, 'checks')",
    'os.path.join(V4D, "checks")', "os.path.join(V4D, 'checks')",
)


def _arg_of(line: str) -> str:
    """取 sys.path.insert(0, X) / sys.path.append(X) 里的 X，去掉空白与末尾注释。"""
    inner = line[line.index("(") + 1:line.rindex(")")]
    if inner.lstrip().startswith("0,"):
        inner = inner.lstrip()[2:]
    return inner.strip()


def _should_drop(line: str) -> bool:
    if "Thesis_Project" in line:        # 写死的旧绝对路径，一定要删
        return True
    try:
        return _arg_of(line) in DROP_ARGS
    except ValueError:
        return False


def process(path: str):
    lines = open(path).read().split("\n")
    out, dropped, kept, i = [], [], [], 0
    while i < len(lines):
        ln = lines[i]

        # `if X not in sys.path:` + 缩进的 append —— 两行一起处理
        if RE_GUARD.match(ln) and i + 1 < len(lines) and RE_SYSPATH.match(lines[i + 1]):
            if _should_drop(lines[i + 1]):
                dropped.append((ln + " ; " + lines[i + 1]).strip())
            else:
                kept.append((ln + " ; " + lines[i + 1]).strip())
                out.extend([ln, lines[i + 1]])
            i += 2
            continue

        if RE_SYSPATH.match(ln):
            if _should_drop(ln):
                dropped.append(ln.strip())
            else:
                kept.append(ln.strip())
                out.append(ln)
            i += 1
            continue

        out.append(ln)
        i += 1

    return "\n".join(out), dropped, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    n_files = n_drop = n_keep = 0
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dirpath, fn)
            new, dropped, kept = process(p)
            if not dropped and not kept:
                continue
            rel = os.path.relpath(p, ROOT)
            if dropped:
                n_files += 1
                n_drop += len(dropped)
                print(f"{rel}")
                for d in dropped:
                    print(f"  - {d}")
            for k in kept:
                n_keep += 1
                print(f"{rel}\n  = 保留 {k}")
            if args.apply and dropped:
                open(p, "w").write(new)

    print(f"\n删除 {n_drop} 处（{n_files} 个文件），保留 {n_keep} 处 enkf vendor 相关"
          + ("（已写入）" if args.apply else "（dry-run）"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
