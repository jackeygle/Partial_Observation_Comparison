"""
import_all.py — Phase 3 的第一道验收：所有 import 目标都解析得到吗

重构改了 186 处 import、95 处 sys.path、以及所有写死的旧路径。这类改动**改错了不会在
语法检查里报错**，只会在真正 import 时炸，或者更糟 —— 静默 import 到另一个同名模块。

两层检查，因为这两件事必须分开：

  Tier A（静态，不执行任何代码）
      用 ast 解析每个 .py，把它的每一条 import 目标喂给 importlib.util.find_spec。
      覆盖全部文件，包括那些顶层就跑 main 的画图/评测脚本。

  Tier B（真 import）
      只对**无副作用的库模块**做真正的 import：crowdcore 的四个，以及各方法的
      model/losses/dataset/... 这些。顺带确认三份同名 losses.py 各自独立 ——
      那是整场重构的主要目的。

第一版直接 importlib.import_module 了所有 99 个模块，结果把各评测脚本的顶层代码全跑了
一遍（很多脚本没有 if __name__ 保护），14 分钟后 32G 内存被撑爆。"能不能 import"和
"跑一遍"是两件事，这里只验前者。

用法（登录节点即可，但要先 source sbatch/_env.sh 拿到 torch）:
    python3 -m refactor_baseline.import_all
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {"__pycache__", ".git", "enkf_lab", "enkf_opt",
             "runs", "check_outputs", "cache", "artifacts"}

#: Tier B：这些模块顶层只有定义，import 它们不会跑任何计算
SAFE_LEAVES = {"config", "navigation", "observation_model", "paths",
               "model", "losses", "dataset", "network", "positional", "sensors",
               "state", "encoding", "prior_model", "variational_solver"}

#: 这些顶层名字来自 vendor 副本，靠 sys.path 在运行时注入，静态解析必然找不到
VENDOR_ROOTS = {"pedpred"}


def py_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in sorted(filenames):
            if fn.endswith(".py"):
                yield os.path.join(dirpath, fn)


def import_targets(path: str):
    """(模块名, 行号) —— 该文件 import 的每一个顶层模块。相对 import 跳过。"""
    try:
        tree = ast.parse(open(path).read(), filename=path)
    except SyntaxError as e:
        yield (f"<语法错误: {e}>", getattr(e, "lineno", 0))
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield (a.name, node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield (node.module, node.lineno)


def tier_a():
    print("[Tier A] 静态解析每一条 import 目标（不执行代码）\n")
    bad = 0
    for p in py_files():
        rel = os.path.relpath(p, ROOT)
        fails = []
        for mod, lineno in import_targets(p):
            if mod.startswith("<"):
                fails.append((lineno, mod))
                continue
            top = mod.split(".")[0]
            if top in VENDOR_ROOTS:
                continue
            try:
                if importlib.util.find_spec(mod) is None:
                    fails.append((lineno, f"找不到 {mod}"))
            except (ImportError, ModuleNotFoundError, ValueError) as e:
                fails.append((lineno, f"{mod}: {type(e).__name__} {e}"))
        if fails:
            bad += len(fails)
            print(f"  {rel}")
            for lineno, msg in fails:
                print(f"      :{lineno}  {msg}")
    print(f"\n  {'全部 import 目标都解析得到' if not bad else f'{bad} 处解析不到'}")
    return bad


def tier_b():
    print("\n[Tier B] 真正 import 无副作用的库模块\n")
    mods = []
    for p in py_files():
        rel = os.path.relpath(p, ROOT)
        parts = rel[:-3].split(os.sep)
        if parts[-1] == "__init__" or os.sep not in rel:
            continue
        if parts[-1] in SAFE_LEAVES and parts[0] in ("crowdcore", "methods"):
            mods.append(".".join(parts))

    bad = 0
    for m in sorted(mods):
        try:
            importlib.import_module(m)
        except BaseException:
            bad += 1
            print(f"  {m}\n      {traceback.format_exc(limit=2).strip().splitlines()[-1]}")
    print(f"  {len(mods) - bad}/{len(mods)} 个库模块 import 成功")

    print("\n[同名模块隔离] 重构的主要目的")
    import methods.dincae.losses as ld
    import methods.senseiver.losses as ls
    import methods.varnet.losses as lv
    for tag, m in (("varnet", lv), ("dincae", ld), ("senseiver", ls)):
        print(f"  {tag:<10} {os.path.relpath(m.__file__, ROOT)}")
    assert len({lv.__file__, ld.__file__, ls.__file__}) == 3, "三份 losses 指向同一个文件！"
    print("  三份同名 losses 各自独立")
    return bad


def main():
    a = tier_a()
    b = tier_b()
    print(f"\n{'[通过]' if not (a or b) else '[失败]'} Tier A {a} 处问题，Tier B {b} 个模块失败")
    return 1 if (a or b) else 0


if __name__ == "__main__":
    sys.exit(main())
