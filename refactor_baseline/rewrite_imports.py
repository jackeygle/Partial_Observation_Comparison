"""
rewrite_imports.py — Phase 2 的机械改写（一次性工具，重构完成后可删）

把重构前那套 "sys.path.insert 之后裸 import" 改成包路径 import，并去掉写死的旧目录。

为什么要脚本而不是手改：涉及 90 个 .py 文件、约 200 条 import 语句，而这类改动
**改错了不会报错** —— Python 会安静地 import 到另一个同名模块（losses.py 有三份），
算出别的数字。脚本可以先 dry-run 看全量 diff，也保证同一条规则处处一致。

用法:
    python3 refactor_baseline/rewrite_imports.py            # dry-run，只报告
    python3 refactor_baseline/rewrite_imports.py --apply    # 真改
"""
from __future__ import annotations

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 模块名 -> 它现在住在哪个包。crowdcore 的三个是全局唯一的；其余按文件所在方法解析。
GLOBAL = {
    "config": "crowdcore.config",
    "navigation": "crowdcore.navigation",
    "observation_model": "crowdcore.observation_model",
    "paths": "crowdcore.paths",
    "prior_model": "methods.varnet.prior_model",
    "variational_solver": "methods.varnet.variational_solver",
    "model_io": "methods.varnet.checks.model_io",
}
# 这些名字在多个方法里都有，必须按上下文解析
PER_METHOD = {"losses", "dataset", "model", "train",
              "state", "encoding",                      # dincae
              "network", "sensors", "positional"}       # senseiver

# compare/ 下的脚本是从哪个方法搬来的 —— 决定它的 dataset/losses 指向谁
COMPARE_ORIGIN = {
    "compare3.py": "senseiver", "compare4.py": "senseiver", "plot_compare3.py": "senseiver",
    "eval_threeway_accuracy.py": "varnet", "compare_channels.py": "varnet",
    "plot_comparison.py": "varnet", "plot_reconstruction_enkf.py": "varnet",
    "bench_speed.py": "varnet", "plot_speed.py": "varnet", "plot_frameworks.py": "varnet",
}

OLD_ABS = "/scratch/work/zhangx29/Thesis_Project/4dvarnet_enkf"


def owning_method(path: str) -> str | None:
    rel = os.path.relpath(path, ROOT)
    parts = rel.split(os.sep)
    if parts[0] == "methods":
        return parts[1]
    if parts[0] == "compare":
        return COMPARE_ORIGIN.get(parts[-1])
    return None


def target(mod: str, path: str) -> str | None:
    if mod in GLOBAL:
        return GLOBAL[mod]
    if mod in PER_METHOD:
        m = owning_method(path)
        return f"methods.{m}.{mod}" if m else None
    return None


TRIPLE = ('"' * 3, "'" * 3)


def _mask_strings(text: str):
    """把三引号字符串的内容挖空（保住行号），让正则只能命中真正的代码行。

    需要这一步的原因：crowdcore/config.py 的文档字符串里有一行 `import config`
    （Usage 示例），第一版脚本把它当成真 import 改掉了。文档里的示例代码和真代码
    长得一模一样，只能靠位置区分。
    """
    out, i, n = [], 0, len(text)
    while i < n:
        hits = [k for k in (text.find(TRIPLE[0], i), text.find(TRIPLE[1], i)) if k != -1]
        if not hits:
            out.append(text[i:])
            break
        j = min(hits)
        q = text[j:j + 3]
        end = text.find(q, j + 3)
        if end == -1:
            out.append(text[i:])
            break
        out.append(text[i:j + 3])
        out.append("".join("\n" if c == "\n" else "\0" for c in text[j + 3:end]))
        out.append(q)
        i = end + 3
    return "".join(out)


#: 这两个文件的文档里**故意**提到旧路径（解释重构历史），不能改
DOC_ONLY = {"crowdcore/__init__.py", "crowdcore/paths.py"}


def rewrite(text: str, path: str):
    notes = []
    masked = _mask_strings(text)

    def fix_import(m):
        """`import X as Y` / `import X`  ->  `from pkg import X as Y`"""
        indent, mod, alias, tail = m.group(1), m.group(2), m.group(3), m.group(4)
        t = target(mod, path)
        if t is None:
            return m.group(0)
        pkg, leaf = t.rsplit(".", 1)
        notes.append(f"import {mod}" + (f" as {alias}" if alias else ""))
        as_part = f" as {alias}" if alias else f" as {mod}" if leaf != mod else ""
        return f"{indent}from {pkg} import {leaf}{as_part}{tail}"

    def fix_from(m):
        """`from X import a, b`  ->  `from pkg.X import a, b`"""
        indent, mod, names, tail = m.group(1), m.group(2), m.group(3), m.group(4)
        t = target(mod, path)
        if t is None:
            return m.group(0)
        notes.append(f"from {mod} import {names.strip()}")
        return f"{indent}from {t} import {names}{tail}"

    mods = "|".join(sorted(set(GLOBAL) | PER_METHOD))
    re_imp = re.compile(
        rf"^([ \t]*)import ({mods})(?: as ([A-Za-z_][A-Za-z_0-9]*))?([ \t]*(?:#.*)?)$")
    re_frm = re.compile(rf"^([ \t]*)from ({mods}) import ([^\n#]+?)([ \t]*(?:#.*)?)$")

    # 逐行处理：只有在 masked（文档字符串已挖空）里也匹配的行才是真代码
    lines, mlines = text.split("\n"), masked.split("\n")
    for i, (ln, ml) in enumerate(zip(lines, mlines)):
        if re_imp.match(ml):
            lines[i] = re_imp.sub(fix_import, ln)
        elif re_frm.match(ml):
            lines[i] = re_frm.sub(fix_from, ln)
    text = "\n".join(lines)

    # 写死的旧目录 -> paths（文档里故意提到它的文件跳过）
    if OLD_ABS in text and os.path.relpath(path, ROOT) not in DOC_ONLY:
        n = text.count(OLD_ABS)
        text = text.replace(f'"{OLD_ABS}"', "paths.method(paths.VARNET)")
        text = text.replace(f"'{OLD_ABS}'", "paths.method(paths.VARNET)")
        text = text.replace(OLD_ABS, "<<PATHS_VARNET>>")   # 剩下的是拼在长串里的，人工看
        notes.append(f"旧绝对路径 x{n}")

    return text, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    changed, total_notes, manual = 0, 0, []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames
                       if d not in {"__pycache__", ".git", "enkf_lab", "enkf_opt",
                                    "refactor_baseline", "runs", "check_outputs",
                                    "cache", "artifacts", "scratch"}]
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dirpath, fn)
            src = open(p).read()
            new, notes = rewrite(src, p)
            if new == src:
                continue
            changed += 1
            total_notes += len(notes)
            rel = os.path.relpath(p, ROOT)
            print(f"{rel}")
            for n in notes:
                print(f"    {n}")
            if "<<PATHS_VARNET>>" in new:
                manual.append(rel)
            if args.apply:
                open(p, "w").write(new)

    print(f"\n{changed} 个文件，{total_notes} 处改写"
          + ("（已写入）" if args.apply else "（dry-run，未写入）"))
    if manual:
        print(f"\n以下文件里旧路径拼在长串中，占位成 <<PATHS_VARNET>>，需人工处理:")
        for r in manual:
            print(f"    {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
