"""
check_against_baseline.py — 重构的验收测试

重构只允许改变代码在哪里，不允许改变代码算出什么。这个脚本把新产出的 json 和
`refactor_baseline/` 里冻结的那份逐个数值叶子比对。

默认要求**完全相等**（不是 np.isclose）。理由：重构不涉及任何浮点运算顺序的改变 ——
搬文件和改 import 不会动到求和次序，所以任何非零差异都说明有东西被改坏了，而不是舍入。
真的遇到不可避免的差异时用 --tol 明确放宽，并且把放宽的理由写进提交信息。

用法（登录节点即可，纯 stdlib）:
    python3 refactor_baseline/check_against_baseline.py senseiver_crowd/check_outputs/eval/compare4.json
    python3 refactor_baseline/check_against_baseline.py new.json --baseline refactor_baseline/compare4.json
    python3 refactor_baseline/check_against_baseline.py new.json --tol 1e-12
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def leaves(obj, path=""):
    """把嵌套结构摊平成 {路径: 标量}。列表用下标入路径，这样错位也能被发现。"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new", help="重构后新产出的 json")
    ap.add_argument("--baseline", default=os.path.join(HERE, "compare4.json"))
    ap.add_argument("--tol", type=float, default=0.0,
                    help="允许的绝对差；默认 0 = 必须完全相等")
    args = ap.parse_args()

    for p in (args.new, args.baseline):
        if not os.path.exists(p):
            sys.exit(f"[fail] 找不到 {p}")

    with open(args.baseline) as f:
        base = dict(leaves(json.load(f)))
    with open(args.new) as f:
        new = dict(leaves(json.load(f)))

    # 结构差异先报，再比数值 —— 少了一个字段和数值变了是两种不同的错
    missing = sorted(set(base) - set(new))
    added = sorted(set(new) - set(base))
    bad, strings, checked, skipped = [], [], 0, 0

    for k in sorted(set(base) & set(new)):
        a, b = base[k], new[k]
        if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) \
                or not isinstance(b, (int, float)):
            # 非数值字段单独一类：重构会合法地改变记录路径的字符串（source、
            # enkf-dir、protocol 里的目录名），但**不允许**改变任何数值。
            # 把两者混在一个判定里，就没法区分"路径搬了"和"算错了"。
            if a != b:
                strings.append((k, a, b))
            else:
                skipped += 1
            continue
        checked += 1
        if math.isnan(a) and math.isnan(b):
            continue
        d = abs(float(a) - float(b))
        if d > args.tol:
            bad.append((k, a, b, d))

    print(f"基线 : {args.baseline}")
    print(f"新产出: {args.new}")
    print(f"比对了 {checked} 个数值、{skipped} 个非数值字段，容差 {args.tol:g}\n")

    if missing:
        print(f"[结构] 新产出里少了 {len(missing)} 个字段，例如:")
        for k in missing[:8]:
            print(f"    {k}")
    if added:
        print(f"[结构] 新产出里多了 {len(added)} 个字段（新增口径是正常的），例如:")
        for k in added[:8]:
            print(f"    {k}")
    if missing or added:
        print()

    if strings:
        print(f"[字符串] {len(strings)} 处不同（路径类字段在重构里合法变化，不判失败）:")
        for k, a, b in strings[:10]:
            print(f"    {k}\n        基线 {a!r}\n        新   {b!r}")
        print()

    if bad:
        print(f"[数值] {len(bad)} 处不一致:")
        for k, a, b, d in bad[:25]:
            tail = f"  差 {d:.3e}" if d is not None else ""
            print(f"    {k}\n        基线 {a!r}\n        新   {b!r}{tail}")
        if len(bad) > 25:
            print(f"    ... 另有 {len(bad) - 25} 处")

    ok = not bad and not missing
    if ok:
        extra = f"（{len(strings)} 处路径字符串变化，见上）" if strings else ""
        print(f"\n[通过] {checked} 个数值全部在容差内，重构没有改变结果。{extra}")
    else:
        print("\n[失败] 重构改变了结果，先修再继续。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
