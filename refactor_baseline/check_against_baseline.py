"""
check_against_baseline.py — the refactor's acceptance test

A refactor is only allowed to change where the code lives, never what the code
computes. This script compares a newly produced json against the frozen one in
`refactor_baseline/`, leaf value by leaf value.

Requires **exact equality** by default (not np.isclose). Reason: the refactor
does not touch any floating-point operation order -- moving files and fixing
imports does not change summation order, so any nonzero difference means
something got broken, not rounding. When an unavoidable difference genuinely
occurs, relax it explicitly with --tol, and write the reason for relaxing it
into the commit message.

Usage (the login node is fine, pure stdlib):
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
    """Flattens a nested structure into {path: scalar}. List indices go into
    the path, so a misalignment can be found too."""
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
    ap.add_argument("new", help="the newly produced json after the refactor")
    ap.add_argument("--baseline", default=os.path.join(HERE, "compare4.json"))
    ap.add_argument("--tol", type=float, default=0.0,
                    help="allowed absolute difference; default 0 = must be exactly equal")
    args = ap.parse_args()

    for p in (args.new, args.baseline):
        if not os.path.exists(p):
            sys.exit(f"[fail] {p} not found")

    with open(args.baseline) as f:
        base = dict(leaves(json.load(f)))
    with open(args.new) as f:
        new = dict(leaves(json.load(f)))

    # Report structural differences first, then compare values -- a missing
    # field and a changed value are two different kinds of error
    missing = sorted(set(base) - set(new))
    added = sorted(set(new) - set(base))
    bad, strings, checked, skipped = [], [], 0, 0

    for k in sorted(set(base) & set(new)):
        a, b = base[k], new[k]
        if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) \
                or not isinstance(b, (int, float)):
            # Non-numeric fields are their own category: a refactor may
            # legitimately change a string that records a path (source,
            # enkf-dir, directory names in protocol), but is **never** allowed
            # to change any numeric value. Mixing the two into one judgement
            # would make it impossible to tell "a path moved" from "a
            # computation broke."
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

    print(f"baseline: {args.baseline}")
    print(f"new     : {args.new}")
    print(f"compared {checked} numeric and {skipped} non-numeric fields, tolerance {args.tol:g}\n")

    if missing:
        print(f"[structure] {len(missing)} fields missing from the new output, e.g.:")
        for k in missing[:8]:
            print(f"    {k}")
    if added:
        print(f"[structure] {len(added)} extra fields in the new output (new conventions are expected), e.g.:")
        for k in added[:8]:
            print(f"    {k}")
    if missing or added:
        print()

    if strings:
        print(f"[strings] {len(strings)} differences (path-like fields legitimately change in a refactor, not a failure):")
        for k, a, b in strings[:10]:
            print(f"    {k}\n        baseline {a!r}\n        new      {b!r}")
        print()

    if bad:
        print(f"[values] {len(bad)} mismatches:")
        for k, a, b, d in bad[:25]:
            tail = f"  diff {d:.3e}" if d is not None else ""
            print(f"    {k}\n        baseline {a!r}\n        new      {b!r}{tail}")
        if len(bad) > 25:
            print(f"    ... {len(bad) - 25} more")

    ok = not bad and not missing
    if ok:
        extra = f" ({len(strings)} path-string changes, see above)" if strings else ""
        print(f"\n[PASS] all {checked} numeric values within tolerance, the refactor did not change the results.{extra}")
    else:
        print("\n[FAIL] the refactor changed the results, fix before continuing.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
