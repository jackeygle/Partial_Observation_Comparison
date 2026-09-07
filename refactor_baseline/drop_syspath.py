"""
drop_syspath.py — Phase 2's second step: remove the now-unneeded sys.path
preamble (a one-off tool)

Before the refactor, every script had to stitch the project root onto sys.path
itself to import sibling modules. Now they are real packages, importing via
`crowdcore.*` / `methods.*.*`; these preambles are not just useless, they also
put a method's own directory onto the search path -- exactly the mechanism that
let losses.py's three same-named copies shadow each other.

**Kept**: insertions that mention enkf_lab / enkf_opt in their arguments -- those
two vendor directories contain a `pedpred -> .` self-symlink, relying on that
directory being placed on sys.path so that `from pedpred.X import Y` inside the
package resolves. Removing it would mean changing the vendor copy's behaviour.

Usage:
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

# Whitelist-style deletion: only removes the handful of forms **recognised** as
# "stitching the project root onto the search path"; every other form is kept
# and reported, for a human to look at.
#
# The first version used a blacklist (kept a line if it mentioned
# enkf_lab/enkf_opt), and it nearly deleted the
# `sys.path.insert(0, root)` at methods/enkf/checks/verify_enkf_opt.py:54 --
# there `root` is a variable, the line has no "enkf" in it, and that script is
# exactly the refactor's acceptance test. A blacklist is the wrong direction
# for this task: a wrong deletion raises no error, it just silently imports a
# different module.
DROP_ARGS = (
    "os.path.dirname(os.path.dirname(os.path.abspath(__file__)))",
    "os.path.dirname(os.path.abspath(__file__))",
    "ROOT", "R", "_V4D", "V4D", "root_dir", "PROJ",
    # the previous step, rewrite_imports.py, replaced hardcoded old paths with
    # this form; also no longer needed after packaging
    "paths.method(paths.VARNET)",
    'os.path.join(ROOT, "checks")', "os.path.join(ROOT, 'checks')",
    'os.path.join(R, "checks")', "os.path.join(R, 'checks')",
    'os.path.join(V4D, "checks")', "os.path.join(V4D, 'checks')",
)


def _arg_of(line: str) -> str:
    """Extracts X from sys.path.insert(0, X) / sys.path.append(X), stripping
    whitespace and a trailing comment."""
    inner = line[line.index("(") + 1:line.rindex(")")]
    if inner.lstrip().startswith("0,"):
        inner = inner.lstrip()[2:]
    return inner.strip()


def _should_drop(line: str) -> bool:
    if "Thesis_Project" in line:        # a hardcoded old absolute path, always drop it
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

        # `if X not in sys.path:` + an indented append -- handled as a pair
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
                print(f"{rel}\n  = kept {k}")
            if args.apply and dropped:
                open(p, "w").write(new)

    print(f"\ndropped {n_drop} occurrences ({n_files} files), kept {n_keep} enkf-vendor-related ones"
          + (" (written)" if args.apply else " (dry-run)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
