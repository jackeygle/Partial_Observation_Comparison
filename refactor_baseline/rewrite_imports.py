"""
rewrite_imports.py — Phase 2's mechanical rewrite (a one-off tool, safe to delete once the refactor is done)

Converts the pre-refactor "bare import after sys.path.insert" pattern into
package-path imports, and removes hardcoded old directories.

Why a script rather than editing by hand: this touches 90 .py files and about
200 import statements, and this class of change **raises no error when done
wrong** -- Python will quietly import a different same-named module (losses.py
has three copies) and compute a different number. A script can dry-run to see
the full diff first, and guarantees the same rule is applied everywhere consistently.

Usage:
    python3 refactor_baseline/rewrite_imports.py            # dry-run, reports only
    python3 refactor_baseline/rewrite_imports.py --apply    # actually rewrite
"""
from __future__ import annotations

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# module name -> which package it now lives in. crowdcore's three are globally
# unique; the rest are resolved by which method the file belongs to.
GLOBAL = {
    "config": "crowdcore.config",
    "navigation": "crowdcore.navigation",
    "observation_model": "crowdcore.observation_model",
    "paths": "crowdcore.paths",
    "prior_model": "methods.varnet.prior_model",
    "variational_solver": "methods.varnet.variational_solver",
    "model_io": "methods.varnet.checks.model_io",
}
# These names exist in multiple methods and must be resolved by context
PER_METHOD = {"losses", "dataset", "model", "train",
              "state", "encoding",                      # dincae
              "network", "sensors", "positional"}       # senseiver

# which method a script under compare/ was moved from -- decides who its dataset/losses point to
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
    """Hollows out the contents of triple-quoted strings (keeping line numbers),
    so regexes can only match real code lines.

    Why this step is needed: crowdcore/config.py's docstring has a line
    `import config` (a Usage example), and the first version of this script
    treated it as a real import and rewrote it. Example code in docs looks
    exactly like real code; only position tells them apart.
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


#: These two files' docs **deliberately** mention the old path (explaining the refactor's history); do not touch
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

    # Processed line by line: only a line that also matches in `masked` (with
    # docstrings hollowed out) is real code
    lines, mlines = text.split("\n"), masked.split("\n")
    for i, (ln, ml) in enumerate(zip(lines, mlines)):
        if re_imp.match(ml):
            lines[i] = re_imp.sub(fix_import, ln)
        elif re_frm.match(ml):
            lines[i] = re_frm.sub(fix_from, ln)
    text = "\n".join(lines)

    # Hardcoded old directory -> paths (files that deliberately mention it are skipped)
    if OLD_ABS in text and os.path.relpath(path, ROOT) not in DOC_ONLY:
        n = text.count(OLD_ABS)
        text = text.replace(f'"{OLD_ABS}"', "paths.method(paths.VARNET)")
        text = text.replace(f"'{OLD_ABS}'", "paths.method(paths.VARNET)")
        text = text.replace(OLD_ABS, "<<PATHS_VARNET>>")   # the rest are stitched into longer strings, for a human to check
        notes.append(f"old absolute path x{n}")

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

    print(f"\n{changed} files, {total_notes} rewrites"
          + (" (written)" if args.apply else " (dry-run, not written)"))
    if manual:
        print(f"\nthe following files have the old path stitched into a longer string, "
              f"placeholdered as <<PATHS_VARNET>>, needs manual handling:")
        for r in manual:
            print(f"    {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
