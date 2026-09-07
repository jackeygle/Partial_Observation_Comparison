"""
import_all.py — Phase 3's first acceptance check: does every import target resolve?

The refactor changed 186 imports, 95 sys.path lines, and every hardcoded old
path. This kind of change **raises no syntax-check error when done wrong** --
it only blows up at actual import time, or worse, silently imports a different
same-named module.

Two tiers, because these two things must be kept separate:

  Tier A (static, executes no code)
      Parses every .py with ast, feeds every one of its import targets to
      importlib.util.find_spec. Covers every file, including plotting/eval
      scripts that run main() at the top level.

  Tier B (real import)
      Only genuinely imports **side-effect-free library modules**: crowdcore's
      four, and each method's model/losses/dataset/... Incidentally confirms
      the three same-named losses.py files stay independent -- that is the
      main point of the whole refactor.

The first version did importlib.import_module on all 99 modules directly,
which ran every evaluation/plotting script's top-level code (many scripts have
no `if __name__` guard), blowing past 32G of memory after 14 minutes.
"Can it be imported" and "run it once" are two different things; only the
former is checked here.

Usage (the login node is fine, but source sbatch/_env.sh first to get torch):
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

#: Tier B: these modules only have definitions at the top level; importing them runs no computation
SAFE_LEAVES = {"config", "navigation", "observation_model", "paths",
               "model", "losses", "dataset", "network", "positional", "sensors",
               "state", "encoding", "prior_model", "variational_solver"}

#: These top-level names come from vendor copies, injected onto sys.path at runtime; static resolution can never find them
VENDOR_ROOTS = {"pedpred"}


def py_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in sorted(filenames):
            if fn.endswith(".py"):
                yield os.path.join(dirpath, fn)


def import_targets(path: str):
    """(module name, line number) -- every top-level module this file imports. Relative imports are skipped."""
    try:
        tree = ast.parse(open(path).read(), filename=path)
    except SyntaxError as e:
        yield (f"<syntax error: {e}>", getattr(e, "lineno", 0))
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield (a.name, node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield (node.module, node.lineno)


def tier_a():
    print("[Tier A] statically resolving every import target (executes no code)\n")
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
                    fails.append((lineno, f"cannot find {mod}"))
            except (ImportError, ModuleNotFoundError, ValueError) as e:
                fails.append((lineno, f"{mod}: {type(e).__name__} {e}"))
        if fails:
            bad += len(fails)
            print(f"  {rel}")
            for lineno, msg in fails:
                print(f"      :{lineno}  {msg}")
    print(f"\n  {'every import target resolves' if not bad else f'{bad} could not be resolved'}")
    return bad


def tier_b():
    print("\n[Tier B] actually importing the side-effect-free library modules\n")
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
    print(f"  {len(mods) - bad}/{len(mods)} library modules imported successfully")

    print("\n[same-name module isolation] the main point of the refactor")
    import methods.dincae.losses as ld
    import methods.senseiver.losses as ls
    import methods.varnet.losses as lv
    for tag, m in (("varnet", lv), ("dincae", ld), ("senseiver", ls)):
        print(f"  {tag:<10} {os.path.relpath(m.__file__, ROOT)}")
    assert len({lv.__file__, ld.__file__, ls.__file__}) == 3, "the three losses files point to the same file!"
    print("  the three same-named losses files are each independent")
    return bad


def main():
    a = tier_a()
    b = tier_b()
    print(f"\n{'[PASS]' if not (a or b) else '[FAIL]'} Tier A {a} problems, Tier B {b} modules failed")
    return 1 if (a or b) else 0


if __name__ == "__main__":
    sys.exit(main())
