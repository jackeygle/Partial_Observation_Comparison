"""
rewrite_sbatch.py — Phase 3: convert 34 sbatch scripts to the new calling convention (a one-off tool)

Three changes:

  1. `module load ...`  ->  `source <root>/sbatch/_env.sh`
     _env.sh handles module load, PYTHONPATH, PYTHONSAFEPATH -- defined in one
     place. **`cd <method dir>` is kept** (path updated to the new location):
     118 relative paths in the scripts are all relative to the method
     directory, so keeping cd means none of them need to change.
     PYTHONSAFEPATH=1 already keeps cwd off sys.path, so staying in the method
     directory is safe.

  2. `python3 -u <script path>.py`  ->  `python3 -u -m <package path>`
     -m is required. `python3 methods/varnet/train.py` would put
     methods/varnet/ onto sys.path[0], bringing back the problem of the three
     same-named losses.py files shadowing each other.

  3. The old directory in `#SBATCH --output=` -> the new directory.

Relative paths in the scripts (--outdir runs/varnet_b0_k1, submit_x_chain.sbatch's
self-resubmission, and the like) don't need to change at all, because cwd stays
the method directory. This is deliberate: the first version had _env.sh move cwd
to the repo root, which would have pointed all 118 relative paths somewhere
wrong -- pure risk, no benefit.

Usage:
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

#: old method directory -> new method directory
DIRMAP = {
    "4dvarnet_enkf": "methods/varnet",
    "dincae_crowd": "methods/dincae",
    "senseiver_crowd": "methods/senseiver",
}
#: old script relative path -> new package path. The cross-method ones from checks/ have already moved to compare/.
MOVED_TO_COMPARE = {"eval_threeway_accuracy", "compare_channels", "plot_comparison",
                    "plot_reconstruction_enkf", "bench_speed", "plot_speed",
                    "plot_frameworks", "compare3", "compare4", "plot_compare3"}
MOVED_TO_ENKF = {"bench_enkf_opt", "bench_enkf_split", "diag_enkf_spread_growth",
                 "diag_sparsification_enkf", "eval_uncertainty_enkf", "export_obs_for_enkf",
                 "run_enkf_baseline", "score_enkf", "verify_enkf_gain_mode",
                 "verify_enkf_opt", "plot_velocity_enkf"}
#: top-level script rename
RENAMED = {"train_varnet": "train"}


def module_for(old_dir: str, script_rel: str) -> str:
    """('4dvarnet_enkf', 'checks/eval_threeway_accuracy.py') -> 'compare.eval_threeway_accuracy'"""
    parts = script_rel[:-3].split("/")            # strip .py
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

            # figure out which method this script originally belonged to
            owner = next((d for d in DIRMAP if f"{OLD}/{d}" in s), None)

            # 1) old directory -> new directory. Both forms need handling: with
            # a trailing slash (--output=.../runs/...) and without one at the
            # end of the line (`cd .../4dvarnet_enkf`). The first version only
            # replaced the former, so not a single cd line got fixed -- and
            # that is the single most important line.
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

            # 3) `module load ...` -> `source _env.sh`, **cd kept**
            #
            # _env.sh deliberately does not cd: the scripts have 118 relative
            # paths (runs/, check_outputs/, sbatch/'s self-resubmission) all
            # relative to the method directory. Keeping cd means none of them
            # need to change; PYTHONSAFEPATH=1 already keeps cwd off sys.path,
            # so staying in the method directory is safe.
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

            # cwd is still the method directory, relative-path arguments don't
            # need touching -- only worth checking scripts without a cd
            if owner and f"cd {OLD}/{DIRMAP[owner]}" not in s:
                manual.append((rel, "no cd to the method directory, relative paths need confirming"))

    print(f"\n{n} sbatch scripts rewritten" + (" (written)" if args.apply else " (dry-run)"))
    if manual:
        print(f"\ncwd changed from the method directory to the repo root; the following "
              f"relative-path arguments need manual confirmation ({len(manual)}):")
        for rel, arg in manual:
            print(f"    {rel}: {arg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
