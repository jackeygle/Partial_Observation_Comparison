"""
paths.py — single source of truth for paths inside the repo

Before the refactor, the string `/scratch/work/zhangx29/Thesis_Project/4dvarnet_enkf`
was hardcoded in 22 places, with several more hardcoding `dincae_crowd`. The
2026-09-03 directory refactor broke all of them silently -- that kind of failure
never raises an error, it just makes a script read no file, or someone else's file.
So paths are centralised here, deriving the repo root from **this file's own
location** and following the repo wherever it moves.

The data itself (ATC's h5 files, grid_cache, the real map) is not here -- it lives
outside the repo, with its path in `config.yaml`'s `data.root` / `navigation.map_dir`,
which is user-editable configuration, not a code constant.

Usage:
    from crowdcore import paths
    paths.method("varnet")                  -> <root>/methods/varnet
    paths.runs("varnet")                    -> <root>/methods/varnet/runs
    paths.check_outputs("dincae")           -> <root>/methods/dincae/check_outputs
    paths.enkf_export("enkf_k1_full")       -> <root>/methods/varnet/check_outputs/enkf_k1_full
    paths.REFERENCE_IMPL                    -> /scratch/work/zhangx29/Partial_observation
"""
from __future__ import annotations

import os

#: Repo root. This file lives at <root>/crowdcore/paths.py, so two levels up is the root.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CROWDCORE = os.path.join(ROOT, "crowdcore")
METHODS = os.path.join(ROOT, "methods")
COMPARE = os.path.join(ROOT, "compare")
SLIDES = os.path.join(ROOT, "slides")
BASELINE = os.path.join(ROOT, "refactor_baseline")

#: Directory names of the five methods. Kept here rather than as scattered string
#: literals, so a rename only touches one place.
VARNET, ENKF, DINCAE, SENSEIVER = "varnet", "enkf", "dincae", "senseiver"

#: The reference implementation (Kazemi Eskeri et al., IROS 2025 workshop).
#: **Outside this repo** -- we only read it, never write to it; methods/enkf/enkf_lab/
#: is a read-only vendor copy of it.
REFERENCE_IMPL = "/scratch/work/zhangx29/Partial_observation"


def method(name: str) -> str:
    """<root>/methods/<name>"""
    return os.path.join(METHODS, name)


def runs(name: str) -> str:
    """Training output: checkpoints, metrics.jsonl, slurm logs."""
    return os.path.join(METHODS, name, "runs")


def run_dir(name: str, tag: str) -> str:
    """A single training run, e.g. run_dir("varnet", "b0_k1") -> methods/varnet/runs/varnet_b0_k1.

    Note varnet's directory names carry a `varnet_` prefix (for historical reasons);
    this function does not add that prefix for the caller, callers pass the full
    directory name.
    """
    return os.path.join(METHODS, name, "runs", tag)


def check_outputs(name: str) -> str:
    """Output of the diagnostic scripts: metric jsons, figures, slurm logs."""
    return os.path.join(METHODS, name, "check_outputs")


def eval_out(name: str) -> str:
    """check_outputs/eval -- where each method puts its final metrics and figures."""
    return os.path.join(METHODS, name, "check_outputs", "eval")


def enkf_export(which: str = "enkf_k1_full") -> str:
    """The EnKF's exported estimated fields (est_*.npz / obs_*.npz), 9 directories, ~17 GB.

    Moved on 2026-09-03 from methods/varnet/check_outputs/ to
    methods/enkf/check_outputs/ -- they are the EnKF's output, and sitting under the
    4DVarNet directory was only a historical artifact (from when the two methods
    shared one directory). /scratch is a single filesystem, so moving 17 GB was an
    instant rename, not a copy.

    The move was deliberately scheduled **after** the numerical acceptance check:
    prove the refactor did not change any number first, then move the data --
    otherwise the two variables get tangled, and a problem afterwards cannot be
    traced to a broken import versus a moved path.
    """
    return os.path.join(METHODS, ENKF, "check_outputs", which)


def enkf_vendor(which: str = "enkf_lab") -> str:
    """methods/enkf/enkf_lab or enkf_opt.

    These two directories are **deliberately not part of the Python package**: they
    contain a `pedpred -> .` self-symlink, relying on `<dir>` being placed on
    sys.path so that `from pedpred.X import Y` inside the package resolves.
    See methods/enkf/__init__.py.
    """
    assert which in ("enkf_lab", "enkf_opt"), which
    return os.path.join(METHODS, ENKF, which)

def require_ckpt(path: str, what: str = "checkpoint") -> str:
    """Return `path`, or exit with an actionable message if it does not exist.

    A fresh clone has no `runs/` (gitignored) and no ATC data, so every
    checkpoint load fails -- and torch's own FileNotFoundError arrives at the
    bottom of a 15-line traceback naming a file the reader has never heard of.
    This says what is actually missing and what to do instead.
    """
    if os.path.exists(path):
        return path
    raise SystemExit(
        f"[missing {what}] {path}\n"
        f"\n"
        f"  runs/ is gitignored, so a fresh clone has no trained weights, and the\n"
        f"  gridded ATC data is not in git either -- neither is needed to READ the\n"
        f"  already-computed results: see compare/results/ and methods/*/check_outputs/.\n"
        f"\n"
        f"  To actually run inference you need the ~1.4 GB package described in\n"
        f"  README.md under 'The minimal package: ~1.4 GB, not 225 GB'.")
