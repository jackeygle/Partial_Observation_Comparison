# `refactor_baseline/` — the refactor's acceptance baseline

Frozen on **2026-09-03**, after commit `2b2d121`, before the directory refactor.

A refactor is only allowed to change **where the code lives**, never **what the
code computes**. This directory is the executable version of that statement: once
the refactor is done, the same commands must reproduce the numbers here, **bit for
bit**.

## Contents

| File | Source | What it guarantees |
|---|---|---|
| `compare4.json` | `senseiver_crowd/checks/compare4.py` (job 20041056) | Pooled/day-averaged error for 5 models x 6 conventions x 4 channels. This is the main result, and also the broadest single test -- it touches `config`, `navigation`, `observation_model`, Senseiver, 4DVarNet, and the EnKF's export all at once |
| `verify_enkf_opt.out` | `4dvarnet_enkf/checks/verify_enkf_opt.py --frames 12 --ensemble 100` | `enkf_opt` is bit-identical to `enkf_lab`. `enkf_lab` is a read-only vendor copy of `Partial_observation`; this is its entire reason to exist |
| `dincae_metrics_test.json` | `dincae_crowd/checks/evaluate.py` | DINCAE's four conventions |
| `senseiver_metrics_test.json` | `senseiver_crowd/checks/evaluate.py` | Senseiver's per-day/per-channel numbers |

## How to run acceptance

`compare4.py` was renamed to `compare/compare5.py` on 2026-09-04 and gained two
more conventions (`defined_full`, `full`) plus a second ensemble arm (MSE). The
regression test **must run with the same arguments as the baseline**, otherwise
the row names and row count both change, and the comparison just reports a pile of
"missing fields":

    # Reproduce the baseline's row set: the two varnet single models + the arm
    # labelled "nll", with the ensemble explicitly turned on
    # (ensembling has been off by default since 2026-09-04 -- the paper never
    # mentions ensembling, so the main table reports single models)
    python3 -m compare.compare5 --varnet a4_k1,b0_k1 --arms "nll=vsb0_s{}" \
        --with-ensemble --out compare/results/_regression.json
    # The 4DVarNet rows have an inherent ~4e-4 jitter, see "reproducibility floor"
    # below, so the tolerance is set from what was actually measured
    python3 refactor_baseline/check_against_baseline.py \
        compare/results/_regression.json --tol 3e-4

The newly added `defined_full` / `full` conventions will be reported as "extra
fields" -- that is expected and not treated as a failure.

Day-to-day main-result runs use the default arguments (`--varnet a4_k1`, the MSE
and NLL ensembles), producing `compare/results/compare5.json`; that file is not the
regression baseline.

The EnKF check runs separately:

    # After the refactor
    python3 -m methods.enkf.checks.verify_enkf_opt --frames 12 --ensemble 100
    diff <(...) refactor_baseline/verify_enkf_opt.out

## Why this directory exists

The three sub-projects are not Python packages -- they are stitched together with
`sys.path.insert/append`, `losses.py` has three same-named copies, and 72 hardcoded
absolute paths are scattered across 21 files. Those are exactly the things the
refactor needs to touch, and they are **all** the kind of thing that "fails
silently when broken -- it just quietly computes a different number." Without a
numerical baseline, there would be no way to tell after the refactor whether
anything got broken.

Once the refactor is complete and acceptance passes, this directory can be
deleted, or kept as a regression test.


## Reproducibility floor: 4DVarNet's numbers are only valid to 3 significant figures

Of compare4's 957 values, 521 differ from the baseline after the refactor, all of
them concentrated in the 4DVarNet rows, with a largest absolute difference of
2.39e-04 (relative 4.2e-04). Senseiver, EnKF, and DINCAE are **bit-identical** across
all three.

This is not caused by the refactor. Same machine, same code, run twice in a row
(job 20051734):

| Method | largest relative difference between the two runs |
|---|---|
| Senseiver | 0.00e+00 |
| EnKF k1 | 0.00e+00 |
| DINCAE | 0.00e+00 |
| **4DVarNet b0_k1** | **4.18e-04** |

Same order of magnitude as the 4.20e-04 seen before vs. after the refactor -- i.e.
this metric of 4DVarNet's is simply not bit-reproducible to begin with.

Reason: `GradSolver` also runs `torch.enable_grad()` for 15-20 backward passes
**at inference time** (that is how it solves, not training), and conv backward's
reduction uses atomicAdd, whose accumulation order differs run to run. Senseiver
only does a forward pass under `torch.no_grad()`, so it is bit-reproducible; the
EnKF reads from npz, pure numpy.

**Impact on interpreting results**: for any 4DVarNet number, the 4th decimal place
is noise. A conclusion like "a4 is 15% better than b0" is far above this floor
(0.1109 vs 0.1284) and is safe; but do not report a difference at the scale of
"0.1782 vs 0.1781" -- that is inside the noise.
