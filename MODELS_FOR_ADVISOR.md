# Trained models, evaluation scripts, and test-day range — for the advisor

Prepared for the 2026-09-07 meeting follow-up ("share the 5 trained models,
eval scripts, and the tested data range"). Framing first, because it matters:
this is **4 independent reconstruction methods + 2 of our own 4DVarNet
uncertainty designs**, not 5 unrelated models — see `README.md`'s method
table and `slides/build_meeting4_deck.py`'s capability slide for the same
framing already shown in the meeting.

## Held-out test days (identical across all methods)

7 days, `split=test`, never seen during training or checkpoint selection:

```
atc-20130811  atc-20130818  atc-20130825  atc-20130901  atc-20130915
atc-20130922  atc-20130929
```

Grid: 36×12 cells, 4 channels (density, vx, vy, var); 290/432 cells are
physically walkable, the rest (142 cells, 32.9%) are the obstacle region
discussed in the meeting.

**Result** (`methods/dincae/checks/measure_obstacle_region.py`, full 7-day
test split, `methods/dincae/check_outputs/eval/obstacle_region.json`):
obstacle-region truth is non-zero only 2.99–3.11% of the time across all four
channels — matching the advisor's "3–4%" estimate almost exactly. The
information-form baseline (`dincae_full`, obstacle cells never in its loss)
is fine on density there (never learns a wrong bias) but is wildly wrong on
velocity — vy MSE 3.85 there vs 0.054 on walkable cells, a >1300x gap —
because it was never taught a target for a channel that only exists where
`density>0`, which is never true in the obstacle region during training. The
`--full-field-loss` ablation (`dincae_ff`, obstacle cells supervised against
physical 0) closes almost all of that gap (vy obstacle MSE 0.0029, a
1340x improvement) at essentially no cost to walkable-region accuracy
(vx walkable MSE actually improves, 0.087 vs 0.133; density/var walkable
MSE within noise). So: **not a bug, DINCAE's information form structurally
cannot learn "obstacle = 0" unless the obstacle region is put in the loss**,
and doing so is nearly free.

## The 4 reconstruction methods

| Method | Checkpoint(s) | Eval script | Result JSON |
|---|---|---|---|
| 4DVarNet (plain MSE, Eq.14) | `methods/varnet/runs/varnet_mse5_s{0..4}/varnet_best.pt` (5 seeds) | `methods.varnet.checks.eval_test_days` / `compare.compare5` | `compare/results/compare5_final.json` (`"4DVarNet MSE s*"`) |
| DINCAE (information form, 16-ckpt output averaging) | `methods/dincae/runs/dincae_full/ckpt_*.pt` | `methods.dincae.checks.evaluate` | `methods/dincae/check_outputs/eval/dincae_metrics_test.json` |
| Senseiver | `methods/senseiver/runs/senseiver_A/best.pt` | `methods.senseiver.checks.evaluate` | folded into `compare5_final.json` (`"Senseiver"`) |
| localised EnKF (+ PedPred3 forward model) | no learned weights — exported ensemble fields at `methods/enkf/check_outputs/enkf_k1_full/` | `methods.enkf.checks.score_enkf` | folded into `compare5_final.json` (`"EnKF k1"`) |

## The 2 4DVarNet uncertainty designs (ours, not in either paper)

| Design | Checkpoint(s) | How σ is produced | Result JSON |
|---|---|---|---|
| `aug0` — sigma inside the prior operator `G(x)` | `methods/varnet/runs/varnet_aug0_s{0..4}/varnet_best.pt` | joint gradient through `G(x)`, NLL loss | `methods/varnet/check_outputs/eval/uncertainty_aug0.json` |
| `vsb0` — read-out head only | `methods/varnet/runs/varnet_vsb0_s{0..4}/varnet_best.pt` | separate head off the final state, NLL loss | `methods/varnet/check_outputs/eval/uncertainty_vsb0.json` |

`aug0` is the one worth presenting in detail next week (better CRPS, sign-stable
across scoring conventions); `vsb0` is kept as the negative control that shows
*why* the bolt-on design underperforms.

DINCAE and EnKF have their uncertainty *built into* the method (information
form's σ̂, ensemble spread respectively) rather than as a separate design —
their result JSONs are `methods/dincae/check_outputs/eval/uncertainty_dincae.json`
and `methods/enkf/check_outputs/eval/uncertainty_enkf_k1.json`. The EnKF's
overconfidence anomaly the advisor flagged is diagnosed separately in
`methods/varnet/check_outputs/eval/enkf_spread_growth.json` (via
`methods.enkf.checks.diag_enkf_spread_growth`) — verdict: `"contractive"`, the
deterministic PedPred3 forecast model itself damps ensemble spread by ~65%
per step regardless of injected noise, so it is not a bug to fix but a
property of the forward model to report as-is.

## One command to reproduce the whole accuracy comparison

```bash
source sbatch/_env.sh
sbatch sbatch/submit_compare5.sbatch
python3 -m compare.plot_compare5
```

## What is and isn't included if this repo is shared as-is

**Correction (this line used to overclaim):** the GitHub repo (code, configs,
small result JSON/PNG, ~nothing over a few MB) is *not* by itself enough to
rerun anything — `git ls-files | grep '\.h5$'` returns **zero** files. The
gridded ATC data (`*.h5`, what every method actually reads) is excluded by
`.gitignore`'s blanket data rules and has never been in git at all, most
likely because the underlying ATC dataset carries its own redistribution
terms, not only because of size. So actually running inference — not just
reading the already-computed results in `check_outputs/` — needs the code
**and** a copy of the gridded data **and** the checkpoint, three separate
things, not one repo clone.

It also does **not** include the `.pt` checkpoint weights themselves (~1 GB)
or the EnKF's exported `.npz` fields (~18 GB) — those are gitignored by
design (see README's "What is not in the repo"). If the advisor wants the
actual trained weights rather than just being able to reproduce them, that
needs a separate transfer off `/scratch/work/zhangx29/Thesis_Project/methods/*/runs/`.

**DINCAE needs one more small file that weights alone don't carry**:
`methods/dincae/artifacts/state_stats.npz` (17KB — per-cell mean field and
per-channel residual std, computed once from the training data, read
separately from the checkpoint at inference time by `state.StateStats`). It
used to be gitignored too, caught by the blanket `*.npz` rule meant for
multi-GB files rather than this one; it is now tracked via an explicit
negation in `.gitignore`, so a fresh clone has it. 4DVarNet and Senseiver
don't have this problem: their normalisation is either baked into the
checkpoint itself (Senseiver's `in_mean`/`in_std` are model buffers, saved
in the same `state_dict`) or not needed at inference time (4DVarNet works in
raw physical units).

## The minimal data package: ~1.4 GB, not 225 GB

The data on Triton spans three pipeline stages in two locations, and **only
the last stage is read at run time**. Anyone reproducing inference needs far
less than the raw dataset:

| Stage | Location | Size | Needed to run? |
|---|---|---|---|
| ① raw ATC CSVs (92 recording days, the true original) | `/scratch/work/zhangx29/ATC/` | 225 GB | **No** — only to rebuild ② from scratch |
| ② trajectory H5, Sundays only (46 days) | `/scratch/work/zhangx29/data/ATC/Sundays/` | 36 GB | **No** — see the note below |
| ③ `grid_cache`, 4-channel 36×12 fields (46 days) | `/scratch/work/zhangx29/data/grid_cache/` | 3.0 GB | **Yes** — this is what every method opens |

Note on ②: the split lists name `ATC/Sundays/atc-YYYYMMDD.h5`, but
`observation_model.split_files()` only takes the filename stem from them and
opens `grid_cache/<stem>_corridor_1.0s.h5` instead. Those 36 GB of
trajectory H5 are therefore a **naming manifest only** — never opened at run
time, and not worth transferring.

So the complete package to hand over is:

| Item | Path | Size |
|---|---|---|
| gridded fields, 7 test days only | `data/grid_cache/atc-{20130811,20130818,20130825,20130901,20130915,20130922,20130929}_corridor_1.0s.h5` | **331 MB** (55–77 MB each; all 46 days would be 3.0 GB) |
| split lists | `data/sunday_atc_{train,valid,test}.lst` | a few KB |
| the real ATC map (walkable/obstacle mask) | `/scratch/work/zhangx29/project_analysis/partial_observation_experiments/shared_workspace/robot_exploration/atc_map/` (`localization_grid.pgm` + `.yaml`) | 3.3 MB |
| DINCAE normalisation stats | `methods/dincae/artifacts/state_stats.npz` | 17 KB (already in git) |
| trained weights | `methods/*/runs/<run>/` | ~1 GB |
| | **total** | **~1.4 GB** |

**The map directory is the one people forget**: `config.yaml`'s
`navigation.map_dir` points outside this repo *and* outside the data root,
into a different project's directory tree. Both external paths are set in
`crowdcore/config.yaml` (`data.root` and `navigation.map_dir`) and are the
only two that need repointing on another machine.

(`paths.REFERENCE_IMPL` → `/scratch/work/zhangx29/Partial_observation` is a
third external path, but it is *not* a runtime dependency — the EnKF's
vendored byte-identical copy already lives in-repo at
`methods/enkf/enkf_lab/`; the original is referenced only for provenance
checks and speed benchmarking.)
