# `enkf_opt/` — the EnKF copy we are allowed to modify

Starts as an exact copy of `../enkf_lab/` (which is itself a pristine copy of the
`Partial_observation` baseline). Experimental changes go HERE; `enkf_lab` stays untouched so
every change can be diffed and, more importantly, numerically verified against it.

## Rule for every change

A change is only acceptable if it does not alter results. Before using a modified version in
any experiment, run the same call on real data through both copies and require

    np.array_equal(enkf_lab_output, enkf_opt_output)   # bit-identical, not "close"

If the outputs differ at all, the change is rejected — or, if a difference is unavoidable and
justified, it must be stated explicitly wherever the numbers are reported.

The check is automated:

    python3 checks/verify_enkf_opt.py --frames 12 --ensemble 100

It drives both copies over the same real frames and compares the analysis mean, the ensemble
spread, the whole ensemble, and the internal `bias_estimate` (that one feeds the next forecast,
so a difference there would compound silently). It also unit-checks each rewritten helper
against the reference implementation kept alongside it (`_localization_matrix_ref`,
`C.nonzero()[1]`, the original bias-backprojection loop, the original `GeneratePartialObs`
loops). Last run: **PASS**, all four arrays bit-identical, all unit checks identical.

## Round 1 — `_localization_matrix` (done earlier)

Profiling one k=1 assimilation step (cProfile, 20 frames, 4 cores; the absolute numbers are from
a more contended node than the benchmark table further down):

    run_day                                    112.5 s
      ENKF.step                                110.2 s
        ENKF.update                            109.5 s
          ENKF._localization_matrix            105.2 s   <-- 96% of total
          numpy pinv + svd                       2.5 s   <-- 2%  (the actual Kalman math)

`_localization_matrix` was a four-deep Python loop over `len(obs_cells) x num_features x H x W`
= 252 x 4 x 36 x 12 per call. Vectorising it gave ~18x end-to-end.

### Care needed when vectorising `_localization_matrix`

The original computes the weight as

    dist = np.sqrt((r - r_obs)**2 + (c - c_obs)**2)
    if dist <= radius:  w = np.exp(-(dist**2) / (2 * radius**2))

Note the round trip: the squared distance is square-rooted and then squared again. That is not
algebraically free in floating point. A vectorised version must reproduce the same sequence
(sqrt, then square) — computing `d2` directly gives slightly different last bits and would fail
the bit-identical check. The threshold test also uses the square-rooted value.

Column layout to preserve: `col_idx = f * H * W + r * W + c`, and the weight does not depend on
the feature index `f`, so the (H*W) block simply repeats `num_features` times across the row.

## Round 2 — the rest of the step

With the loop gone, a re-profile (200 frames of `atc-20130811`, 50 of them observed, m in
[436, 1216], ensemble 100, 16 cores of a batch-csl node) showed the remaining time was:

    pinv -> svd (m x m)                         4.67 s   44%   <-- the Kalman math, now the top cost
    forecast (RNG draw + surrogate)             4.24 s   40%
    C.nonzero()                                 1.4  s   13%   <-- called THREE times per step
    _localization_matrix                        0.32 s    3%   <-- called TWICE per step

Two independent things were wrong with that: the step recomputed fixed quantities, and the gain
was evaluated in the most expensive way available.

### (a) Dead recomputation and allocation — bit-identical, always on

`update()` computed the localization matrix twice, `C.nonzero()` three times, and `X_ano` /
`Y_ano` / `X_mean` / `y_mean` twice, all with identical arguments and no state change in between
(the second block in `enkf_lab/pedpred/ENKF.py` is dead code). Everything is now computed once.
Alongside that:

  * `C.nonzero()[1]` -> `_obs_indices(C)`: `argmax` plus a total nonzero count, two cheap passes
    instead of a scan plus two index arrays, with a fallback to the original call if `C` is not
    a pure selection matrix. ~9 ms -> ~3 ms, and once instead of three times.
  * `Loc` is built as one `(m, H*W)` block and broadcast over the feature blocks instead of
    being `np.tile`d to `(m, state_dim)` (14 MB written per step). `exp` is evaluated only on
    cells that pass the radius test.
  * `R` and `regularization*np.eye(m)` are diagonal — added onto the diagonal in place instead
    of materialising two dense m x m matrices (8 MB each) to add whole.
  * `np.sqrt(np.diag(R))` was recomputed on every ensemble member; hoisted.
  * `_clip_bounds` clips into a fresh buffer instead of `X.copy()` then four clips into it.
  * `_backproject_obs_bias` was a Python loop over m; one scatter (same last-write-wins result
    for repeated indices, which happen when two agents' ranges overlap).
  * inflation done in place, `0.01 * proc_noise_vec` hoisted out of `forecast`.
  * `GeneratePartialObs.cells_within_range` / `get_observation_matrix` /
    `reconstruct_observation` vectorised (used by `ENKF.main()`, not by the checks drivers).

None of this touches arithmetic, so the default path stays bit-identical. Verified.

Two things were deliberately NOT done on this path, because they would change the last bits:

  * `Y_f = (C @ X_f.T).T` -> `X_f[:, obs_idx]`. The values read are identical (C is 0/1), but
    the matmul returns a Fortran-ordered array and `np.mean`'s pairwise-summation order depends
    on the memory layout, so `y_mean` would differ. Used only in `gain_mode="ensemble"`.
  * the per-member loop -> one GEMM. Same operands, different accumulation order.

### (b) `gain_mode` — how the gain is evaluated

    K = (P_xy @ pinv(P_yy)) * Loc.T

`pinv` is an SVD of an m x m matrix (m = 4 x observed cells, ~800-1200 here) whose informative
rank is at most the ensemble size N = 100, followed by a `state_dim x m x m` product: ~14 GFLOP
per step. `P_yy = c*Y_ano^T Y_ano + D` with `D` diagonal, so Woodbury collapses the whole gain to

    P_xy P_yy^-1  =  X_ano^T A^-1 Md,   Md = c*Y_ano D^-1,   A = I_N + Md Y_ano^T

with `A` only N x N: ~0.2 GFLOP per step. `A = I + c*Y_ano D^-1 Y_ano^T` is SPD with eigenvalues
>= 1, so it is also better conditioned than forming `P_yy^-1` at all. `D` is strictly positive
(`obs_std**2 + 1e-3`) and the measured condition number of `P_yy` is far below the point where
`pinv`'s rcond cutoff truncates anything, so `pinv(P_yy) == P_yy^-1` here — the two routes are
the same gain, not two different regularisations.

Two exact extras ride along: observations whose localization row is all zero get an all-zero
gain column and contribute nothing, so they are dropped from the two big products (~70% of rows
under the `observed_cells=None` fallback, see below), and the member loop becomes one GEMM.

    gain_mode="pinv"      (default) the original expression -> bit-identical to enkf_lab
    gain_mode="ensemble"            Woodbury -> ~6x faster per analysis step, NOT bit-identical

Default comes from `$ENKF_GAIN_MODE`, so an existing driver can be switched without an edit:

    ENKF_GAIN_MODE=ensemble python3 checks/run_enkf_baseline.py --enkf-src opt ...

### Is `gain_mode="ensemble"` safe to run experiments with?

It is a different rounding, so the rule above says it has to be reported as such. What it
actually costs was measured over 400 frames of `atc-20130811` (100 observed), both modes from
the same seed with the same number of RNG draws (`checks/verify_enkf_gain_mode.py`):

    max |estimate difference|      2.1e-15   (3.5e-15 relative)
    RMS |estimate difference|      7.5e-17
    worst frame                    t=0  — the drift does NOT amplify over the run
    mean MSE vs ground truth       0.03367063 both modes (delta +2e-14 %)
    analysis skips (LinAlgError)   0 both modes

So the reported metric is unchanged to 14 significant digits and the feedback loop
(analysis -> bias EMA -> surrogate -> clipping) damps rather than amplifies the difference. The
default is still `pinv` so that previously published numbers stay exactly reproducible.

## Measured effect

    srun -p batch-csl -c 16 --mem=24G -t 1:30:00 python3 -u checks/bench_enkf_opt.py \
        --frames 200 --repeats 2 --make-prev /tmp/enkf_prev

All variants in ONE process on ONE node, interleaved, after warming up each: 200 frames of
`atc-20130811`, 50 of them observed (m in [436, 1216], mean 799), ensemble 100, radius 7, 16
cores of csl23 (Xeon Gold 6248). Spread across repeats was <= 0.02 s, i.e. below the width of
the numbers below. Full output in `check_outputs/eval/bench_enkf_opt.json`.

| variant                                   |  s / 200 frames | ms/frame | cores used | vs round 1 | vs original |
|-------------------------------------------|----------------:|---------:|-----------:|-----------:|------------:|
| `enkf_lab`, untouched original            |          200.57 |   1002.9 |       3.05 |            |        1.0x |
| round 1 (localization vectorised)         |            9.57 |     47.8 |      15.95 |      1.00x |       21.0x |
| round 2, `gain_mode="pinv"` (default)     |            7.90 |     39.5 |      15.95 |      1.21x |       25.4x |
| round 2, `gain_mode="ensemble"`           |            4.02 |     20.1 |      15.95 |      2.38x |       49.8x |

`cores used` is CPU-seconds / wall-seconds (1.0 = one core busy throughout): the original's
Python loop is serial, which is why it cannot use the 16 cores it was given, while everything
after round 1 is BLAS-bound and does.

Only 50 of these 200 frames carry observations, so the end-to-end column understates the effect
on the part that was actually optimised. Subtracting the forecast cost (identical in all three,
~16 ms/frame here) leaves the analysis step at roughly

    round 1   ~127 ms   ->   round 2 pinv   ~97 ms   ->   round 2 ensemble   ~16 ms

The gap widens with m, since the SVD is O(m^3) while the ensemble-space form is linear in m, and
it widens with observation density — at k=1 with every frame observed the analysis is paid every
frame rather than every fourth.

### What is left

`forecast()` is now the largest single cost in `gain_mode="ensemble"` — it runs on every frame,
observed or not, and takes ~80% of that variant's time (profiled: ~10 ms in the surrogate's
convolutions, ~7 ms drawing 100 x 1728 legacy-MT19937 gaussians, the rest clipping). The RNG
cost is not reducible without changing the random stream, which would change results outright.
The surrogate is reducible: `_f_model` now takes the device from the model instead of hard-coding
CPU, so loading the model onto a GPU moves those convolutions there.
That also fixes an actual bug — `ENKF.main()` and `ENKF.estimate_noise()` load the model with
`DEVICE="cuda"` when a GPU is present, and the hard-coded CPU input tensor made them crash on a
device mismatch.

## A pre-existing bug that is preserved on purpose

When `update()` is called with `observed_cells=None` — which is what every driver in this
project does — the reference derives the cell for observation row `k` as
`(obs_idx[k] // W, obs_idx[k] % W)`. `obs_idx` is a *state* index, `f*H*W + r*W + c`, so the
"row" comes out as `f*H + r`: correct for the density channel, off the grid for the other three.
Those rows land further from every grid cell than the localization radius, so their localization
weights are all zero and those observations do not influence the analysis at all — the filter
effectively assimilates only via the f=0 rows.

This is NOT fixed here. It is what the published EnKF baseline numbers were produced with, and
changing it would change results, which this copy is not allowed to do silently. It is also
what makes the active-column shortcut in `gain_mode="ensemble"` so effective (~70% of rows
dropped). If the baseline is ever re-run, this is the first thing to revisit — pass real
`observed_cells` and the filter gets three more channels of information.
