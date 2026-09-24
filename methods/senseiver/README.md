# methods/senseiver — reproducing Senseiver on the ATC crowd field

> Part of [Partial Observation Comparison](../../PROJECT_OVERVIEW.md) — see the project overview for the problem statement, the scoring-convention pitfall that decides the ranking, and which checkpoint backs which published number.

Two rows of the final six-method comparison come from this directory. It shares only
the **data pipeline and observation configuration** with the other methods
(`crowdcore/`), and imports none of them.

Paper: Santos et al., *The Senseiver: attention-based global field reconstruction
from sparse observations* (NeurIPS ML4PS 2022 workshop; published version Nature MI
2023). Reference implementation:
[`OrchardLANL/Senseiver`](https://github.com/OrchardLANL/Senseiver) (the official PyTorch
code).

**The goal is to reproduce the paper's method**, not to improve on it first. Every
place that deviates from the reference implementation is listed below with its
reason.

---

## Status: this line is closed (2026-09-18)

**The baseline is the deliverable.** Two checkpoints are kept, and both are in the
final comparison evaluated by `supervisor_evaluation/evaluate.py` (numbers in the
repository [README](../../README.md#results-seven-test-days)):

| checkpoint | row in the final comparison | what it is |
|---|---|---|
| `runs/senseiver_A/best.pt` | Senseiver-A | faithful reproduction of the paper (variant A); blind walkable RMSE 0.222 |
| `runs/capacity/base32_k16_s123/best.pt` | **Senseiver-G (ours)** | grid latent with direct read-out ("G-direct") plus a 16-frame causal observation window (k=16), 60 epochs, seed 123; blind walkable RMSE 0.197, the best of the six methods. The control every experiment below was compared against |

Everything after this section is an **extension that did not beat the baseline**.
Nothing is pending — the sections are kept as the record of what was tried.

Seven-day test set, walkable pooled blind RMSE, identical protocol
(`check_outputs/history_refinement/eval/`):

| run | RMSE | vs baseline |
|---|---:|---:|
| baseline `capacity/base32_k16_s123` | 0.19687 | — |
| `framewise/k16_s123` | 0.19707 | +0.10% |
| `history_loss/w05_k16_s123` | 0.19791 | +0.53% |

Trained to 60 epochs but never evaluated on the test set, because their validation
blind MSE never cleared the baseline's 0.01850 (seed 123, matched budget):

| run | valid blind MSE | vs baseline |
|---|---:|---:|
| `motion_tokens/advected_k16_s123` | 0.01839 | -0.6% |
| `factorial_2x2/s_topo_s123` | 0.01866 | +0.9% |
| `factorial_2x2/t_ssm_s123` | 0.01977 | +6.9% |
| `factorial_2x2/ts_ssm_s123` | 0.01983 | +7.2% |

The temporal × spatial 2×2 ablation is therefore answered in the negative: neither
the per-cell SSM temporal mixer nor the geodesic-distance spatial bias helps, and
combining them is the worst of the four. Motion-compensated tokens are the only
extension nominally in front, by 0.6% on one seed's validation split, and are the
only one worth a test-set evaluation if this line is ever reopened.

**Why it stopped.** Four independent probes agree that the temporal axis is
exhausted: the k=8 → k=16 window sweep saturates, the Δ-encoding ablation shows the
order scalar adds nothing, the temporal-mixer architectures land at parity or worse,
and the history-weighted loss degrades optimisation. The remaining blind-cell error
is not a time-encoding problem.

---

## End to end

```bash
cd path/to/this/repo && source sbatch/_env.sh   # the repo root; module load + PYTHONPATH
cd methods/senseiver            # python below runs from here; sbatch from the repo root

# 1. Self-check (GPU node, ~1 minute)
srun -p gpu-debug --gres=gpu:1 -t 00:14:00 --mem=16G bash -c \
  'module load scicomp-pytorch-env/2026.1; python3 -u checks/check_model.py; python3 -u checks/check_sensors.py'

# 2. Training (self-chains to 100 epochs)
(cd ../.. && sbatch methods/senseiver/sbatch/submit_train.sbatch)
#   -> runs/senseiver_A/{last.pt, best.pt, metrics.jsonl}

# 3. Evaluate on the 7 held-out days (convention exactly matches 4DVarNet / EnKF)
(cd ../.. && sbatch methods/senseiver/sbatch/submit_eval.sbatch --ckpt runs/senseiver_A/best.pt --tag _test)
#   -> check_outputs/eval/senseiver_metrics_test.json
```

**torch only runs on GPU nodes** (`sbatch`, or `srun -p gpu-debug --gres=gpu:1`),
never on the login node.

---

## Directory layout

| path | role |
|---|---|
| `positional.py` | the paper's `a = PE(chi)`: sin-cos positional encoding |
| `model.py` | the paper's `z = E(a_s,s)` and `s_hat_q = D(z,a_q)`: Encoder / Decoder; plus `GridEncoder` (variant G, an extension) |
| `sensors.py` | **new in this project**: observation -> variable-length sensor token set (padding + pad_mask) |
| `network.py` | the whole model (plain PyTorch, no Lightning) |
| `losses.py` | training loss (the reference implementation's unweighted MSE, copied as-is) + a diagnostic breakdown |
| `dataset.py` | one day -> samples; observation config read verbatim from `crowdcore/config.yaml` |
| `train.py` | training loop (Adam / AMP / `--resume`) |
| `checks/check_model.py` | permutation invariance, padding invariance, differentiability, dimension assertions |
| `checks/check_sensors.py` | invariants of the token construction |
| `checks/check_grid.py` | variant G's self-checks (sensor / cell permutation, A untouched) + per-step timing |
| `checks/trace_pipeline.py` | prints the shape at every step of the data pipeline; this README's Pipeline section is its output |
| `checks/evaluate.py` | held-out-day evaluation, metrics copied verbatim from `eval_test_days.py` |
| `sbatch/` | SLURM submission scripts (training self-chains to resume) |
| `runs/` `check_outputs/` | artifacts |

---

## Method: where Senseiver's three components land in the code

Paper sec.2:

```
a    = PE(chi)                     positional.PositionalEncoder
z    = E(PE(chi_s), s(chi_s,t))    model.Encoder      sensor set -> fixed-length latent
s_q  = D(z, PE(chi_q))             model.Decoder      latent + query coords -> field value
```

Structural detail from Appendix A, as implemented by the reference code: the
encoder block = cross-attention (learnable latents as Q, sensors as K/V) + a
self-attention block, with `num_layers` blocks **sharing weights** (`layer_1` is
independent, `layer_n` is reused `num_layers-1` times). Decoding = the query's
positional encoding concatenated with a learnable vector as Q, z as K/V, a single
cross-attention layer followed by a linear output head (`latent_size=1`).

### Why this method (the argument to make)

**Not because of scalability.** The paper's selling point is that decoding cost is
decoupled from the domain size, handling a 128x128x512 domain. ATC only has
36x12 = 432 cells, so that selling point is entirely wasted here -- a dense CNN
would be cheaper.

**Because the sensor set is variable-length, moving, and unordered.** The set and
count of cells the three robots see each second keeps changing (measured: **185-265
cells per frame** at k=1). Cross-attention is naturally permutation-invariant and
length-agnostic for such a set, whereas methods like Voronoi-CNN / dense
convolution are awkward when "sensor positions change every frame."
`checks/check_model.py` turns permutation invariance and padding invariance into
rerunnable assertions (measured max|delta| of 2.7e-7 and 0.0 respectively) -- this is
the premise the method's case rests on.

---

## Pipeline: one frame of data from the raw grid to the loss

Every shape below was actually produced by `checks/trace_pipeline.py`, not
inferred.

```
1  grid_cache, one day                    (T, 4, 36, 12)   float32
   methods/varnet's Stage-2 output, 1 frame per second
        |
        |  observation_model.generate_observations  <- crowdcore, config read from config.yaml
        v
2  Y      (T, 4, 36, 12)   noisy partial observation, 0 where unobserved
   Omega  (T, 36, 12) bool  which cells the robots saw this second (varies per
                            frame; 105-265 over the 64 traced frames, 202.8 average
                            over the full day)
        |
        |  dataset.load_day   flatten H x W -> HW, subsample frames by --stride
        v
3  X  (N, 4, 432)   target (ground truth)
   Y  (N, 4, 432)   observation
   Om (N, 432) bool
        |
        |  sensors.build_batch   <- the step added by this project
        v
4  tokens   (B, Nmax, 68)   Nmax = the largest sensor count in the batch;
                            68 = 4 channel values + 64-dim positional encoding
   pad_mask (B, Nmax) bool  True = a padding slot
        |
        |  model.Encoder   cross-attn (latents as Q, sensors as K/V) + self-attn,
        |                  weights shared across 3 layers
        v
5  z  (B, 64, 32)   <- fixed length! independent of the sensor count -- this is
                       the whole point of Senseiver
        |
        |  model.Decoder   query coordinates as Q, z as K/V
        |  coords (B, 432, 64)  <- positional encoding for the entire grid,
        |                          every frame queries all 432 cells
        v
6  out (B, 432, 4)  ->  reshape  ->  (B, 4, 36, 12)
        |
        |  losses.senseiver_loss = unweighted 4-channel MSE on the raw field
        |                          (copied from the reference implementation)
        v
7  compared against X. Blind = ~Omega broadcast to 4 channels (measured ~42%)
```

### Four key points

**Why 4 -> 5 is this method's entire value.** The sensor count per frame ranges
from 224 to 265, but z is always `(64, 32)`. A variable-length, unordered
observation set is compressed into a fixed-length representation --
`checks/check_model.py` turns this into an assertion (shuffled order max|delta| =
2.7e-7, padded to different lengths max|delta| = 0.0).

**6 queries all 432 cells, not just the 290 walkable ones.** Reason: see deviation
#1 -- robots can see cells they cannot drive into (measured 32.1%), and the
evaluation convention scores non-walkable cells too.

**Normalisation only happens at 4.** The first 4 dimensions of the tokens are
`(Y - mean) / std`, but both 6's output and 7's loss operate on the **raw field**.
Normalising the input side is feature conditioning; per-channel scaling of the
target would be equivalent to adding a weight to the loss, which would change the
quantity being optimised.

**The four channels are trained together, and the loss is dominated by vx.** The
four channels are summed unweighted (the paper's Eq., copied as-is), and vx's std
is 0.54 while var's is only 0.12, so vx dominates a single MSE. This is not a bug,
it is a property of this metric -- the header of
`compare/compare_channels.py` complains about the same thing. **Both
the single number and the per-channel breakdown must be reported** (`evaluate.py`
writes both into the JSON).

### Two easily-misread "1"s

| Name | What it is | What it is **not** |
|---|---|---|
| `latent_size = 1` | the number of learnable token vectors in the decoder, concatenated after each query point's positional encoding to form Q. The reference implementation's `s_parser.py` hardcodes it to 1, with the comment "collapse from n_sensors to 1 observation" | not the number of output channels. Setting it to 2 would just turn each query point into two rows of Query |
| `dec_num_cross_attention_heads = 1` | the number of attention heads in the decoder's cross-attention | likewise unrelated to the channel count |

**The number of output channels is decided by `im_ch`**, taken from
`config.yaml`'s `state_shape: [4,36,12]`, and is always 4. Can be checked directly
from the checkpoint: `decoder.postproc.weight` has shape `(4, 32)`,
`encoder.preproc.weight` has shape `(32, 68)`, `in_mean` has 4 elements.

---

## Deviations from the reference implementation (each with its reason)

**Required (copying as-is would be wrong)**

1. **Removed `pix_avail` ("a value of 0 means invalid"), query all 432 cells.**
   The reference implementation picks which pixels train with `data[0]!=0`
   (`dataloaders.py:152`) and at test time does `output_im[data==0]=0`
   (`network_light.py:126`). That hack exists to skip regions with **nothing to
   reconstruct** (land in sea-temperature data, solid material in porous media).
   No such cells exist on the ATC grid: density 0 is a valid value, non-walkable
   regions still have a ground truth, and **the evaluation convention scores
   them**. `check_sensors.py` measures **32.1% of observed cells falling on
   non-walkable ground** (robots can see pillars they cannot drive into, see the
   methods/varnet's README) -- further reason the query set cannot be clipped to
   walkable cells.
2. **Added dimension assertions to `Decoder`.** The decoder's cross-attention
   treats `dec_num_latent_channels` as the KV dimension, while the channel count
   of the incoming `z` is decided by the **encoder**; a mismatch between the two
   silently misaligns. The reference implementation has no such check -- its
   README's examples happen to always set them equal, hiding this trap.

**Added (the reference implementation has no such scenario)**

3. **Variable-length sensor sets: padding + `pad_mask`.** This path actually
   **already exists** in the reference implementation
   (`Encoder.forward(x, pad_mask)` -> `CrossAttention` -> `key_padding_mask`), it
   is just that its dataloader never passes a value, because its sensor set is
   fixed-length. We did not change the architecture, only used this path for the
   first time.
4. **A guard for empty sensor sets.** When `obs_every_k > 1`, some frames have no
   observation at all, and empty K/V in cross-attention makes softmax produce NaN.
   A single all-zero dummy token is inserted and marked valid; the model can only
   output a constant field for such a frame -- that is an informational fact, not
   an implementation flaw.
5. **Per-channel input normalisation.** The reference implementation's 5 datasets
   are **all single-channel** (`datasets.py`'s `sea`/`pipe`/`cylinder`/`plume`/`pore`
   all have a last dimension of 1), so it only does one global scalar division and
   gives no recipe for multiple channels. We have 4 channels whose scales differ by
   more than an order of magnitude. Approach: **normalise only the encoder input,
   leave the target and the loss on the raw field throughout**. Reason: the
   comparison convention is unweighted 4-channel MSE on the raw field, and any
   per-channel scaling of the target would be secretly adding a weight to the loss.

**Engineering**

6. **Removed `fairscale`'s `checkpoint_wrapper`.** `activation_checkpoint` is never
   exposed by `s_parser.py` in the reference implementation, is always False, and
   is dead code.
7. **No PyTorch-Lightning.** The reference implementation's `train.py` calls
   `Trainer()` **without passing `accelerator`/`devices`**; the `gpu_device`
   computed by `s_parser.py` is only used in the `--test` branch, so specifying a
   GPU index on the command line has no effect during training. We need to
   self-chain and resume on SLURM, so writing our own loop is simpler.
8. **`space_bands` 32 -> 16.** The frequencies are `linspace(1, dim/2, bands)`;
   with W=12 the highest frequency is only 6, so 32 bands is pure redundancy.
9. **`lr` defaults to 1e-3** (the reference argparse default is 1e-4). Its README's
   example with tens of thousands of frames (pipe) also uses 1e-3, and we are in
   the millions-of-frames range, the same regime.
10. **No pixel subsampling.** The reference implementation randomly queries only
    `batch_pixels` pixels per step, because its domain is too large to query in
    full. Querying all 432 cells at once is negligible cost, removing one source of
    randomness unrelated to the paper.
11. **`--stride` frame subsampling.** Adjacent seconds are highly redundant. This
    is not really a change -- the reference implementation likewise trains on
    `training_frames` frames randomly drawn from all frames.

**Explicitly not carried over**

12. **The reference implementation's sensor-count augmentation.**
    `dataloaders.py:168-171` only does this for the `pipe` dataset -- "randomly
    take `40+300|N(0,1)|` sensors out of 6144 per batch" -- and it is what trains
    the curve in the paper's Fig.2b, "any sensor count works at inference time";
    its README never mentions it. We do not need it: moving robots **naturally**
    give a different count every frame, so this augmentation is already built in.

---

## The fairness contract with the other two methods

All three methods face **exactly the same** observations; differences only come
from the method itself:

| Anchor | Approach |
|---|---|
| Observation parameters | read verbatim from `crowdcore/config.yaml`'s `observation` section; this directory sets no defaults of its own |
| Observation generation | calls `om.generate_observations` + `nav.build_valid_mask_from_config` directly |
| Data split | `om.split_files()`; 32 training days / 7 validation days / 7 test days, training days strictly earlier than test days |
| Metrics | blind MSE (`mask<0.5`) + full-field MSE, on the raw field, four channels unweighted, including non-walkable cells |
| Physical clipping | same as the EnKF: density[0,5], vx/vy[-5,5], var[0,2] (on by default) |
| Initial value | none initialised from the ground truth |
| Timing | only the model's forward pass is timed, not data preparation (a constant shared by all three methods) |

Benchmark (full-day convention, blind MSE): **4DVarNet 0.0338**, **EnKF 0.0392**.

### One sentence that must go in the conclusions

This version of Senseiver **reconstructs frame t from only frame t's observation**
(as in the paper, with no temporal encoding at all -- the paper's sec.3 Discussion
explicitly says a sin-cos time encoding was tried and failed). 4DVarNet, by
contrast, sees a dT=200-frame time window. This is **not a fair comparison, it is a
deliberate ablation**: it quantifies exactly "how much is the time dimension
worth." Without this sentence, the table would be misread.

---

## Results (7 held-out days, full day, obs_every_k=1, same clipping across all three)

> **Superseded scope.** This section predates 2026-09-05: it reports MSE over *all
> 432 cells* (obstacle cells included), a scope the project no longer reports. The
> authoritative numbers are pooled RMSE over walkable cells from
> `supervisor_evaluation/` (repository README, "Results"). Kept as a record;
> recompute in the walkable scope before citing any number from it.

> **Unit note.** This section reports **MSE**, under what the main tables now
> call the `allcells` convention; the final tables report
> **RMSE**, so the same quantity appears as 0.0284 here and 0.168 there
> (0.168^2 = 0.0282). Its 4DVarNet rows were produced by `eval_test_days.py`,
> which was removed on 2026-09-09 in favour of `compare/compare5.py` as the
> single implementation; the numbers below are kept as recorded at the time and
> differ slightly from compare5's for the same models.

Produced by `checks/evaluate.py` (single method) and, for the cross-method
per-channel table, `compare/compare5.py` (the three-way `compare3.py` it
originally came from was deleted on 2026-09-08 as a duplicate implementation).
4DVarNet's numbers are **rerun by us using its own
`eval_test_days.py`** (`--outdir` pointed at this directory, nothing written into
methods/varnet), matching its archived values to 4 decimal places, confirming the
archived convention also had clipping ON.

### Summary table

| Method | Blind MSE | Full-field MSE | Parameters | ms/frame |
|---|---|---|---|---|
| **Senseiver** | **0.0284 +/- 0.0034** | **0.0171** | **65,892** | **0.047** |
| 4DVarNet `a4_k1` | 0.0308 +/- 0.0036 | 0.0272 | 2,096,432 | 0.184 |
| EnKF `k1` | 0.0462 | -- | -- | -- |

Senseiver's blind MSE is lower than 4DVarNet-a4's on 7/7 days, difference mean +0.00237, std 0.00030 (far
smaller than the difference itself).

> **Historical warning, now resolved.** `enkf_metrics.json` used to hold
> **0.0392**, scored from a `check_outputs/enkf/` export that was
> **obs_every_k=4 with only 400 frames per day** — and the first 400 frames of a
> day are an empty field (density only 1/6.9 of the full day's), so it was a
> doubly favourable subset. That export directory was deleted on 2026-09-08 along
> with the whole k=4 line, `score_enkf.py` now defaults to the full-day k=1
> export, and the file has been re-scored at **0.0462**. The lesson stands: an
> EnKF number is meaningless without its `obs_every_k` and frame range.

### Per-channel blind MSE -- the ranking is **not** consistent

| Method | density | vx | vy | var | total |
|---|---|---|---|---|---|
| Senseiver | **0.0139** | 0.0788 | **0.0144** | **0.0064** | **0.0284** |
| 4DVarNet `a4_k1` | 0.0177 | **0.0753** | 0.0207 | 0.0111 | 0.0312 |
| EnKF `k1` | 0.0143 | 0.1103 | 0.0209 | 0.0394 | 0.0462 |

The figure for this table used **small multiples** rather than a grouped bar chart: the four
channels differ by an order of magnitude, and a shared y-axis would flatten
density/vy/var into invisible slivers, leaving the reader only able to see the
difference in vx -- which happens to be the one channel where 4DVarNet leads,
giving an impression opposite to the data. Each panel has its own y-axis; panels
are not comparable to each other.

**Three things that must be reported alongside the summary table:**

1. **4DVarNet's vx error is lower** (0.0753 vs 0.0788, a 4.4% difference), and vx accounts for
   **69%** of total blind error. Senseiver's overall lead comes **entirely from
   the other three channels** (density -21%, vy -30%, var -42%). In other words:
   for velocity along the corridor's main direction, variational assimilation's
   dynamical prior is still stronger.
2. **The EnKF nearly ties on density** (0.0143 vs 0.0139), despite trailing 46%
   overall -- it is dragged down by var (0.0394, 6x Senseiver's). The
   worst-overall method is competitive on the most physically meaningful channel.
3. Reporting only the total gives the false impression that "Senseiver is ahead across
   the board." **The per-channel breakdown must always be given alongside it.**

### Spatial decomposition of the error

Blind cells make up 53.1% (nearly constant day to day). Splitting full-field MSE
into blind / observed:

| Method | Blind | **Observed** | Full field |
|---|---|---|---|
| Senseiver | 0.0284 | **0.0043** | 0.0171 |
| 4DVarNet `a4_k1` | 0.0308 | **0.0231** | 0.0272 |
| Relative advantage | **+7.7%** | **+81.4%** | +37% |

**That 37% on the full field comes mostly from the observed cells, not from
guessing the unobserved region more accurately.** The decoder lets every query
point attend directly to that location's sensor token, so on observed cells it is
essentially reproducing the observation (error 0.0043, close in magnitude to the
observation noise itself); 4DVarNet's solution is pulled by the prior term
through 20 steps of learned gradient descent, which smooths away some of the
observed cells' accuracy.

Neither number is cheating -- both methods get the same observations, and
recovering the observed locations is genuinely part of the task. But **leading
with full-field MSE would significantly inflate the conclusion**; the
methods/varnet's README's choice to label blind MSE as *the real task* is justified.

### Ablation: how much do observations actually contribute

Erase the sensor **readings**, keep only their **positions** and the pad_mask
(`checks/evaluate.py --ablate-values`):

| | Blind MSE | Full-field MSE |
|---|---|---|
| normal | 0.0284 | 0.0171 |
| positions only | 0.0343 (**+21%**) | 0.0515 (+201%) |

The model is genuinely using the observations, but **a substantial share of the
blind-cell reconstruction comes from a learned corridor climatology, not from the
current observations**. And 21% is an **upper bound** -- erasing the readings feeds
the model constants it has never seen (an out-of-distribution input), and a model
trained specifically as a pure climatology would only do better.

So the correct statement is: **at this coverage (47%) and this sparsity, a direct,
amortised regressor is more effective than iterative variational assimilation**,
not "attention mechanisms make especially good use of sparse observations."

### One setup difference that favours this method

Senseiver reconstructs frame t using only frame t's observation; 4DVarNet, when
reconstructing frame t, can use observations from the entire dT=200 window.
**4DVarNet strictly receives more information**, a setup unfavourable to
Senseiver, and it still leads overall. This makes the overall-score conclusion
stronger, but does not change point 1 above (its vx error is higher).

---

## Variant G (grid latent) -- an extension, not the reference implementation

**Status (2026-09-14): done.** 3 configurations x 3 seeds x 40 epochs, training jobs
20250271-20250279, outputs in `runs/grid/<config>_s<seed>/`; evaluated with
`compare.compare5 --arms '' --no-with-ensemble`, whose Senseiver row reproduces the
published numbers exactly. **Verdict: G-direct beats A40 clearly; G-dec only
marginally** -- see "Result" below.

### The question

Our state is 4 x 36 x 12 = 1,728 numbers per frame; variant A's latent array is
64 x 32 = 2,048 numbers -- **larger than the state it is supposed to compress**.
Methods that work in a latent space (e.g. latent 4DVar on ERA5, 69x256x128 = 2.26 M
numbers compressed to 69,632) are forced there by dimension; we are not. In variant
A the latent's only real job is to turn a variable-length, unordered sensor set
into something fixed-length -- and the grid itself is fixed-length.

So: **does replacing the 64 abstract latents with one latent token per grid cell
change the reconstruction error?** No time is added in this experiment.

Where variant A stands, in the reported scope (blind walkable cells, pooled RMSE,
`compare5_final.json` -> `walkable`): **first**, 0.2222, ahead of DINCAE 0.2267 and
4DVarNet's reported single model (MSE s3) 0.2301. Per channel it is best on vy and
var, second on density (0.0242 vs DINCAE 0.0233, MSE), and behind 4DVarNet s3 on vx
by 6.8% (0.1380 vs 0.1292 MSE) -- a gap the size of 4DVarNet's own seed spread on
that number (0.1292-0.1377 across its five seeds). vx is 70% of A's blind-walkable
error, so it is the channel to watch, but the case for G does not rest on it.

> **Correction, 2026-09-14, before any result existed.** This section first
> motivated G with "Senseiver is 4th and 20% behind on vx". Those numbers came from
> the `defined` cell set (velocity scored only where density > 0), which
> `compare5.py` computes but the project does not report (repo README, "Scoring
> scope"). The decision rule below was moved to the reported scope while all nine
> jobs were still queued.

### Configurations

| name | latent | readout | parameters | purpose |
|---|---|---|---|---|
| A40 (control) | 64 abstract vectors | query decoder | 65,892 | variant A retrained at the same 40-epoch budget |
| G-dec | 432 cell tokens | query decoder | 65,924 (+0.05%) | the latent form is the only change |
| G-direct | 432 cell tokens | `Linear(32->4)` per cell | 56,260 (-15%) | "the grid is the state"; the base a temporal extension would build on |

`train.py --latent-mode grid [--readout direct]` selects G. The defaults give
variant A, and old checkpoints (whose hparams lack these keys) load as A.

Cell tokens start from **the cell's positional encoding through one linear map**,
not a free vector per cell. A free per-cell vector (432 x 32 parameters) could
memorise that cell's mean field, and the sensor-value ablation above already shows
climatology carries most of the blind-zone skill -- that would confound "grid
structure helps" with "a per-cell bias helps".

### Checks before training (`checks/check_grid.py`, all pass)

| check | G-dec | G-direct |
|---|---|---|
| sensor permutation invariance, max\|Δ\| | 3.0e-7 | 1.4e-6 |
| padding invariance | 0.0 | 0.0 |
| cell permutation (G-dec invariant / G-direct equivariant) | 3.3e-7 | 1.7e-6 |
| cells distinguishable (guards a trivially-passing equivariance) | -- | outputs differ by 2.0 |

**Variant A is untouched**: the A checkpoint reproduces, bit for bit, an output
saved on a fixed input *before* any code changed (max|Δ| = 0), `git diff` shows no
deleted lines in `model.py`, and `check_model.py` still passes.

Speed. The micro-benchmark in `check_grid.py` (V100, B=64, AMP, forward+backward
only) gave A 41.7 / G-dec 44.4 / G-direct 40.7 ms per step and peak memory rising
from 0.20 to 2.1 GB. **Real training epochs differ from that, and by hardware**:

| hardware | A40 | G-dec | G-direct |
|---|---|---|---|
| A100, s/epoch | 118-119 | 143-145 (1.2x) | 132-134 (1.1x) |
| V100, s/epoch | 137 | 238 (1.7x) | -- |

On A100 the benchmark's "about the same" roughly holds; on V100 G-dec is much
slower than it predicted. Do not budget G runs from the micro-benchmark.

### Decision rule (fixed before seeing results)

Primary metric: **blind-walkable pooled RMSE** (the reported scope), 7 held-out
days, against A40 -- variant A retrained at the same budget, not the published
100-epoch A. G beats A only if all three hold:

1. its mean over the 3 seeds is lower;
2. G's worst seed beats A40's best seed -- 4DVarNet's five seeds span 2.3% in this
   RMSE (0.2301-0.2355) and 6.6% in its vx MSE, so a single-seed difference means
   nothing until Senseiver's own spread is measured, which A40 x 3 does;
3. no channel's 3-seed mean (blind-walkable MSE) degrades by more than 5%.

Also reported, not deciding: all-walkable RMSE, the per-channel breakdown, and the
`defined` cell set as a diagnostic of the cells where people are moving. If the seed
ranges overlap, abstract and grid latents are equivalent at this scale and a
temporal extension builds on variant A.

Entering the accuracy table would follow the repo rule: one model per method, the
seed with the lowest validation blind-walkable error.

### Result

Blind-walkable pooled RMSE on the 7 held-out test days, each run's `best.pt`
(`check_outputs/grid/eval/summary.json`, written by `checks/summarize_grid.py`):

| config | s123 | s124 | s125 | mean | seed spread | all-walkable mean |
|---|---|---|---|---|---|---|
| A40 | 0.2229 | 0.2233 | 0.2231 | 0.2231 | 0.18% | 0.1580 |
| G-dec | 0.2225 | 0.2202 | 0.2227 | 0.2218 (-0.56%) | 1.12% | 0.1537 |
| G-direct | 0.2193 | 0.2197 | 0.2196 | **0.2195 (-1.58%)** | 0.20% | **0.1510** |

Per channel, 3-seed mean blind-walkable MSE:

| config | density | vx | vy | var |
|---|---|---|---|---|
| A40 | 0.0246 | 0.1390 | 0.0240 | 0.0115 |
| G-dec | 0.0241 (-1.8%) | 0.1374 (-1.1%) | 0.0238 (-1.0%) | 0.0115 (-0.2%) |
| G-direct | 0.0235 (-4.5%) | 0.1347 (-3.0%) | 0.0233 (-2.7%) | 0.0113 (-1.9%) |

**Under the rule both G variants beat A40. They are not equally convincing.**

- **G-direct beats A40 decisively.** Every seed improves by the same amount
  (paired -0.0036 / -0.0035 / -0.0035); its worst seed is 0.0032 below A40's best,
  8x A40's whole seed range; every channel improves; it has 15% fewer parameters.
- **G-dec passes by a hair.** Criterion 2 holds with a 0.0002 margin (0.2227 vs
  0.2229). Two of its three seeds are within 0.0004 of A40, and almost all of its
  mean gain comes from s124 alone (-0.0030). Read it as "about the same as A40, one
  good seed", not as a finding.

So the gain comes from **reading each cell's value off its own token** (G-direct vs
G-dec), much more than from swapping the abstract latent for cell tokens while
keeping the query decoder (G-dec vs A40).

Also measured:

- **Senseiver's own seed variance is tiny**: 0.18% (A40) and 0.20% (G-direct) in
  blind-walkable RMSE, against 2.3% across 4DVarNet's five MSE seeds. The caution
  borrowed from 4DVarNet ("single-seed differences under ~10% mean nothing") was
  far too loose for this method.
- **Epoch budget**: A40_s123 at 40 epochs is 0.2229, the published 100-epoch variant
  A with the same seed is 0.2222 -- 60 more epochs buy 0.3%. G-direct at 40 epochs
  is below the published A regardless.
- **vx**: G-direct narrows the gap to 4DVarNet's reported single model (MSE s3,
  0.1292) from 7.6% (A40) to 4.3%; it does not close it. On density G-direct
  (0.0235) is now level with DINCAE (0.0233).
- The `defined` diagnostic moves the same way (vx 0.6346 -> 0.6143).

What this does **not** license yet:

- These are **test-set** numbers. The accuracy table picks one model per method by
  **validation** error, and `best.pt` here was chosen by `train.py`'s own validation
  metric (blind MSE over all 432 cells), not blind-walkable. Picking G-direct's seed
  from the table above would be selecting on the test set. Its three seeds are
  within 0.0004, so the choice barely moves the number, but it must be made on
  validation.
- G-direct is an extension. It can enter the comparison only as a separately named
  row next to the faithful Senseiver, not in place of it.

### What this is and is not

G changes the architecture, so it is **not** a faithful reproduction. The claim "the
network matches the reference implementation parameter tensor for parameter
tensor" belongs to variant A only; G is reported as a separately named extension.

Seeds are 123 / 124 / 125. Seed 123 is variant A's original seed, so A40_s123 vs
variant A isolates the epoch budget alone.

---

## Temporal extension (built on G-direct) -- an extension, not the reference implementation

**Status (2026-09-15): sweep done -- every window beats k=1, best k = 8 (-9.0%); the controls confirm the model uses each observation's time offset; and the Δ/k scalar turns out unnecessary.** **k = 16 added (2026-09-15), training.** Self-checks pass (`checks/check_temporal.py`: token-order
invariance, a changed Δ changes the output, k=1 bit-identical to G-direct and to variant A,
`TemporalDayBank` == `DayBank` on real data). Jobs 20258328-20258336 (time limits k=2 3 h,
k=4 4 h, k=8 6 h; H200/A100 only). The decision rule below was written before any temporal
model was trained.

### Design

Built on G-direct, the variant that beat A40. The 432 cell tokens' cross-attention
now reads the sensor tokens of the **last k frames** instead of the current frame
only. Each sensor token carries its relative time offset Δ = t_query - t_token
(`positional.TemporalEncoding`): a learnable `nn.Embedding(k, 8)` plus the scalar
Δ/k, concatenated before the encoder's preproc. So every cell can read what was seen
at its own and at nearby positions over the last k-1 seconds.

- **No sine-cosine encoding of time.** The paper (sec.3) reports it failed; dropped on
  2026-09-13 and not tested.
- **Relative, not absolute, time.** Only k values, each trained millions of times; no
  clock, so no falling back on time-of-day climatology.
- Windows never cross a day; near the start of a day they are shorter.
- `TemporalDayBank` picks the same target frames with the same observations as
  `DayBank` for a given seed, so a k-run is paired with the G-direct run of that seed.
- `compare/compare5.py`'s Senseiver inference gained a branch taken only by models with
  `time_window > 1`. **Verified after the change** (jobs 20258254/55): variant A still
  scores 0.222195 / 0.156433 and G-direct s123 0.219305 / 0.150740 (blind / all
  walkable), max|Δ MSE| = 0 on both.

Files: `positional.TemporalEncoding`, `sensors.build_batch_temporal`,
`dataset.TemporalDayBank`, `network.py` (`time_window`, `time_dim`, `time_scalar`;
default k=1 adds nothing), `train.py --time-window k`, `checks/check_temporal.py`,
`checks/measure_temporal_prior.py` (stage 0).

### Stage 0: is there anything in the past to use? (`checks/measure_temporal_prior.py`)

First 3 training days, training seed 123, blind walkable cell-frames only:

| window k | 2 | 4 | 8 | 16 |
|---|---|---|---|---|
| blind cell observed somewhere inside the window | 9.9% | 25.0% | 45.4% | 68.7% |

| lag (frames) | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| vx autocorrelation, cells occupied at both times | 0.837 | 0.564 | 0.293 | 0.177 | 0.121 |
| vx autocorrelation, all walkable cells | 0.605 | 0.301 | 0.129 | 0.076 | 0.052 |

Median time since a blind cell was last seen: 9 frames (P25 4, P75 19). **Gate: borderline**
-- recall@4 (25%) misses the 30% go line and lag-4 vx correlation (0.29) misses 0.5, while
recall@8 (45%) is far from the 15% stop line. The two numbers pull against each other: a
longer window reaches more blind cells, but what it reaches is older, and vx at a cell
decorrelates within about 2 frames. This measures same-cell persistence only; it cannot see
whether a model reads upstream cells' history to infer where people moved. The plan had no
rule for "borderline"; the sweep was run anyway. k = 16 was left out at first (about 6 h per
run for little expected information) and added later, once k = 2-8 showed no levelling off --
see Runs.

### Runs

k = 2, 4, 8 x seeds 123 / 124 / 125, 40 epochs, G-direct otherwise unchanged. **k = 1 is
the existing G-direct x 3** (same seeds, same budget; its architecture differs only by
the absent, constant time channels).

**k = 16 was added afterwards** (3 seeds, 9 h time limit, H200/A100), because the gain had not
levelled off by k = 8: each doubling of the window had bought about another 3%. This extension
was decided after seeing the k = 2-8 results; the decision rule is unchanged -- k = 16 is judged
against k = 1 exactly like the other windows, and the best k is re-picked among every window
that passes. Real A100 epochs so far: k=2 160 s, k=4 220 s, k=8 312 s, projecting roughly
460-500 s at k = 16.

### Decision rule (fixed before any temporal result)

Primary metric: blind-walkable pooled RMSE, 7 held-out test days, against G-direct
(k=1). A window k beats k=1 only if all three hold:

1. its mean over the 3 seeds is lower;
2. its worst seed beats G-direct's best seed (0.2193);
3. no channel's 3-seed mean blind-walkable MSE degrades by more than 5%.

vx is the channel to watch: it is the dynamic quantity a window should help most, and
where 4DVarNet, which assimilates a 200-frame window, still leads. The best k is the
lowest blind-walkable mean among those that pass; if none passes, the conclusion is
that a window of past observations does not help this model.

### Result of the sweep

Blind-walkable pooled RMSE, 7 held-out test days, each run's `best.pt`
(`check_outputs/temporal/eval/summary.json`, from `checks/summarize_temporal.py`):

| k | s123 | s124 | s125 | mean | vs k=1 | seed spread | all-walkable mean |
|---|---|---|---|---|---|---|---|
| 1 (G-direct) | 0.2193 | 0.2197 | 0.2196 | 0.2195 | -- | 0.20% | 0.1510 |
| 2 | 0.2123 | 0.2123 | 0.2125 | 0.2124 | -3.3% | 0.12% | 0.1458 |
| 4 | 0.2061 | 0.2058 | 0.2057 | 0.2059 | -6.2% | 0.20% | 0.1421 |
| **8** | 0.1996 | 0.1998 | 0.2001 | **0.1998** | **-9.0%** | 0.26% | **0.1383** |

Per channel, 3-seed mean blind-walkable MSE (change vs k=1):

| k | density | vx | vy | var |
|---|---|---|---|---|
| 1 | 0.0235 | 0.1347 | 0.0233 | 0.0113 |
| 2 | 0.0216 (-8.0%) | 0.1253 (-7.0%) | 0.0225 (-3.6%) | 0.0110 (-2.4%) |
| 4 | 0.0202 (-14.1%) | 0.1165 (-13.5%) | 0.0219 (-6.0%) | 0.0109 (-3.3%) |
| 8 | 0.0185 (-21.0%) | 0.1091 (-19.0%) | 0.0213 (-8.6%) | 0.0107 (-4.8%) |

**Every window passes all three criteria; best k = 8.** Paired per-seed gains are nearly
identical within each k (k=8: -0.0197 / -0.0200 / -0.0195), and every seed of every window
beats G-direct's best seed by a wide margin.

What stands out:

- **The gain grows steadily with k and has not levelled off at 8.** Each doubling buys roughly
  another 3% (-3.3 / -6.2 / -9.0). k = 8 is the largest window tested, so "best k = 8" is the
  edge of the sweep, not an optimum.
- **Stage 0 was too pessimistic.** It measured whether the *same cell* was seen recently and
  how fast that cell's vx decorrelates (0.29 at lag 4), and came out borderline. The model gains
  most exactly where that measure said the information runs out, so it must use more than
  same-cell persistence -- plausibly where people upstream were a few seconds earlier.
- **vx, the channel where Senseiver trailed, is now its largest win.** k = 8 reaches 0.1091,
  15.6% below 4DVarNet's reported single model (0.1292), where G-direct was 4.3% above it.
  Density improves most (-21%).

Not yet licensed:

- Whether the model uses **when** each observation was made, rather than simply seeing more
  observations -- that is what the controls below test.
- These are test-set numbers of an extension: before any table entry, select on validation and
  report as a separately named row, exactly as for G.

### Time-usage controls (k = 8, `checks/eval_temporal_controls.py`)

Same checkpoints and days; only the Δ given at inference changes. Blind-walkable pooled RMSE:

| seed | normal | Δ shuffled | Δ all 0 |
|---|---|---|---|
| 123 | 0.1996 | 0.2541 (+27.3%) | 0.2631 (+31.9%) |
| 124 | 0.1998 | 0.2532 (+26.8%) | 0.2533 (+26.8%) |
| 125 | 0.2001 | 0.2559 (+27.9%) | 0.2632 (+31.6%) |
| **mean** | **0.1998** | **0.2544 (+27.3%)** | **0.2599 (+30.1%)** |

Per channel, 3-seed mean blind-walkable MSE: vx 0.1091 -> 0.1858 (shuffled) / 0.1948 (zero);
density 0.0185 -> 0.0286 / 0.0289.

**The model uses when each observation was made.** If it only profited from seeing more
tokens, scrambling or erasing Δ would leave the score unchanged; instead both make it far
worse -- worse even than having no window at all (k = 1, 0.2195, is 9.9% above normal). Read
the +27-30% as how badly the model is misled when stale observations are presented as current,
not as the value of the timing information; that value is the sweep's -9.0%. A model trained
on k frames but never given Δ would measure "more tokens without timing" directly; it was not
run.

Scoring check: `normal` reproduces compare5's evaluation of the same checkpoint exactly for s124
and s125, and to max|Δ MSE| = 8.9e-10 for s123 (identical to four decimals). The s123 pair ran
on different GPU types -- compare5 on a V100, the control on an H200 -- while both s124 and s125
pairs ran on H200s. So cross-GPU-type bit-reproducibility, which held for the k = 1 models, does
not hold at k = 8's token counts; within one GPU type it does.

### Is the order scalar needed? (k = 8, `--no-time-scalar`)

Same seeds and budget, only the Δ/k scalar removed. The models then have 56,580 parameters
instead of 56,612 (32 fewer preproc weights), confirmed in each training log, so the flag took
effect. Blind-walkable pooled RMSE:

| seed | with Δ/k | embedding only | diff |
|---|---|---|---|
| 123 | 0.1996 | 0.2007 | +0.0011 |
| 124 | 0.1998 | 0.2001 | +0.0004 |
| 125 | 0.2001 | 0.1994 | -0.0007 |
| **mean** | **0.1998** | **0.2001** | **+0.0003 (+0.13%)** |

Per channel the difference stays within 0.7%. The paired differences change sign and the seed
ranges overlap (0.1996-0.2001 vs 0.1994-0.2007), so **the scalar is not needed**: the learnable
embedding learns the order of the offsets by itself.

### After the sweep

- **Is time actually used?** Evaluation-only controls on the best k: shuffle Δ across
  tokens, and set every Δ to 0. If neither hurts, any gain came from seeing more
  tokens, not from knowing when they were seen.
- **Is the order scalar needed?** Retrain the best k without Δ/k (`--no-time-scalar`).

### One sentence that must go in the conclusions

With a window, Senseiver no longer sees one frame only: the earlier caveat "4DVarNet
sees a 200-frame window while Senseiver sees one frame" stops applying. The fair
statement becomes: both assimilate a time window (4DVarNet's is still 200/k times
longer), and the difference is in how they assimilate it.

---

## Measured facts about the data

| Quantity | Value |
|---|---|
| Grid | 36x12 = 432 cells, 4 channels; 290 walkable cells |
| Observation coverage (k=1) | 185-265 cells per frame, ~44-47% |
| Observed cells falling on non-walkable ground | 32.1% |
| Channel scale (within walkable, std) | density 0.17 / vx 0.46 / vy 0.16 / var 0.11 |
| Frame count | ~40k frames/day x 32 training days ~= 1.28M frames |
| Model parameter count | 65,892 (default hyperparameters) |
| Data preparation | 10.3 s/day (simulating a full day's observations), 144 MB/day (stride=4) |

44% coverage means **this is not the Senseiver paper's sparse regime** (NOAA is
10-300 sensors / 64800 cells, < 1%). Do not cite its sparsity selling point in the
report.

---

## Gotchas

- **Imports go through package paths, never `sys.path`**: since the 2026-09 refactor
  nothing in this directory touches `sys.path`. `sbatch/_env.sh` sets
  `PYTHONPATH=<repo root>` and `PYTHONSAFEPATH=1`, and everything runs as
  `python3 -m methods.senseiver.<module>` from the repo root. Several methods still
  ship their own `losses.py` / `dataset.py` / `model.py`, so a bare `import losses`
  fails on purpose -- always import `methods.senseiver.losses`.
- **`/tmp` is node-local**: what a compute node writes to `/tmp` is invisible to
  the login node. Artifacts always go to `runs/` and `check_outputs/`.
- **The first 400 frames are an empty field**: evaluation numbers from
  `--frames 400` are noticeably over-optimistic, fine for a smoke test, not to be
  treated as a conclusion.
- **The EnKF's 0.0392 must not be cited**:
  `methods/varnet/check_outputs/eval/enkf_metrics.json` comes from
  `check_outputs/enkf/`, a doubly-favourable subset with obs_every_k=**4** and only
  **400** frames per day. The full-day k=1 estimate is in
  `check_outputs/enkf_k1_full/`, re-scored at **0.0462**. Likewise
  `test_metrics_matched_clip.json`'s 0.0258 is also under the 400-frame convention.
- **Single-day conclusions do not extrapolate**: per-channel rankings differ
  between a single day and seven days (we hit this: on one day the EnKF looks best
  on density, over seven days Senseiver is slightly ahead). Any per-channel claim
  must use the full 7 days.
- `observation_model.generate_observations`'s docstring says "only observe
  walkable cells," which does not match this project's README or the actual data
  (measured 32.1% falling outside). Trust the data.
- **`sbatch/submit_train.sbatch` resumes into variant A by default**: `OUT` defaults
  to `runs/senseiver_A`, and the script adds `--resume` whenever `$OUT/last.pt`
  exists. A new run whose `OUT` does not reach the job would continue training --
  and overwrite -- the official variant A checkpoint that compare5 reads. Set `OUT`
  to an absolute path **and** pass `--out` among the script arguments (argparse
  keeps the last one). A copy of variant A is in `runs/senseiver_A_backup/`
  (best.pt md5 `ad0f002e...`).
- **`--seed` is not only the initialisation**: it also seeds the robot paths that
  generate the training and validation observations (`DayBank(seed=...)`). Compare
  variants at matched seeds. Evaluation observations are fixed (seed 0) regardless.
- **A smoke test shorter than ~20 steps with `--amp` looks like "no learning"**: the loss is
  `reduction='sum'`, so GradScaler's initial scale (65536) overflows fp16 and it skips
  and halves for the first ~18-19 steps (measured: 19/40 for k=1, 18/40 for k=4; identical
  validation numbers across the smoke's two 7-step epochs). Harmless in real training --
variants A and G carry the same ~19 skipped steps out of ~200k -- but to check that a
change learns, run without `--amp` or for more steps.

---

## Temporal × spatial 2×2 ablation (2026-09-16)

The capacity, data, seed, optimiser, k=16 window, grid latent and direct readout
are fixed.  Only the two named factors change:

| run | temporal mixer | spatial cross-attention |
|---|---|---|
| Baseline | flattened temporal token set + relative-time encoding | global |
| T-SSM | per-cell diagonal SSM, oldest to newest | global |
| S-Topo | baseline temporal token set | max-normalised geodesic-distance bias |
| TS-SSM | per-cell diagonal SSM, oldest to newest | geodesic-distance bias |

All four models have exactly 56,676 trainable parameters.  The SSM collapses a
cell's observed history to one final state token before spatial attention.  The
geodesic distance is the 8-connected shortest path over the existing static
walkable mask; disconnected/non-walkable endpoints receive the maximum normalised
distance.  The bias is soft (`-scale * distance`, fixed `scale=1`) rather than a
hard mask, so no query loses all keys.

Training convention: 60 epochs, batch 64, Adam 1e-3, AMP, seed 123, three-hour
H200 jobs with checkpoint resume/self-chaining.  Baseline is the existing
`runs/capacity/base32_k16_s123/best.pt`; the three new cells live under
`runs/factorial_2x2/`.

### Motion-compensated temporal-token diagnostic

`--temporal-mixer advected_tokens` retains the baseline's flattened k=16 token
set, relative-time encoding, global attention and exact 56,676-parameter count.
Its only change is to encode each historical token at a reliability-decayed
velocity extrapolation, `(row, col) + dt * exp(-dt/tau) * (vx, vy)`, clipped to
the grid.  `tau=3 s` was selected once on the first validation day from the
predeclared set `{1,2,3,4,6,8}`: averaged over horizons 1--15, this simple
transport reduced clean-density MSE by 9.8% versus persistence, while un-decayed
constant velocity was 74.6% worse.  Current-frame tokens (`dt=0`) are bitwise
the original spatial encoding.
This isolates whether temporal blur comes from attaching old observations to
where pedestrians were rather than where their measured velocity suggests they
have moved.  It is a diagnostic, not yet a topology-aware transport model: the
straight extrapolation can cross walls, which a later graph-transport version
would address if this experiment is positive.

### Observation-trajectory diversity diagnostic

The original loader restarts the robot simulator with the same seed for every
day. Because the map is static, robot positions and observation masks are
identical at equal frame indices across all 32 training days.
`--trajectory-mode per_day` instead uses `obs_seed + day_index`: the experiment
remains exactly reproducible while every day gets a different path.

Model initialisation and observations now have separate controls, `--seed` and
`--obs-seed`. Validation can use an explicitly disjoint seed range through
`--valid-obs-seed`; evaluation supports the same `--obs-seed` and
`--trajectory-mode` controls. The first paired run keeps the k=16 G-direct
baseline architecture and budget unchanged:

| run | training paths | validation paths |
|---|---|---|
| existing fixed-path baseline | seed 123 on every day | seed 123 on every day |
| diverse-path | seeds 123..154 | seeds 10000..10002 |

Final comparison evaluates both checkpoints on the same collection of held-out
trajectory seeds. Their training-time validation losses are not directly
comparable because the validation protocols differ.

### History-use refinement (2026-09-17)

Before changing the temporal architecture, `checks/diagnose_history_age.py`
measured blind-walkable test error by the time since each cell was last
observed. It compares matched k=1/40-epoch, k=16/40-epoch and the current
k=16/60-epoch baseline checkpoints on all seven test days. Results are pooled
over cell-frames in
`check_outputs/temporal/history_age_k1_k16.json`:

| last observed | share of blind cells | k=1 RMSE | k=16/60 RMSE | change |
|---|---:|---:|---:|---:|
| 1 s | 9.79% | 0.16636 | 0.13296 | -20.07% |
| 2--4 s | 20.81% | 0.21347 | 0.17819 | -16.53% |
| 5--8 s | 18.31% | 0.22906 | 0.20087 | -12.30% |
| 9--15 s | 19.50% | 0.23304 | 0.21271 | -8.72% |
| 16+ s | 31.54% | 0.22435 | 0.21216 | -5.43% |
| never seen | 0.05% | 0.14250 | 0.14019 | -1.62% |

The cells directly reachable through the k=16 history (ages 1--15) are 68.42%
of blind cell-frames and improve by 12.94%, versus 5.43% outside the window.
This licenses a targeted objective without introducing a motion assumption:
`--history-loss-weight w` gives cells that are blind now but observed earlier
in the input window weight `1+w`; `w=0` is exactly the baseline loss. The first
controlled run uses `w=0.5`, otherwise the exact k=16 baseline configuration
(job 20310546, `runs/history_loss/w05_k16_s123`). In its first scheduler segment
the ordinary training MSE and weighted `objective` are logged separately; after
the first resume the legacy `train_mse` field contains that weighted objective.
This logging-only difference does not affect optimisation. Validation metrics
remain the unchanged unweighted diagnostics used for selection and comparison.

The next architectural stage is implemented but gated on the weighted-loss
result: `--temporal-mixer framewise` applies one shared spatial cross-attention
to each sparse frame, performs learned temporal attention over the resulting
dense per-cell latent sequence, and adds the historical result to the current
latent through a gate initialised near zero. It never changes a token's original
coordinate. The full k=16 model has 56,804 parameters versus the baseline's
56,676 (+128, 0.23%). Token-permutation invariance, padding invariance, history
sensitivity, complete gradient flow and a GPU forward/backward smoke test pass
(job 20313381). The parameter-matched full run was then submitted in parallel
with the simpler objective (job 20313743, `runs/framewise/k16_s123`); it keeps
the original unweighted MSE, fixed paths and every other baseline setting.
Before the H200 continuation, its frame cross-attention was made computationally
compact: each call now receives only that frame's valid tokens instead of all
16 frames followed by a mask. This changes neither parameters nor the attention
softmax. A float64 reference comparison against the original masked computation
matched outputs to `8.9e-16`, input gradients to `7.1e-15`, and parameter
gradients to `2.9e-14`, so the epoch-0 checkpoint remains compatible.
