# senseiver_crowd — reproducing Senseiver on the ATC crowd field

**The third method route.** The first two (4DVarNet reproduction + Localized EnKF
comparison) live in [`../4dvarnet_enkf/`](../4dvarnet_enkf/). This directory only
reuses its **data pipeline and observation configuration** (`config.yaml` /
`observation_model.py` / `navigation.py`), does not depend on any of its
conclusions, and does not modify any of its lines.

Paper: Santos et al., *The Senseiver: attention-based global field reconstruction
from sparse observations* (NeurIPS ML4PS 2022 workshop; published version Nature MI
2023). Reference implementation:
[`../../reference/Senseiver/`](../../reference/Senseiver/) (the official PyTorch
code).

**The goal is to reproduce the paper's method**, not to improve on it first. Every
place that deviates from the reference implementation is listed below with its
reason.

---

## End to end

```bash
module load scicomp-pytorch-env/2026.1
cd /scratch/work/zhangx29/Thesis_Project/senseiver_crowd

# 1. Self-check (GPU node, ~1 minute)
srun -p gpu-debug --gres=gpu:1 -t 00:14:00 --mem=16G bash -c \
  'module load scicomp-pytorch-env/2026.1; python3 -u checks/check_model.py; python3 -u checks/check_sensors.py'

# 2. Training (self-chains to 100 epochs)
sbatch sbatch/submit_train.sbatch
#   -> runs/senseiver_A/{last.pt, best.pt, metrics.jsonl}

# 3. Evaluate on the 7 held-out days (convention exactly matches 4DVarNet / EnKF)
sbatch sbatch/submit_eval.sbatch --ckpt runs/senseiver_A/best.pt --tag _test
#   -> check_outputs/eval/senseiver_metrics_test.json
```

**torch only runs on GPU nodes** (`sbatch`, or `srun -p gpu-debug --gres=gpu:1`),
never on the login node.

---

## Directory layout

| path | role |
|---|---|
| `positional.py` | the paper's `a = PE(chi)`: sin-cos positional encoding |
| `model.py` | the paper's `z = E(a_s,s)` and `s_hat_q = D(z,a_q)`: Encoder / Decoder |
| `sensors.py` | **new in this project**: observation -> variable-length sensor token set (padding + pad_mask) |
| `network.py` | the whole model (plain PyTorch, no Lightning) |
| `losses.py` | training loss (the reference implementation's unweighted MSE, copied as-is) + a diagnostic breakdown |
| `dataset.py` | one day -> samples; observation config read verbatim from `4dvarnet_enkf/config.yaml` |
| `train.py` | training loop (Adam / AMP / `--resume`) |
| `checks/check_model.py` | permutation invariance, padding invariance, differentiability, dimension assertions |
| `checks/check_sensors.py` | invariants of the token construction |
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
   4dvarnet_enkf's Stage-2 output, 1 frame per second
        |
        |  observation_model.generate_observations  <- 4dvarnet_enkf, config read from config.yaml
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
`4dvarnet_enkf/checks/compare_channels.py` complains about the same thing. **Both
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
   4dvarnet_enkf README) -- further reason the query set cannot be clipped to
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
| Observation parameters | read verbatim from `4dvarnet_enkf/config.yaml`'s `observation` section; this directory sets no defaults of its own |
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

Produced by `checks/evaluate.py` (single method) and, for the cross-method
per-channel table, `compare/compare5.py` (the three-way `compare3.py` it
originally came from was deleted on 2026-09-08 as a duplicate implementation).
4DVarNet's numbers are **rerun by us using its own
`eval_test_days.py`** (`--outdir` pointed at this directory, nothing written into
4dvarnet_enkf), matching its archived values to 4 decimal places, confirming the
archived convention also had clipping ON.

### Summary table

| Method | Blind MSE | Full-field MSE | Parameters | ms/frame |
|---|---|---|---|---|
| **Senseiver** | **0.0284 +/- 0.0034** | **0.0171** | **65,892** | **0.047** |
| 4DVarNet `a4_k1` | 0.0308 +/- 0.0036 | 0.0272 | 2,096,432 | 0.184 |
| 4DVarNet `b0_k1` | 0.0338 +/- 0.0040 | 0.0290 | 2,051,596 | 0.086 |
| EnKF `k1` | 0.0462 | -- | -- | -- |

Senseiver beats 4DVarNet-a4 on 7/7 days, difference mean +0.00237, std 0.00030 (far
smaller than the difference itself).

> **Convention warning**: the **0.0392** in
> `4dvarnet_enkf/check_outputs/eval/enkf_metrics.json` **must not be used** -- it
> comes from `check_outputs/enkf/`, which is **obs_every_k=4 with only 400 frames
> per day**. And the first 400 frames of each day are an empty field (density is
> only 1/6.9 of the full day's), a doubly favourable subset. The full-day k=1 EnKF
> estimate is in `check_outputs/enkf_k1_full/`, re-scored at **0.0462**.

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

1. **4DVarNet wins on vx** (0.0753 vs 0.0788, 4.4% better), and vx accounts for
   **69%** of total blind error. Senseiver's overall lead comes **entirely from
   the other three channels** (density -21%, vy -30%, var -42%). In other words:
   for velocity along the corridor's main direction, variational assimilation's
   dynamical prior is still stronger.
2. **The EnKF nearly ties on density** (0.0143 vs 0.0139), despite trailing 46%
   overall -- it is dragged down by var (0.0394, 6x Senseiver's). The
   worst-overall method is competitive on the most physically meaningful channel.
3. Reporting only the total gives the false impression that "Senseiver wins across
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
4dvarnet_enkf README's choice to label blind MSE as *the real task* is justified.

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
stronger, but does not change point 1 above (it loses on vx).

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

- **`sys.path` uses `append`, not `insert(0)`**: `4dvarnet_enkf` also has a
  `losses.py`; inserting at the front would shadow this directory's same-named
  module.
- **`/tmp` is node-local**: what a compute node writes to `/tmp` is invisible to
  the login node. Artifacts always go to `runs/` and `check_outputs/`.
- **The first 400 frames are an empty field**: evaluation numbers from
  `--frames 400` are noticeably over-optimistic, fine for a smoke test, not to be
  treated as a conclusion.
- **The EnKF's 0.0392 must not be cited**:
  `4dvarnet_enkf/check_outputs/eval/enkf_metrics.json` comes from
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
