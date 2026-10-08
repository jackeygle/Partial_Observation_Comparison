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

## Status: final per-day checkpoints selected (2026-10-07)

| checkpoint | row in the final comparison | what it is |
|---|---|---|
| `runs/senseiver_A_pd/epoch_092.pt` | Senseiver-A | paper architecture (variant A); 100 epochs, seed 123, per-day routes; test blind walkable RMSE 0.223 |
| `runs/sv_G_k16_pd/epoch_054.pt` | **Senseiver-G (ours)** | one latent token per grid cell with direct read-out, plus a 16-frame causal window; 60 epochs, seed 123, per-day routes; test blind walkable RMSE 0.196 |

Both runs keep every epoch (`--save-every-epoch`), and the reported epoch is chosen
by `checks/select_checkpoint.py` on all seven validation days, every frame, blind
walkable cells, clipped -- the same scope as DINCAE's and 4DVarNet's selection
(`select_valid.json` in each run). The earlier runs `runs/senseiver_A/` and
`runs/capacity/base32_k16_s123/` (same settings) kept only `best.pt`, chosen by
`train.py`'s own score on three validation days, every 20th frame; they are kept
for reference and are no longer the reported models.
The later fixed-route full runs `runs/senseiver_A_full/` and
`runs/capacity/base32_k16_s123_full/` were superseded by the
per-day route retraining shown above.

Both are packaged in `supervisor_evaluation/models/` and scored by
`supervisor_evaluation/evaluate.py` (numbers in the repository
[README](../../README.md#results-and-how-to-read-them)). The extensions tried after
k=16 — temporal mixers, a history-weighted loss, motion-compensated tokens, a
geodesic spatial bias — none beat it; their code, runs and results are in the git tag
`archive-full-2026-09-24`.

---

## End to end

```bash
source sbatch/_env.sh           # from the repository root

# Senseiver-A (self-chains to 100 epochs), every epoch kept
EPOCHS=100 OUT=$PWD/methods/senseiver/runs/senseiver_A_pd \
  sbatch methods/senseiver/sbatch/submit_train.sbatch --save-every-epoch --trajectory-mode per_day

# Senseiver-G: grid latent, direct read-out, 16-frame window, 60 epochs, seed 123 (default)
EPOCHS=60 OUT=$PWD/methods/senseiver/runs/sv_G_k16_pd \
  sbatch methods/senseiver/sbatch/submit_train.sbatch \
  --latent-mode grid --readout direct --time-window 16 --save-every-epoch --trajectory-mode per_day

# After both training chains finish, select each epoch on the whole validation split -> <run>/select_valid.json
# (--epochs A-B splits the work over several jobs; then --merge)
sbatch methods/senseiver/sbatch/select_checkpoint.sbatch methods/senseiver/runs/senseiver_A_pd
sbatch methods/senseiver/sbatch/select_checkpoint.sbatch methods/senseiver/runs/sv_G_k16_pd

# Evaluation against the other methods
python3 supervisor_evaluation/evaluate.py prepare
sbatch supervisor_evaluation/sbatch/full.sbatch
```

`train.py`'s `best.pt` uses a quick validation score (three days, every 20th
frame) and is not the reported checkpoint. **torch only runs on GPU
nodes** (`sbatch`, or `srun -p gpu-debug --gres=gpu:1`), never on the login node.

---

## Directory layout

| path | role |
|---|---|
| `positional.py` | the paper's `a = PE(chi)`: sin-cos positional encoding; `TemporalEncoding` for the time window |
| `model.py` | the paper's `z = E(a_s,s)` and `s_hat_q = D(z,a_q)`: Encoder / Decoder; plus `GridEncoder` (variant G) |
| `sensors.py` | **new in this project**: observation -> variable-length sensor token set (padding + pad_mask), also over a time window |
| `network.py` | the whole model (plain PyTorch, no Lightning) |
| `losses.py` | training loss (the reference implementation's unweighted MSE) + a diagnostic breakdown |
| `dataset.py` | one day -> samples (`DayBank`, `TemporalDayBank`); observation config read verbatim from `crowdcore/config.yaml` |
| `train.py` | training loop (Adam / AMP / `--resume`) |
| `sbatch/submit_train.sbatch` | training job (self-chains to resume) |
| `runs/` | `metrics.jsonl` of the two final runs (checkpoints gitignored) |

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
`checks/check_model.py` (archive tag) turned permutation invariance and padding invariance into
rerunnable assertions (measured max|delta| of 2.7e-7 and 0.0 respectively) -- this is
the premise the method's case rests on.

---

## Pipeline: one frame of data from the raw grid to the loss

Every shape below was actually produced by `checks/trace_pipeline.py` (archive tag), not
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
`checks/check_model.py` (archive tag) turned this into an assertion (shuffled order max|delta| =
2.7e-7, padded to different lengths max|delta| = 0.0).

**6 queries all 432 cells, not just the 290 walkable ones.** Reason: see deviation
#1 -- robots can see cells they cannot drive into (measured 32.1%). The reported
scores then use the walkable cells only.

**Normalisation only happens at 4.** The first 4 dimensions of the tokens are
`(Y - mean) / std`, but both 6's output and 7's loss operate on the **raw field**.
Normalising the input side is feature conditioning; per-channel scaling of the
target would be equivalent to adding a weight to the loss, which would change the
quantity being optimised.

**The four channels are trained together, and the loss is dominated by vx.** The
four channels are summed unweighted (the paper's Eq., copied as-is), and vx's std
is 0.54 while var's is only 0.12, so vx dominates a single MSE. This is not a bug,
it is a property of this metric -- the header of
`compare/compare_channels.py` (archive tag) complained about the same thing. **Both
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
   On the ATC grid density 0 is a valid value ("nobody is here") and non-walkable
   cells still have a ground truth, so zero cannot mean "invalid". A check script
   (archive tag) measured **32.1% of observed cells falling on non-walkable ground**
   (robots can see pillars they cannot drive into, see the methods/varnet README), so
   those observations are real inputs and the query set is kept at all 432 cells.
   Only walkable cells are scored.
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

## The fairness contract with the other methods

Every method faces **exactly the same** observations; differences only come from
the method itself:

| Anchor | Approach |
|---|---|
| Observation parameters | read verbatim from `crowdcore/config.yaml`'s `observation` section; this directory sets no defaults of its own |
| Observation generation | calls `om.generate_observations` + `nav.build_valid_mask_from_config` directly |
| Data split | `om.split_files()`; 32 training days / 7 validation days / 7 test days, training days strictly earlier than test days |
| Robot routes | seeded by the day's date (`trajectory_mode: per_day`), identical for every method on a day |
| Metrics | pooled RMSE on unobserved walkable cells (the headline) and on all walkable cells, raw field, four channels unweighted; scored by `supervisor_evaluation/evaluate.py` |
| Physical clipping | same as the EnKF: density[0,5], vx/vy[-5,5], var[0,2] (on by default) |
| Initial value | none initialised from the ground truth |
| Timing | only the model's forward pass is timed, not data preparation (a constant shared by all three methods) |

Results are in the repository [README](../../README.md#results-and-how-to-read-them).

### One sentence that must go in the conclusions

**Senseiver-A reconstructs frame t from only frame t's observation** (as in the
paper, with no temporal encoding at all -- the paper's sec.3 Discussion explicitly
says a sin-cos time encoding was tried and failed), while 4DVarNet sees a 200-frame
window and DINCAE t-1..t+1. Senseiver-G adds a causal 16-frame window of past
observations. The gap between A and G is therefore what the time dimension is
worth to this architecture; without this sentence, the table would be misread.

---

## Senseiver-G: grid latent and a temporal window (our extension)

**Grid latent ("G-direct").** Our state is 4 × 36 × 12 = 1,728 numbers per frame,
while variant A's latent array is 64 × 32 = 2,048 numbers — larger than the state it
is meant to compress. In variant A the latents only turn a variable-length, unordered
sensor set into something fixed-length, and the grid already is fixed-length. Variant
G replaces the 64 abstract latents with one latent token per grid cell (432 tokens)
and reads each cell's output directly from its own token (`--latent-mode grid
--readout direct`). On matched seeds it beat variant A clearly.

**Temporal window.** The 432 cell tokens' cross-attention reads the sensor tokens
of the **last k frames** instead of the current frame only (`--time-window k`); the
final model uses k = 16. Each sensor token carries its relative time offset Δ = t_query - t_token
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
- `compare/compare5.py`'s Senseiver inference has a branch taken only by models with
  `time_window > 1`; variant A's scores are bit-identical with and without it.

Files: `positional.TemporalEncoding`, `sensors.build_batch_temporal`,
`dataset.TemporalDayBank`, `network.py` (`time_window`, `time_dim`, `time_scalar`;
default k=1 adds nothing), `train.py --time-window k`.

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
  the login node. Artifacts always go to `runs/`.
- **The first 400 frames are an empty field**: evaluation numbers from
  `--frames 400` are noticeably over-optimistic, fine for a smoke test, not to be
  treated as a conclusion.
- `observation_model.generate_observations`'s docstring says "only observe
  walkable cells," which does not match this project's README or the actual data
  (measured 32.1% falling outside). Trust the data.
- **`sbatch/submit_train.sbatch` resumes into variant A by default**: `OUT` defaults
  to `runs/senseiver_A`, and the script adds `--resume` whenever `$OUT/last.pt`
  exists. A new run whose `OUT` does not reach the job would continue training --
  and overwrite -- the official variant A checkpoint. Set `OUT`
  to an absolute path **and** pass `--out` among the script arguments (argparse
  keeps the last one).
- **`--seed` is not only the initialisation**: it also seeds the robot paths that
  generate the training and validation observations (`DayBank(seed=...)`). Compare
  variants at matched seeds. The reported evaluation generates its own observations,
  with routes seeded by each test day's date.
- **A smoke test shorter than ~20 steps with `--amp` looks like "no learning"**: the loss is
  `reduction='sum'`, so GradScaler's initial scale (65536) overflows fp16 and it skips
  and halves for the first ~18-19 steps (measured: 19/40 for k=1, 18/40 for k=4; identical
  validation numbers across the smoke's two 7-step epochs). Harmless in real training --
variants A and G carry the same ~19 skipped steps out of ~200k -- but to check that a
change learns, run without `--amp` or for more steps.

---
