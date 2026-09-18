"""
sensors.py — observation -> variable-length sensor token set
==========================================

In the reference implementation this step is only two lines
(`dataloaders.py:164-165`):

    sensor_values = self.indexed_sensors[frames,]                     # (B, Ns, C)
    sensor_values = torch.cat([sensor_values, self.sensor_positions], -1)

because its sensors are a **fixed set of cells**, Ns is a constant, and the
positional encoding can be pre-repeated. Our sensors are **moving robots**, and
the set and count of cells observed differ every frame, so this step has to be
redone: fetch this frame's observed cells -> concatenate positional encoding ->
pad to the batch's max length -> generate a pad_mask.

One "sensor" = one observed cell; its reading is that cell's C-dimensional
vector, exactly the paper's `s_i in R^{N_c}` (multi-channel sensor) setup.
token = [C channel values, P-dim positional encoding], matching the reference
implementation's concatenation order.

Two things that had to be decided ourselves (the reference implementation gives
no answer):

 1. **Input normalisation.** The reference implementation's 5 datasets are all
    single-channel, and `datasets.py` only does one global scalar division
    (`sea /= sea.max()`, `cyl / 11.0960`). We have 4 channels whose scales differ
    by more than an order of magnitude (std about 0.17 / 0.46 / 0.16 / 0.11
    respectively). Here we do **per-channel standardisation, applied only to the
    encoder's input** -- the target and the loss stay on the raw field
    throughout. Reason: the comparison convention is unweighted 4-channel MSE on
    the raw field, and any per-channel scaling of the target would be secretly
    adding a weight to the loss. Standardising the input is just feature
    conditioning; it does not change the quantity being optimised.

 2. **Empty sensor sets.** When `obs_every_k > 1`, some frames have no
    observation at all, and empty K/V in cross-attention makes softmax produce
    NaN. Here a single all-zero dummy token is inserted and marked valid; the
    model can only output a constant field for such a frame -- that is an
    informational fact, not an implementation flaw.

`build_batch_temporal` (the temporal extension, not in the reference) packs a window
of frames per sample instead of one frame, and also returns each token's relative
time offset Δ. A one-frame window gives exactly `build_batch`'s tokens.
"""
from __future__ import annotations

import numpy as np
import torch


def build_batch(Y_flat, Omega_flat, pos_enc, in_mean, in_std):
    """Packs a batch of frames' observations into a padded batch of sensor tokens.

    Input
        Y_flat     (B, C, HW) float32   noisy partial observation (0 where unobserved)
        Omega_flat (B, HW)    bool      which cells were observed in that frame
        pos_enc    (HW, P)    float32   positional encoding for the whole grid
                                        (positional.PositionalEncoder)
        in_mean/in_std (C,)   float32   the encoder input's per-channel standardisation statistics

    Output
        tokens   (B, Nmax, C+P) float32 torch
        pad_mask (B, Nmax)      bool    torch, True = a padding slot (fed to
                                        nn.MultiheadAttention's key_padding_mask)
        n_sens   (B,)           int     the actual number of sensors per frame (for diagnostics)
    """
    B, C, _ = Y_flat.shape
    P = pos_enc.shape[1]
    idx = [np.flatnonzero(Omega_flat[b]) for b in range(B)]
    n_sens = np.array([len(i) for i in idx], dtype=np.int64)
    n_max = max(1, int(n_sens.max()))

    tokens = np.zeros((B, n_max, C + P), dtype=np.float32)
    pad_mask = np.ones((B, n_max), dtype=bool)              # start with everything marked as padding
    for b, ii in enumerate(idx):
        if len(ii) == 0:
            pad_mask[b, 0] = False                          # empty set: leave one all-zero dummy token
            continue
        v = Y_flat[b][:, ii].T                              # (n_b, C)
        tokens[b, :len(ii), :C] = (v - in_mean) / in_std    # only the input is standardised
        tokens[b, :len(ii), C:] = pos_enc[ii]
        pad_mask[b, :len(ii)] = False

    return (torch.from_numpy(tokens), torch.from_numpy(pad_mask),
            torch.from_numpy(n_sens))


def build_batch_temporal(windows, pos_enc, in_mean, in_std, return_cell_idx=False):
    """Like `build_batch`, but each sample is a window of frames, and every token also
    gets its relative time offset Δ.

    Input
        windows    list of B pairs (Y_win (K_b, C, HW) float32, Omega_win (K_b, HW) bool),
                   oldest frame first; the LAST row is the frame being reconstructed.
                   K_b <= the model's time_window -- shorter at the start of a day,
                   where earlier frames do not exist.
        pos_enc, in_mean, in_std   as in `build_batch`

    Output
        tokens   (B, Nmax, C+P) float32 torch   one token per observed cell per frame
        pad_mask (B, Nmax)      bool    torch   True = padding slot
        dt       (B, Nmax)      int64   torch   Δ = K_b-1-j for frame j of the window
                                                (0 = the current frame); 0 on padding
        n_sens   (B,)           int             real token count per sample

    The same cell seen in several frames becomes several tokens with the same
    positional encoding and different Δ. An all-empty window gets the same single
    all-zero dummy token as `build_batch`, with Δ = 0.
    """
    B = len(windows)
    C = windows[0][0].shape[1]
    P = pos_enc.shape[1]
    packed = []
    for Yw, Ow in windows:
        K = Yw.shape[0]
        vals, cells, dts = [], [], []
        for j in range(K):
            ii = np.flatnonzero(Ow[j])
            if len(ii) == 0:
                continue
            vals.append(Yw[j][:, ii].T)                        # (n_j, C)
            cells.append(ii)
            dts.append(np.full(len(ii), K - 1 - j, dtype=np.int64))
        packed.append((vals, cells, dts))
    n_sens = np.array([sum(len(c) for c in cells) for _, cells, _ in packed], dtype=np.int64)
    n_max = max(1, int(n_sens.max()))

    tokens = np.zeros((B, n_max, C + P), dtype=np.float32)
    pad_mask = np.ones((B, n_max), dtype=bool)
    dt = np.zeros((B, n_max), dtype=np.int64)
    cell_idx = np.zeros((B, n_max), dtype=np.int64)
    for b, (vals, cells, dts) in enumerate(packed):
        if n_sens[b] == 0:
            pad_mask[b, 0] = False                              # empty window: one dummy token, Δ = 0
            continue
        v, ii, d = np.concatenate(vals), np.concatenate(cells), np.concatenate(dts)
        n = len(ii)
        tokens[b, :n, :C] = (v - in_mean) / in_std
        tokens[b, :n, C:] = pos_enc[ii]
        dt[b, :n] = d
        cell_idx[b, :n] = ii
        pad_mask[b, :n] = False

    result = (torch.from_numpy(tokens), torch.from_numpy(pad_mask),
              torch.from_numpy(dt), torch.from_numpy(n_sens))
    if return_cell_idx:
        result += (torch.from_numpy(cell_idx),)
    return result


def query_all(pos_enc, batch):
    """Queries the entire grid. Returns (batch, HW, P).

    The reference implementation randomly queries only `batch_pixels` pixels per
    step, because its domain is too large to query in full (a 3D porous medium
    is 128x128x512). ATC is only 36x12=432 cells, so querying the whole grid at
    once is negligible cost, and no pixel subsampling is done -- removing one
    source of randomness unrelated to the paper.

    The reference implementation's `pix_avail` (cells with value 0 do not
    participate) is **removed outright** here: its purpose is to skip cells with
    "nothing to reconstruct" (land in sea-temperature data, solid material in
    porous media). No such cells exist on the ATC grid -- non-walkable regions
    still have a ground truth, and the evaluation convention scores them too --
    so all 432 cells must be queried.
    """
    return pos_enc[None].expand(batch, -1, -1)
