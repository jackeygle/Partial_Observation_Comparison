"""
observation_model.py  —  Baseline Component 1: partial-observation generation
=============================================================================

Role in the framework
----------------------
This is the FIRST component of the variational reconstruction baseline (the
"4DVarNet-style" framework adapted to the ATC pedestrian dataset, treated as a
third example alongside Lorenz-63 / Lorenz-96).

It answers exactly one question: given the COMPLETE macroscopic ATC field, what
do the robots actually see?  Everything downstream (prior model, variational
cost, iterative solver) consumes the outputs produced here.

Analogy to the paper (Lorenz-96): there the 40-dim state is observed through a
fixed 20-dim observation, x_obs = C x.  Here the role of `C` is played by a
moving multi-robot sensing model, so the observed entries change every second.

Inputs / Outputs (the contract this module guarantees)
------------------------------------------------------
INPUT
  - A grid_cache HDF5 file produced for the ATC corridor at 1.0 s period.
    dataset 'grid'  : float32, shape (T, C=4, H=36, W=12)
                      channels = [density, vx, vy, velocity-variance]
    dataset 'time'  : float64, shape (T,)        (seconds; period = 1.0 s = T window unit)

OUTPUT  (numpy arrays, all float32 unless noted)
  - X       : (T, C, H, W)   full ground-truth state           (the unknown to reconstruct)
  - Y       : (T, C, H, W)   partial observation on the grid    (= X on observed cells, 0 elsewhere;
                                                                  + sensor noise if enabled)
  - Y_clean : (T, C, H, W)   same as Y but without noise        (for diagnostics)
  - Omega   : (T, H, W) bool observation mask (1 = observed)    (shared across the 4 channels)
  - X0      : (T, C, H, W)   initial full-state estimate        (input to the solver; zeros / interp / prev)
  - positions : (T, num_agents, 2) int   robot (row,col) per second (for visualisation / provenance)

Conventions fixed for the baseline — ALL VALUES LIVE IN config.yaml
-------------------------------------------------------------------
  num_agents, sensing_range, obs_std, init_method : config.yaml -> observation
  state shape (4, 36, 12), 1 s / frame            : config.yaml -> grid
  (Provenance comments for every value are also in config.yaml: why the radius
  is 7, how the noise std was calibrated.)

This module is pure numpy + h5py (+ matplotlib only for the optional figures);
it has no dependency on any previous code, so its inputs and outputs are fully
owned and traceable.

This module is import-only (the pure functional API). For verification,
diagnostics and figures see `checks/check_observation_model.py`:
    python3 checks/check_observation_model.py --file <grid_cache.h5> --frames 60
(figures/report go to check_outputs/observation_model/)
"""

from __future__ import annotations

import os

import h5py
import numpy as np

from crowdcore import navigation                                        # walkable region / is_valid / A* path planning

# --------------------------------------------------------------------------- #
# Fixed conventions — ALL centralised in config.yaml (supervisor's requirement);
# this module only reads them. The PROVENANCE of every value is documented in
# config.yaml comments (why the sensing radius is 7, how the noise std was calibrated).
# --------------------------------------------------------------------------- #
from crowdcore import config

DATA_ROOT = config.get("data", "root")
GRID_CACHE = os.path.join(DATA_ROOT, config.get("data", "grid_cache"))  # pre-gridded data directory

# The state has 4 channels, in fixed order [density, x-velocity, y-velocity, velocity variance]
CHANNEL_NAMES = tuple(config.get("grid", "channels"))
NUM_AGENTS = config.get("observation", "num_agents")     # number of robots (sensors)
SENSING_RANGE = config.get("observation", "sensing_range")  # sensing radius (cells, Euclidean distance)
# Per-channel sensor noise std. Provenance (data calibration, 0.25 x robust std): see config.yaml comments.
OBS_STD = np.array(config.get("observation", "obs_std"), dtype=np.float32)


# --------------------------------------------------------------------------- #
# 1. Load the full state X
# --------------------------------------------------------------------------- #
def load_state(path):
    """Load one grid_cache day.

    Input : path to atc-YYYYMMDD_corridor_1.0s.h5
    Output: X (T, C, H, W) float32, time (T,) float64
    """
    with h5py.File(path, "r") as f:
        # The 'grid' dataset is the full state sequence: (frames T, channels 4, height 36, width 12)
        X = f["grid"][:].astype(np.float32)             # (T, C, H, W)
        # 'time' is the timestamp in seconds per frame; if the file lacks it, fall back to 0,1,2,...
        time = f["time"][:] if "time" in f else np.arange(X.shape[0], dtype=np.float64)
    # Sanity check: confirm this really is a 4-dim, 4-channel grid and not something else
    assert X.ndim == 4 and X.shape[1] == 4, f"unexpected grid shape {X.shape}"
    return X, time


def resolve_file(arg):
    """Accept an explicit path, a date stem (e.g. atc-20121028), or 'first'."""
    # 'first'/None: automatically take the first .h5 in the directory, handy for quick tests
    if arg in (None, "first"):
        files = sorted(f for f in os.listdir(GRID_CACHE) if f.endswith(".h5"))
        if not files:
            raise FileNotFoundError(f"no grid_cache files under {GRID_CACHE}")
        return os.path.join(GRID_CACHE, files[0])
    # Already an existing absolute path: use it directly
    if os.path.isabs(arg) and os.path.exists(arg):
        return arg
    # Otherwise treat it as a date stem (e.g. atc-20121028) and complete it to the full filename
    cand = os.path.join(GRID_CACHE, arg if arg.endswith(".h5") else f"{arg}_corridor_1.0s.h5")
    if os.path.exists(cand):
        return cand
    raise FileNotFoundError(f"cannot resolve grid_cache file from {arg!r}")


def split_files(split="train", suffix="_corridor_1.0s.h5"):
    """Return the list of grid_cache files for a given split (train/valid/test).

    Reads data/sunday_atc_{split}.lst (containing ATC/Sundays/atc-YYYYMMDD.h5 entries) and maps them to grid_cache.
    """
    lst = os.path.join(DATA_ROOT, config.get("data", "split_list").format(split=split))
    paths = []
    with open(lst) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stem = os.path.splitext(os.path.basename(line))[0]   # atc-YYYYMMDD
            gc = os.path.join(GRID_CACHE, stem + suffix)
            if os.path.exists(gc):
                paths.append(gc)
    return paths


# --------------------------------------------------------------------------- #
# 2. Multi-agent sensing model
# --------------------------------------------------------------------------- #
class MultiAgentSensor:
    """A small team of robots navigating the corridor; each observes a disk of cells.

    The sensing model (corridor-aware, A*-routed):
      - each robot starts at a random WALKABLE cell and is given a random WALKABLE
        goal (only reachable cells, via `valid_mask`);
      - it follows the A* shortest obstacle-avoiding path toward the goal, one cell
        per second; on arrival it picks a new walkable goal and replans;
      - a cell is OBSERVED at time t if it is walkable AND within `sensing_range`
        (Euclidean) of ANY robot at time t.
    The spatial mask is shared by all 4 channels (per-channel selection happens in
    generate_observations); a robot reaching a cell measures density/vx/vy/var there.

    If `valid_mask` is None every cell is treated as walkable (empty-map fallback).
    """

    def __init__(self, grid_size, sensing_range=SENSING_RANGE,
                 num_agents=NUM_AGENTS, seed=0, valid_mask=None, line_of_sight=None):
        self.H, self.W = grid_size                       # grid size (height, width)
        self.sensing_range = float(sensing_range)
        self.num_agents = num_agents
        # Walkable-region mask; None means the whole grid is walkable (empty-map fallback)
        self.valid_mask = (np.ones((self.H, self.W), dtype=bool)
                           if valid_mask is None else valid_mask.astype(bool))
        # Line-of-sight occlusion: obstacles block sight as well as movement
        # (a robot cannot see through a stall to the other side).
        # The visibility matrix is precomputed from the real map
        # (navigation.cell_visibility), zero runtime cost.
        if line_of_sight is None:
            line_of_sight = config.get("observation", "line_of_sight", default=False)
        self.line_of_sight = bool(line_of_sight)
        self._vis = (navigation.cell_visibility(self.sensing_range)
                     .reshape(self.H * self.W, self.H, self.W)
                     if self.line_of_sight else None)
        # default_rng(seed): fixed random seed so robot trajectories are reproducible
        rng = np.random.default_rng(seed)
        self._rng = rng
        # Each robot: random WALKABLE start cell, plan an A* path to a WALKABLE goal
        self.positions = np.zeros((num_agents, 2), dtype=np.int64)
        self._paths = [None] * num_agents                 # each robot's current A* path
        self._path_idx = [0] * num_agents                 # current step index along the path
        for i in range(num_agents):
            self.positions[i] = navigation.random_valid_cell(self.valid_mask, rng)
            self._plan_new_path(i)
        # Precompute row/col coordinate grids for the sensing-disk test
        self._rr, self._cc = np.meshgrid(
            np.arange(self.H), np.arange(self.W), indexing="ij")

    def _plan_new_path(self, i):
        """Give robot i a random walkable goal and plan an A* path (re-draw the goal if planning fails)."""
        start = tuple(int(x) for x in self.positions[i])
        for _ in range(20):                               # retry a few times at most to find a reachable goal
            goal = navigation.random_valid_cell(self.valid_mask, self._rng)
            path = navigation.astar(start, goal, self.valid_mask)
            if path and len(path) > 1:
                self._paths[i] = path
                self._path_idx[i] = 0
                return
        self._paths[i] = [start]                          # fallback: stay in place
        self._path_idx[i] = 0

    def step(self):
        """Advance every robot one cell along its A* path; replan on arrival."""
        for i in range(self.num_agents):
            path, idx = self._paths[i], self._path_idx[i]
            if idx + 1 >= len(path):                       # reached the goal -> plan a new path
                self._plan_new_path(i)
                path, idx = self._paths[i], self._path_idx[i]
            if idx + 1 < len(path):                        # advance one cell along the path
                self._path_idx[i] = idx + 1
                self.positions[i] = path[idx + 1]

    def observed_mask(self):
        """(H, W) bool: cells a robot can SEE this second = sensing disk ∩ line-of-sight.

        NOTE — observation is NOT intersected with the walkable (navigation) mask.
        "Where a robot can DRIVE" (walkable) and "what a robot can SEE" (this mask)
        are different physical questions: a robot standing next to a pillar cannot
        drive into it, but it CAN see the crowd around it. Intersecting with the
        walkable mask would wrongly blind those obstacle-adjacent, populated cells
        (~1/4 of crowd mass sits next to mall pillars). Sight is limited only by
        range and line-of-sight; walls block the sight line (line_of_sight=True).

        Measured on atc-20130811 (2026-09-05), because "walls hold ~0 density" used to
        be asserted here without a number and it is only half true:

          * non-walkable cells hold 5.57% of the day's total crowd mass, NOT ~0 --
            the map's 50%-occupancy rule rounds pillar/stall edges into obstacles
            while the tracker still puts people there (peak density 1.83 on such a
            cell, against 2.80 on the walkable side);
          * but of the non-walkable cells the robots actually SEE, 98.74% are empty,
            and they carry 0.95% of the observed crowd mass.

        So the sight rule costs ~1% of the observed signal, which is why intersecting
        with walkable is not worth a retrain: it would delete that 1% of true signal
        while those cells stay in the ground truth under the allcells/full conventions,
        turning them into permanent blind spots.
        """
        observed = np.zeros((self.H, self.W), dtype=bool)
        r2 = self.sensing_range ** 2                      # compare squared distances, avoiding the sqrt
        for (r0, c0) in self.positions:
            # (row diff)^2 + (col diff)^2 <= radius^2 means inside this robot's sensing disk;
            # |= is logical OR: coverage from multiple robots is unioned
            # (a cell counts as observed if ANY robot sees it)
            disk = (self._rr - r0) ** 2 + (self._cc - c0) ** 2 <= r2
            if self._vis is not None:                     # line-of-sight occlusion: disk AND visible
                disk &= self._vis[int(r0) * self.W + int(c0)]
            observed |= disk
        return observed


# --------------------------------------------------------------------------- #
# 3. Observation operator: generate Y and Omega from X
# --------------------------------------------------------------------------- #
def generate_observations(X, sensing_range=SENSING_RANGE, num_agents=NUM_AGENTS,
                          add_noise=True, seed=0, obs_channels=None, valid_mask=None,
                          obs_std=None, obs_every_k=None):
    """Apply the moving multi-robot sensor over a whole sequence.

    Input
        X            (T, C, H, W)
        obs_channels : which channels are observed. Default None = all 4 channels
                       observed together (a robot reaching a cell measures
                       density/vx/vy/variance simultaneously). Pass e.g. (0,) for
                       "measure density only, not velocity", or a (C,) boolean
                       array. This provides per-channel independent observation flexibility.
        valid_mask   : (H,W) bool walkable region. Default None = built per the
                       config.yaml navigation section (hybrid: data criterion
                       "a pedestrian was ever present" AND real-map removal of
                       interior pillar/stall cells). Robots navigate with A*
                       only within the walkable region and only observe walkable cells.
        obs_std      : (C,) per-channel observation noise std. Default None = use
                       the module constant OBS_STD (data-calibrated, 0.25 x robust
                       std). To switch to REAL sensor specifications, just pass a
                       length-C array, e.g.
                       generate_observations(X, obs_std=[sigma_density, sigma_vx, sigma_vy, sigma_var]).
        obs_every_k  : TEMPORAL observation sparsity. Only every k-th frame carries
                       observations (the rest have an empty mask); the solver still
                       reconstructs ALL frames. Matches the paper's "observation
                       sampling coarser than the assimilation time step" (Lorenz-96
                       observes every 4 steps, Lorenz-63 every 8). Default None =
                       config observation.obs_every_k (1 = observe every frame).

    Output: dict with
        Y        (T, C, H, W) float32  noisy partial observation on the grid
        Y_clean  (T, C, H, W) float32  noiseless partial observation
        Omega    (T, H, W)    bool     spatial mask: cells the robots see this second (for plotting / coverage stats)
        Omega_c  (T, C, H, W) bool     per-channel mask: whether each (channel,cell) is actually observed
                                       (default = Omega broadcast to 4 channels; may differ due to obs_channels)
        positions(T, num_agents, 2) int robot positions per second
    """
    T, C, H, W = X.shape
    if valid_mask is None:                               # not given -> build per config
        valid_mask = navigation.build_valid_mask_from_config(X)
    sensor = MultiAgentSensor((H, W), sensing_range, num_agents, seed=seed,
                              valid_mask=valid_mask)
    noise_rng = np.random.default_rng(seed + 1)          # separate RNG stream: noise decoupled from robot paths

    # Step 1: advance the robots frame by frame; record the SPATIAL observation
    # mask Omega and the robot positions for every second
    if obs_every_k is None:                              # temporal sparsity (paper: coarse time grid)
        obs_every_k = config.get("observation", "obs_every_k", default=1)
    Omega = np.zeros((T, H, W), dtype=bool)
    positions = np.zeros((T, num_agents, 2), dtype=np.int64)
    for t in range(T):
        sensor.step()                                    # robots move one cell (every frame)
        positions[t] = sensor.positions
        if t % obs_every_k == 0:                         # observations only on every k-th frame;
            Omega[t] = sensor.observed_mask()            #   other frames stay empty, yet are still
                                                         #   reconstructed (coarse-time-grid observation)

    # Step 2: build the PER-CHANNEL mask Omega_c. First turn "which channels are
    # observed" into a boolean vector chan_sel (C,):
    chan_sel = np.zeros(C, dtype=bool)
    if obs_channels is None:                             # default: all 4 channels observed
        chan_sel[:] = True
    else:
        chan_sel[np.asarray(obs_channels, dtype=int)] = True
    # A (channel, cell) is observed = the cell is seen by a robot (Omega) AND
    # that channel is sensed (chan_sel)
    Omega_c = Omega[:, None, :, :] & chan_sel[None, :, None, None]   # (T,C,H,W) bool

    # Step 3: cut the observed part out of X with the per-channel mask, and add
    # Gaussian noise on the observed entries
    m = Omega_c.astype(np.float32)
    Y_clean = (X * m).astype(np.float32)                 # observed (channel,cell) keep truth, rest = 0
    if add_noise:
        std = OBS_STD if obs_std is None else np.asarray(obs_std, dtype=np.float32)
        noise = noise_rng.normal(size=X.shape).astype(np.float32) * std[None, :, None, None]
        Y = (Y_clean + noise * m).astype(np.float32)     # noise only on observed (channel,cell)
    else:
        Y = Y_clean.copy()

    return {"Y": Y, "Y_clean": Y_clean, "Omega": Omega, "Omega_c": Omega_c,
            "positions": positions, "valid_mask": valid_mask}


# --------------------------------------------------------------------------- #
# 4. Initial full-state estimate X0
# --------------------------------------------------------------------------- #
def fill_missing_state(Y, mask, method="prev"):
    """Build a complete field X0 from the PARTIAL observations by filling the
    unobserved cells (this is the solver's starting point).

    (Renamed from `initialize_state`: the function does not "initialise" anything
    generic — it reconstructs an initial *full* field from partial observations.)

    Works PER CHANNEL, so it supports the flexible masking where some channels are
    observed and others are not in the same cell.

    Methods:
      'zeros'  : unobserved (channel,cell) = 0.
      'prev'   : carry the most recent observed value forward in time, per channel;
                 (channel,cell) never yet observed stays 0. (default, usually best)
      'interp' : spatial nearest-observed fill per frame & per channel.

    Input
        Y    : (T,C,H,W)
        mask : (T,C,H,W) per-channel bool  OR  (T,H,W) spatial bool (broadcast to C)
    Output: X0 (T,C,H,W) float32
    """
    T, C, H, W = Y.shape
    # Normalise to a per-channel mask mc (T,C,H,W): a (T,H,W) spatial mask is
    # broadcast across the 4 channels
    if mask.ndim == 3:
        mc = np.broadcast_to(mask[:, None, :, :], Y.shape)
    else:
        mc = mask
    mc = mc.astype(bool)
    # All methods keep the observed (channel,cell) values; they only differ in
    # how the UNOBSERVED entries are filled
    X0 = (Y * mc.astype(np.float32)).astype(np.float32)

    # Method 1: unobserved (channel,cell) stay 0
    if method == "zeros":
        return X0

    # Method 2 (default): forward fill in time — if a (channel,cell) is not
    # observed this second, use its most recent observed value
    if method == "prev":
        last = np.zeros((C, H, W), dtype=np.float32)      # per (channel,cell): "last observed value"
        seen = np.zeros((C, H, W), dtype=bool)            # per (channel,cell): "ever observed?"
        for t in range(T):
            obs = mc[t]                                   # (C,H,W) this second's per-channel observations
            last[obs] = Y[t][obs]                         # refresh memory where observed
            seen |= obs
            fill = ~obs & seen                            # unobserved now, but observed before
            X0[t][fill] = last[fill]                      # fill from memory
        return X0

    # Method 3: spatial nearest-neighbour fill — per frame and per channel,
    # copy the value of the nearest observed cell
    if method == "interp":
        from scipy import ndimage                          # scipy is in the env; import lazily
        for t in range(T):
            for f in range(C):
                obs = mc[t, f]
                if obs.all() or not obs.any():            # skip fully observed / fully unobserved
                    continue
                idx = ndimage.distance_transform_edt(
                    ~obs, return_distances=False, return_indices=True)
                X0[t, f] = Y[t, f][tuple(idx)]            # fill the channel from nearest observed cells
        return X0

    raise ValueError(f"unknown init method {method!r}")


# --------------------------------------------------------------------------- #
# step 4: cut the per-frame sequence into solver windows
# --------------------------------------------------------------------------- #
# The last step of this pipeline. Phi and the solver expect (B, C, T, H, W): channels and
# time on separate axes, with C before T. Getting there from per-frame data is a reshape
# plus a transpose, and that transpose is why this lives in exactly one place —
# `(0, 2, 1, 3, 4)` with two of the digits swapped still yields a correctly-shaped tensor,
# so a mistake there does not raise, it silently mixes up channels and time. There were
# nine hand-written copies of it across train_varnet.py and checks/, differing in whether
# they truncated, whether they returned numpy or torch, and what they called the channel
# axis. One of them did not truncate at all and only worked because its input happened to
# be an exact multiple of dT.

def to_windows(arr, dT, as_torch=False, device=None):
    """Cut (T, C, H, W) into non-overlapping (n_windows, C, dT, H, W).

    arr       : numpy array or anything np.asarray accepts, frames first
    dT        : frames per window; the tail that does not fill a window is dropped
    as_torch  : return a float32 torch tensor instead of a numpy array
    device    : move the tensor here (implies as_torch)
    """
    a = np.asarray(arr)
    n = (a.shape[0] // dT) * dT
    C, H, W = a.shape[1:]
    w = np.ascontiguousarray(a[:n].reshape(-1, dT, C, H, W).transpose(0, 2, 1, 3, 4))
    if as_torch or device is not None:
        import torch                                  # lazy: most callers of this module never
        t = torch.from_numpy(w).float()               # need torch, and it is a heavy import
        return t.to(device) if device is not None else t
    return w


def from_windows(win):
    """Inverse of to_windows: (n_windows, C, dT, H, W) -> (n_windows*dT, C, H, W)."""
    a = np.asarray(win)
    nw, C, dT, H, W = a.shape
    return np.ascontiguousarray(a.transpose(0, 2, 1, 3, 4).reshape(nw * dT, C, H, W))
