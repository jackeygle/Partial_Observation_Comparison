"""
navigation.py  —  walkable map, validity check, and A* path planning
=====================================================================

Why this exists
---------------
The ATC corridor is NOT an empty box: it has walls / unreachable cells. Robots
must (a) only start/aim at reachable cells and (b) move along feasible paths that
avoid obstacles — not teleport in straight lines. This module provides:

  - build_valid_mask_from_map()      : walkable region from the REAL ATC map
  - is_valid(cell, mask)             : cell is inside the grid AND walkable
  - astar(start, goal, mask)         : shortest obstacle-avoiding path (8-connected)
  - cell_visibility(radius)          : line-of-sight (walls block sight)

The walkable region is the HYBRID rule (build_valid_mask_from_config):
data criterion "a pedestrian was ever present" AND map criterion "less than 50%
obstacle" (real ATC map at 0.05 m/px, aggregated to the state's 1 m resolution,
ROS-semantics solid obstacles), then largest connected component. The data
criterion carves the true corridor (and excludes everything beyond the boundary
walls); the map criterion removes pillar/stall cells; the 50% threshold is the
supervisor's uniform occupancy rule (previously "any obstacle pixel").
Line-of-sight uses the same map obstacle definition (walls block sight).
"""

from __future__ import annotations

import heapq
import os
from math import sqrt

import numpy as np


# --------------------------------------------------------------------------- #
# Real ATC localization map -> per-cell obstacle fraction / walkable mask
# --------------------------------------------------------------------------- #
def load_atc_map(map_dir=None):
    """Load the real ATC localization map in ROS format (PGM + YAML).

    Output: (img uint8 HxW, resolution m/px, origin (x0,y0))
    Note: the image is effectively binary (0 = laser obstacle points, 1.4%;
    127 = everything else, 98.6%). The 127 value covers BOTH the free space
    inside the corridor AND the unmapped area outside the building; the corridor
    extent is fixed by the 36x12 subset box, and within it the map decides
    free vs obstacle (see build_valid_mask_from_map).
    """
    import yaml
    from PIL import Image
    import config as _cfg
    map_dir = map_dir or _cfg.get("navigation", "map_dir")
    with open(os.path.join(map_dir, "localization_grid.yaml")) as f:
        meta = yaml.safe_load(f)
    img = np.asarray(Image.open(os.path.join(map_dir, meta["image"])))
    return img, float(meta["resolution"]), np.array(meta["origin"][:2], dtype=np.float64)


def world_to_map_pixel(xy, img_shape, map_res, map_origin):
    """World coordinates (x,y) in metres -> map pixel (row, col). ROS convention: origin = bottom-left corner, image row 0 = top."""
    col = ((xy[..., 0] - map_origin[0]) / map_res).astype(int)
    row = img_shape[0] - 1 - ((xy[..., 1] - map_origin[1]) / map_res).astype(int)
    return row, col


def _per_cell_pixel_fraction(pixel_mask, subset="corridor", grid_res=1.0, map_dir=None):
    """Fraction of `pixel_mask`-True pixels inside each 1 m grid cell -> (H, W).

    Each 1 m cell covers ~20x20 map pixels (0.05 m/px). For every cell we sample a
    grid of points inside it, map them local -> world -> map pixel (two transforms),
    and average the boolean `pixel_mask` there. Shared by map_obstacle_fraction and
    build_valid_mask_from_map (raw vs inflated obstacle mask).
    """
    from data_pipeline.h5_to_grid import SUBSETS, rotation_matrix   # grid geometry definitions
    img, map_res, map_origin = load_atc_map(map_dir)
    s = SUBSETS[subset]
    origin, theta, (H, W) = np.asarray(s["origin"]), s["theta"], s["shape"]
    R = rotation_matrix(theta)
    n_sub = max(int(grid_res / map_res), 1)
    offs = (np.arange(n_sub) + 0.5) / n_sub               # uniform sample points within a cell
    oi, oj = np.meshgrid(offs, offs, indexing="ij")
    frac = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            local = np.stack([i + oi.ravel(), j + oj.ravel()], axis=1)
            world = local * grid_res @ R.T + origin
            r, c = world_to_map_pixel(world, img.shape, map_res, map_origin)
            ok = (r >= 0) & (r < img.shape[0]) & (c >= 0) & (c < img.shape[1])
            if ok.any():
                frac[i, j] = pixel_mask[r[ok], c[ok]].mean()
    return frac


def _obstacle_pixels(map_dir=None, solid=True):
    """Boolean obstacle mask at MAP pixel resolution (0.05 m/px), ROS occupancy semantics.

    Follows the ROS map convention used by the ORIGINAL robot_exploration project
    (exploration.py _ros_occupancy_masks), with thresholds read from the map's own
    localization_grid.yaml (negate=0, occupied_thresh=0.65, free_thresh=0.196):

        occ_prob = 1 - px/255
        OCCUPIED : occ_prob > occupied_thresh   (the black laser points, value 0)
        FREE     : occ_prob < free_thresh       (px > 205 — none exist in this map)
        UNKNOWN  : everything else              (value 127, 98.6% of the map)

    `solid=True` applies the ROS reading of UNKNOWN: an unknown region fully
    ENCLOSED by occupied pixels — the interior of a pillar/stall that the laser
    could never see into — belongs to the obstacle, while unknown space connected
    to the rest of the corridor stays traversable. Implementation: bridge
    laser-scan gaps in the outlines (closing, kernel navigation.closing_kernel,
    default 7 px ≈ 0.35 m) so genuinely-enclosed booth/room outlines close up,
    then fill their interiors (binary_fill_holes). The result = OCCUPIED ∪
    enclosed-UNKNOWN, one solid obstacle mask shared by the walkable mask AND
    line-of-sight. (A non-enclosing thin wall line is NOT filled by this — it is
    handled by the data-wall criterion, see build_valid_mask_from_config.)
    """
    import yaml
    import config as _cfg
    map_dir = map_dir or _cfg.get("navigation", "map_dir")
    with open(os.path.join(map_dir, "localization_grid.yaml")) as f:
        meta = yaml.safe_load(f)
    img, _, _ = load_atc_map(map_dir)
    occ_prob = 1.0 - img.astype(np.float64) / 255.0
    if int(meta.get("negate", 0)):
        occ_prob = 1.0 - occ_prob
    obst = occ_prob > float(meta.get("occupied_thresh", 0.65))      # OCCUPIED per ROS
    if solid:
        from scipy import ndimage
        k = int(_cfg.get("navigation", "closing_kernel", default=7))  # bridge laser-scan gaps
        obst = ndimage.binary_closing(obst, structure=np.ones((k, k)))
        obst = ndimage.binary_fill_holes(obst)                      # + enclosed UNKNOWN
    return obst


def map_obstacle_fraction(map_dir=None, subset="corridor", grid_res=1.0,
                          occupied_thresh=0.65):
    """Per-cell RAW obstacle-pixel fraction occ_frac (H, W) from the real map.
    (Diagnostic only — used by checks/check_map_orientation.py; the pipeline
    walkable mask uses inflation, see build_valid_mask_from_map below.)
    """
    obst = _obstacle_pixels(map_dir, occupied_thresh)
    return _per_cell_pixel_fraction(obst, subset, grid_res, map_dir)


def build_valid_mask_from_map(subset="corridor", robot_radius_m=None, grid_res=1.0,
                              map_dir=None):
    """Walkable region straight from the REAL map, 50% occupancy threshold.

    Supervisor's rule (Maryam, 2026-07): keep the map at the SAME 1 m resolution as the
    state and use a uniform 50% occupancy threshold — a 1 m grid cell is an OBSTACLE
    only if at least half of it is covered by obstacle pixels. (The earlier stricter
    rule "ANY obstacle pixel -> wall" over-narrowed the corridor.) The only step after
    that is keeping the largest connected component so robots can't be trapped on an
    unreachable island.

      1. take the real map's obstacle pixels (0.05 m/px);
      2. a 1 m cell is walkable if LESS THAN 50% of it is obstacle pixels;
      3. keep the largest connected walkable component.

    (`robot_radius_m` is accepted for signature compatibility but not used.)
    The corridor extent is fixed by the 36x12 subset box; the map decides free/obstacle.
    """
    import config as _cfg
    obst = _obstacle_pixels(map_dir)
    occ_frac = _per_cell_pixel_fraction(obst, subset, grid_res, map_dir)
    walkable = occ_frac < 0.5                                       # >=50% obstacle pixels -> cell is wall
    if _cfg.get("navigation", "keep_largest_component", default=True):
        walkable = largest_component(walkable)
    return walkable


_VIS_CACHE = {}                                           # in-process cache: {radius: (N,N) bool}


def cell_visibility(radius, subset="corridor", grid_res=1.0, map_dir=None):
    """Precompute line-of-sight visibility between cell pairs, at the 1 m GRID level.

    A sight line from the centre of cell a to the centre of cell b is BLOCKED if it
    passes through any INTERMEDIATE 1 m cell that is a WALL — i.e. a cell whose
    obstacle fraction is >= navigation.occupancy_thresh (default 0.5), the SAME
    definition used for the walkable mask. So a cell marked as wall blocks sight
    exactly as it blocks driving; the display map and the observation computation
    use one identical wall definition, and by construction "through-wall"
    observations = 0.

    Output: vis (N, N) bool, N=H*W. vis[a,b] = a can see b (unblocked & within radius).
    Depends only on the static map; computed once per radius, cached in-process.
    """
    key = (round(float(radius), 3), subset)
    if key in _VIS_CACHE:
        return _VIS_CACHE[key]
    from data_pipeline.h5_to_grid import SUBSETS
    H, W = SUBSETS[subset]["shape"]
    wall = _map_wall(subset, grid_res, map_dir)          # SAME wall definition as the walkable mask
    N = H * W
    ij = np.stack(np.meshgrid(np.arange(H), np.arange(W), indexing="ij"), -1).reshape(-1, 2) + 0.5
    d = np.linalg.norm(ij[:, None] - ij[None], axis=-1)   # centre-to-centre distance, in CELLS

    def blocked(a, b):
        """Sample the a->b segment in cell coords; blocked if any INTERMEDIATE cell is a wall."""
        n = max(int(np.hypot(*(b - a)) / 0.1), 2)         # ~0.1-cell step: never skips a crossed cell
        ai, aj, bi, bj = int(a[0]), int(a[1]), int(b[0]), int(b[1])
        for t in np.linspace(0.0, 1.0, n):
            ci, cj = int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t)
            if (ci, cj) == (ai, aj) or (ci, cj) == (bi, bj):
                continue                                  # skip the endpoint cells themselves
            if 0 <= ci < H and 0 <= cj < W and wall[ci, cj]:
                return True
        return False

    vis = np.zeros((N, N), dtype=bool)
    for a, b in np.argwhere((d <= radius + 1e-9) & (np.arange(N)[:, None] <= np.arange(N)[None])):
        vis[a, b] = vis[b, a] = (a == b) or not blocked(ij[a], ij[b])
    _VIS_CACHE[key] = vis
    return vis


def largest_component(mask):
    """Keep only the largest 8-connected component (prevents obstacle removal from leaving isolated islands that trap robots)."""
    from collections import deque
    seen = np.zeros_like(mask, dtype=bool)
    best = np.zeros_like(mask, dtype=bool)
    H, W = mask.shape
    for si, sj in np.argwhere(mask):
        if seen[si, sj]:
            continue
        q, comp = deque([(si, sj)]), [(si, sj)]
        seen[si, sj] = True
        while q:
            r, c = q.popleft()
            for dr, dc, _ in _NEIGHBORS:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
                    comp.append((rr, cc))
        if len(comp) > best.sum():
            best[:] = False
            best[tuple(np.array(comp).T)] = True
    return best


def _visited_union(split="train", tau=None):
    """Cell was genuinely walked, pooled over ALL days of a split (cached).

    The data criterion is pooled over the WHOLE training set — as maps-of-dynamics
    work does (e.g. CLiFF-map builds its data layer from the full trajectory set) —
    so the mask is FIXED across days: no per-day drift, no chance "wrapped" holes
    from a single quiet day, and no test-day ground truth involved.

    A cell counts as walked iff, on SOME training day, its per-frame PEAK density
    exceeded `tau` (config navigation.visited_tau, default 0.25). The threshold
    rejects triangular-kernel spill-over from pedestrians in the NEIGHBOURING
    cell: someone actually standing in a 1 m cell deposits weight >= 0.25 there
    (worst case: at a corner), whereas density leaking across a wall from the
    next cell stays well below that. With tau=0 stray kernel tails would let the
    mask bleed past the boundary walls (the "green cells outside the wall" bug).

    Cached as a .npy next to the grid_cache files, keyed by split and tau.
    """
    import observation_model as om                        # lazy: avoid circular import
    import config as _cfg
    if tau is None:
        tau = float(_cfg.get("navigation", "visited_tau", default=0.25))
    cache = os.path.join(om.GRID_CACHE, f"visited_union_{split}_tau{tau:g}_corridor.npy")
    if os.path.exists(cache):
        return np.load(cache)
    vis = None
    for fp in om.split_files(split):
        X, _ = om.load_state(fp)
        v = (np.asarray(X)[:, 0].max(axis=0) > tau)       # this day: peak density exceeds tau
        vis = v if vis is None else (vis | v)
    if vis is None:
        raise FileNotFoundError(f"no grid_cache files for split {split!r}")
    np.save(cache, vis)
    return vis


def _map_wall(subset="corridor", grid_res=1.0, map_dir=None):
    """(H, W) bool: cells the MAP marks as WALL — shared by the walkable mask AND
    line-of-sight, so driving and sight always use one identical wall definition.

    Two selectable rules (navigation.obstacle_rule):

      "footprint" (default) — CONNECTED-OBSTACLE FOOTPRINT. Label the solid
        obstacle pixels into connected components, drop components smaller than
        footprint_min_px (isolated laser noise), then a 1 m cell is a wall if a
        surviving component covers more than footprint_overlap of it (default
        0.30). This captures the whole footprint of a connected structure
        (e.g. the stalls lining the corridor) instead of only its >50%-covered
        core, so partially-covered edge cells of a real obstacle become walls too.

      "per_cell" — the supervisor's baseline: a cell is a wall iff its raw
        obstacle fraction >= occupancy_thresh (default 0.5).
    """
    import config as _cfg
    rule = str(_cfg.get("navigation", "obstacle_rule", default="footprint")).lower()
    obst = _obstacle_pixels(map_dir)
    if rule == "footprint":
        from scipy import ndimage
        min_px = int(_cfg.get("navigation", "footprint_min_px", default=30))
        overlap = float(_cfg.get("navigation", "footprint_overlap", default=0.30))
        lab, _n = ndimage.label(obst)                      # connected obstacle components
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        big = np.isin(lab, np.where(sizes >= min_px)[0])   # drop noise-size components
        occ = _per_cell_pixel_fraction(big, subset, grid_res, map_dir)
        return occ > overlap
    thr = float(_cfg.get("navigation", "occupancy_thresh", default=0.5))
    occ = _per_cell_pixel_fraction(obst, subset, grid_res, map_dir)
    return occ >= thr


def build_valid_mask_from_config(X=None):
    """Walkable region for the pipeline (single entry point) — the HYBRID rule.

    Three stages (see check_outputs/navigation/valid_mask_islands.png):

      1. DATA wall   : a cell is walkable only if a pedestrian genuinely walked
                       it on some training day (peak density > visited_tau=0.25,
                       pooled over ALL training days — see _visited_union). This
                       carves the true corridor and excludes everything beyond
                       the boundary walls; the tau rejects cross-wall kernel spill.
      2. MAP wall    : remove cells the map marks as wall — _map_wall(), the
                       connected-obstacle footprint rule (see there), so the
                       stalls lining the corridor are captured as whole
                       structures, not just their >50% cores.
      3. ISLANDS     : keep the largest connected walkable component only.

    `X` is accepted for backward compatibility but NO LONGER USED: the data
    criterion comes from the fixed training-set union, so the mask is identical
    for every day (train/valid/test).
    """
    import config as _cfg
    walkable = ~_map_wall("corridor", 1.0)                 # 2. map wall (connected footprint)
    walkable &= _visited_union("train")                    # 1. data wall (training-set union)
    if _cfg.get("navigation", "keep_largest_component", default=True):
        walkable = largest_component(walkable)             # 3. remove isolated islands
    return walkable


def is_valid(cell, valid_mask):
    """True if `cell` (row, col) is inside the grid and is a walkable cell."""
    r, c = int(cell[0]), int(cell[1])
    H, W = valid_mask.shape
    return 0 <= r < H and 0 <= c < W and bool(valid_mask[r, c])


# 8-connected neighbours (incl. diagonals) + step costs (orthogonal 1, diagonal sqrt2)
_NEIGHBORS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
              (-1, -1, sqrt(2)), (-1, 1, sqrt(2)), (1, -1, sqrt(2)), (1, 1, sqrt(2))]


def astar(start, goal, valid_mask):
    """A* shortest path on the walkable grid (8-connected, obstacle-avoiding).

    Input : start/goal (row, col) ints; valid_mask (H, W) bool
    Output: path = list of (row, col) cells from start to goal (inclusive),
            or None if start/goal invalid or no path exists.

    Cost = Euclidean step length (diagonals cost sqrt(2)); heuristic = straight-line
    Euclidean distance to goal (admissible, so A* returns an optimal path). A diagonal
    step is only allowed if the diagonal cell itself is walkable (corners may still be
    cut; that is acceptable for this coarse 36x12 grid).
    """
    start, goal = (int(start[0]), int(start[1])), (int(goal[0]), int(goal[1]))
    if not is_valid(start, valid_mask) or not is_valid(goal, valid_mask):
        return None                                       # start/goal out of bounds or on a wall: no solution
    if start == goal:
        return [start]                                    # trivial case: staying put is the path
    H, W = valid_mask.shape

    # Heuristic h(cell) = straight-line (Euclidean) distance to the goal. It is
    # always <= the true remaining cost (a straight line is shortest), i.e.
    # "admissible" — the precondition for A* to guarantee an optimal result.
    # (Correctness verified: checks/check_navigation.py cross-checked against an
    # independent Dijkstra; all 373 random paths optimal.)
    def h(cell):
        return sqrt((cell[0] - goal[0]) ** 2 + (cell[1] - goal[1]) ** 2)

    # Three core data structures:
    #   open_heap : min-heap ordered by f = g + h — the top is always the
    #               "most promising looking" cell. g = KNOWN ACTUAL cost from
    #               start to this cell, h = OPTIMISTIC ESTIMATE to the goal.
    #   g_score   : currently known minimum g per cell (updated when a shorter path is found)
    #   came_from : records which cell each cell was reached from, for backtracking at the end
    open_heap = [(h(start), 0.0, start)]                  # (f, g, cell)
    came_from = {}
    g_score = {start: 0.0}
    closed = set()                                        # "finalised" cells (shortest cost settled)
    while open_heap:
        _, g, cur = heapq.heappop(open_heap)              # pop the smallest f
        if cur == goal:
            # Goal popped = its g can no longer improve (guaranteed by admissible h)
            # => the current path is the optimal path
            path = [cur]
            while cur in came_from:                       # backtrack along came_from to the start
                cur = came_from[cur]
                path.append(cur)
            return path[::-1]                             # reverse into start -> goal order
        if cur in closed:
            continue                                      # stale duplicate in the heap (lazy deletion), skip
        closed.add(cur)
        for dr, dc, step in _NEIGHBORS:                   # expand the 8 neighbours
            nb = (cur[0] + dr, cur[1] + dc)
            # Skip neighbours that are out of bounds / not walkable (wall) / finalised
            # — this is where "obstacle avoidance" happens
            if not (0 <= nb[0] < H and 0 <= nb[1] < W) or not valid_mask[nb] or nb in closed:
                continue
            ng = g + step                                 # new cost to nb via cur (orthogonal 1 / diagonal sqrt2)
            if ng < g_score.get(nb, float("inf")):        # found a shorter path to nb -> record and push
                g_score[nb] = ng
                came_from[nb] = cur
                heapq.heappush(open_heap, (ng + h(nb), ng, nb))
                # Note: the old (worse) copy of nb stays in the heap; it is
                # recognised via the closed set and discarded when popped
    return None                                           # heap empty = reachable area exhausted without reaching goal: unreachable


def random_valid_cell(valid_mask, rng):
    """Pick a uniformly random walkable cell (row, col)."""
    cells = np.argwhere(valid_mask)                       # (K, 2) all walkable cells
    return tuple(int(x) for x in cells[rng.integers(0, len(cells))])
