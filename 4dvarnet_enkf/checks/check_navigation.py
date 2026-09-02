"""
check_navigation.py  —  verification & figures for navigation.py (A* path planning)
====================================================================================

This is NOT part of the pipeline. It only *exercises* `navigation.py`:
  - hand-crafted grids where the correct answer is known (unit tests),
  - cross-check against an independent Dijkstra on random obstacle maps
    (A* must return the SAME optimal cost — verifies optimality),
  - invariant checks on random start/goal pairs (path continuity, walkability,
    reachability consistent with BFS flood fill),
  - a figure of a few A* paths on the real ATC walkable mask (visual check).

Run:
    module load scicomp-pytorch-env/2026.1
    python3 checks/check_navigation.py                     # synthetic tests + real-data figure
    python3 checks/check_navigation.py --synthetic-only    # no data needed
"""

from __future__ import annotations

import argparse
import heapq
import os
import sys
from collections import deque
from math import sqrt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make parent-level modules importable

import numpy as np

from navigation import _NEIGHBORS, astar, is_valid, random_valid_cell


# --------------------------------------------------------------------------- #
# Helpers (independent of navigation.py internals)
# --------------------------------------------------------------------------- #
def path_cost(path):
    """Sum of Euclidean step lengths along the path."""
    return sum(sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) for a, b in zip(path, path[1:]))


def octile(a, b):
    """Optimal 8-connected cost on an EMPTY grid: max(d) + (sqrt2-1)*min(d)."""
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dr, dc) + (sqrt(2) - 1) * min(dr, dc)


def dijkstra_cost(start, goal, mask):
    """Reference shortest-path cost WITHOUT heuristic (independent implementation)."""
    H, W = mask.shape
    dist = {start: 0.0}
    heap = [(0.0, start)]
    while heap:
        d, cur = heapq.heappop(heap)
        if cur == goal:
            return d
        if d > dist.get(cur, float("inf")):
            continue
        for dr, dc, step in _NEIGHBORS:
            nb = (cur[0] + dr, cur[1] + dc)
            if 0 <= nb[0] < H and 0 <= nb[1] < W and mask[nb] and d + step < dist.get(nb, float("inf")):
                dist[nb] = d + step
                heapq.heappush(heap, (d + step, nb))
    return None


def bfs_component(start, mask):
    """All cells 8-connected-reachable from start (flood fill) — ground truth reachability."""
    seen, q = {start}, deque([start])
    H, W = mask.shape
    while q:
        cur = q.popleft()
        for dr, dc, _ in _NEIGHBORS:
            nb = (cur[0] + dr, cur[1] + dc)
            if 0 <= nb[0] < H and 0 <= nb[1] < W and mask[nb] and nb not in seen:
                seen.add(nb)
                q.append(nb)
    return seen


def assert_path_valid(path, start, goal, mask):
    """Invariants every returned path must satisfy (regardless of the mask)."""
    assert path[0] == start and path[-1] == goal, "path must run start -> goal"
    assert len(set(path)) == len(path), "path must not revisit a cell"
    for cell in path:
        assert is_valid(cell, mask), f"path leaves the walkable region at {cell}"
    for a, b in zip(path, path[1:]):
        assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1, f"non-adjacent step {a}->{b}"


# --------------------------------------------------------------------------- #
# 1) Hand-crafted grids — answers known by construction
# --------------------------------------------------------------------------- #
def test_handcrafted():
    free = np.ones((10, 10), dtype=bool)

    # Empty grid: cost must equal the optimal 8-connected (octile) distance
    for s, g in [((0, 0), (9, 9)), ((0, 0), (0, 9)), ((2, 3), (7, 4))]:
        p = astar(s, g, free)
        assert_path_valid(p, s, g, free)
        assert abs(path_cost(p) - octile(s, g)) < 1e-9, f"empty grid {s}->{g}: not optimal"

    # start == goal
    assert astar((3, 3), (3, 3), free) == [(3, 3)]

    # Invalid endpoints: out of bounds / on a wall -> None
    wall = free.copy(); wall[5, 5] = False
    assert astar((-1, 0), (2, 2), free) is None
    assert astar((0, 0), (10, 0), free) is None
    assert astar((5, 5), (0, 0), wall) is None
    assert astar((0, 0), (5, 5), wall) is None

    # A wall with a gap: the path must detour, cost = optimal cost from independent Dijkstra
    walled = np.ones((7, 7), dtype=bool)
    walled[1:7, 3] = False                                # column 3 blocked except a gap at row 0
    s, g = (6, 0), (6, 6)
    p = astar(s, g, walled)
    assert_path_valid(p, s, g, walled)
    assert all(walled[c] for c in p)
    assert abs(path_cost(p) - dijkstra_cost(s, g, walled)) < 1e-9, "walled grid: not optimal"
    assert path_cost(p) > octile(s, g), "path should be longer than straight line (it detours)"

    # Fully separated: no path -> None
    blocked = np.ones((7, 7), dtype=bool)
    blocked[:, 3] = False
    assert astar((3, 0), (3, 6), blocked) is None

    # float / np.int inputs should also work (astar casts to int internally)
    assert astar((0.0, 0.0), (np.int64(2), np.int64(2)), free) is not None
    print("[1/3] hand-crafted grids: all assertions passed")


# --------------------------------------------------------------------------- #
# 2) Random masks — optimality vs Dijkstra + reachability vs BFS
# --------------------------------------------------------------------------- #
def test_random(n_masks=20, pairs_per_mask=20, seed=0):
    rng = np.random.default_rng(seed)
    n_paths = n_none = 0
    for _ in range(n_masks):
        H, W = int(rng.integers(5, 15)), int(rng.integers(5, 15))
        mask = rng.random((H, W)) > 0.35                  # ~35% obstacles
        if not mask.any():
            continue
        for _ in range(pairs_per_mask):
            s = random_valid_cell(mask, rng)
            g = random_valid_cell(mask, rng)
            p = astar(s, g, mask)
            reachable = g in bfs_component(s, mask)
            if p is None:
                n_none += 1
                assert not reachable, f"astar returned None but {s}->{g} is reachable"
            else:
                n_paths += 1
                assert reachable, f"astar found a path but BFS says {s}->{g} unreachable"
                assert_path_valid(p, s, g, mask)
                ref = dijkstra_cost(s, g, mask)
                assert abs(path_cost(p) - ref) < 1e-9, \
                    f"suboptimal: astar={path_cost(p):.6f} dijkstra={ref:.6f} ({s}->{g})"
    print(f"[2/3] random masks: {n_paths} paths optimal vs Dijkstra, "
          f"{n_none} unreachable pairs consistent with BFS")


# --------------------------------------------------------------------------- #
# 3) Real ATC mask — invariants + figure for visual inspection
# --------------------------------------------------------------------------- #
def test_real_data(file, outdir, seed=0, n_pairs=50, n_plot=4):
    from observation_model import load_state, resolve_file
    import navigation
    X, _ = load_state(resolve_file(file))
    mask = navigation.build_valid_mask_from_config(X)     # the mask the pipeline actually uses (hybrid)
    print(f"[load] walkable {int(mask.sum())}/{mask.size} cells (source per config.yaml)")

    rng = np.random.default_rng(seed)
    paths = []
    for _ in range(n_pairs):
        s, g = random_valid_cell(mask, rng), random_valid_cell(mask, rng)
        p = astar(s, g, mask)
        assert (p is not None) == (g in bfs_component(s, mask)), f"reachability mismatch {s}->{g}"
        if p is not None:
            assert_path_valid(p, s, g, mask)
            assert abs(path_cost(p) - dijkstra_cost(s, g, mask)) < 1e-9, f"suboptimal on ATC mask {s}->{g}"
            paths.append(p)
    print(f"[3/3] ATC mask: {len(paths)}/{n_pairs} pairs connected, all paths optimal & valid")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        os.makedirs(outdir, exist_ok=True)
        H, W = mask.shape                                 # 36x12: tall narrow grid, render at true aspect (square cells)
        fig, ax = plt.subplots(figsize=(W * 0.35 + 1.5, H * 0.35 + 1.0))
        ax.imshow(np.where(mask, 1, 0), cmap=ListedColormap(["#9e9e9e", "#ffffff"]),
                  aspect="equal", origin="upper")
        for p in sorted(paths, key=len)[-n_plot:]:        # plot the longest few paths — wall detours show best
            arr = np.asarray(p)
            ax.plot(arr[:, 1], arr[:, 0], "o-", ms=3, lw=1.5)
            ax.scatter([arr[0, 1], arr[-1, 1]], [arr[0, 0], arr[-1, 0]],
                       c=["green", "red"], s=60, zorder=3)
        ax.set_title("A* paths on ATC walkable mask\n(green=start, red=goal, gray=wall)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        out = os.path.join(outdir, "astar_paths.png")
        fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
        print(f"[figure] {out}")
    except Exception as e:                                # never block the numeric check on plotting
        print(f"[figure] skipped ({type(e).__name__}: {e})")


def main():
    ap = argparse.ArgumentParser(description="Verify navigation.py (A* path planning)")
    ap.add_argument("--file", default="first", help="grid_cache path, date stem, or 'first'")
    ap.add_argument("--outdir", default="check_outputs/navigation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic-only", action="store_true", help="skip the real-data section")
    args = ap.parse_args()

    test_handcrafted()
    test_random(seed=args.seed)
    if not args.synthetic_only:
        test_real_data(args.file, args.outdir, seed=args.seed)
    print("\nall navigation checks passed")


if __name__ == "__main__":
    main()
