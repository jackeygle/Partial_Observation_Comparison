"""
config.py  —  load config.yaml (the single source of truth for all parameters)
==============================================================================

Usage:
    import config
    config.CFG["observation"]["sensing_range"]      # dict access
    config.get("observation", "sensing_range")      # path access with clear errors

Design: parameters live in config.yaml (documented, versioned); modules read
their constants from here; train_varnet.py uses these as argparse DEFAULTS so a
single run can still override anything on the command line. The original project
kept the same idea in Partial_observation/Parameters.yaml.
"""

from __future__ import annotations

import os

import yaml

_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(_ROOT, "config.yaml")

with open(CONFIG_PATH) as _f:
    CFG = yaml.safe_load(_f)

#: DATA_VARIANT=all92 (route 2, 2026-10-12): all 92 ATC days, Wednesdays and Sundays, with the
#: same chronological boundaries as the Sunday split. It has its own data root (split lists,
#: grid_cache, the walkable-mask cache) and every artifact fitted on the training days carries
#: VARIANT_SUFFIX, so nothing of the Sunday-only setup is read or overwritten. Unset, nothing
#: changes.
DATA_VARIANT = os.environ.get("DATA_VARIANT", "")
if DATA_VARIANT == "all92":
    CFG["data"]["root"] = "/scratch/work/zhangx29/data_all92"
    CFG["data"]["split_list"] = "atc_all_{split}.lst"
elif DATA_VARIANT:
    raise SystemExit(f"unknown DATA_VARIANT {DATA_VARIANT!r}")
VARIANT_SUFFIX = "_all92" if DATA_VARIANT == "all92" else ""


def get(*keys, default=None):
    """config.get('observation', 'sensing_range') -> 7; raises with the failing key when the path does not exist."""
    node = CFG
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            if default is not None:
                return default
            raise KeyError(f"config.yaml missing key: {'.'.join(keys)} (failed at {k!r})")
        node = node[k]
    return node
