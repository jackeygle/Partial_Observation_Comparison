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
