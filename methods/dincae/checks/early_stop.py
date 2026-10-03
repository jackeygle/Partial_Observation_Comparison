"""early_stop.py — score a DINCAE run's new checkpoints on the validation split and decide
whether training should stop (supervisor_evaluation/RESELECTION_PROTOCOL.md).

    python3 -m methods.dincae.checks.early_stop --run-dir runs/dincae_reselect_obsloss

Same scoring as checks/select_checkpoint.py (blind walkable cells, clipped, all validation
days, the run's own routes), cached per checkpoint in <run>/early_stop.json. Writes the
selection to check_outputs/eval/select_<run>_valid.json, in select_checkpoint.py's format,
and exits 3 once the best checkpoint is at least --patience epochs older than the newest.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch

from crowdcore import observation_model as om
from methods.dincae.checks.select_checkpoint import ROOT, score_ckpt
from methods.dincae.state import StateStats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--min-epoch", type=int, default=30)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    run_dir = a.run_dir if os.path.isabs(a.run_dir) else os.path.join(ROOT, a.run_dir)
    cache_p = os.path.join(run_dir, "early_stop.json")
    cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {"scores": {}}
    todo = []
    for p in sorted(glob.glob(os.path.join(run_dir, "ckpt_*.pt"))):
        ep = int(re.search(r"ckpt_(\d+)\.pt$", p).group(1))
        if ep >= a.min_epoch and str(ep) not in cache["scores"]:
            todo.append((ep, p))
    if todo:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        stats, files = StateStats(), om.split_files("valid")
        for ep, p in todo:
            mse, _ = score_ckpt(p, stats, files, dev, 0, a.batch)
            cache["scores"][str(ep)] = mse
            print(f"[early_stop] epoch {ep}: valid blind walkable RMSE {np.sqrt(mse):.5f}",
                  flush=True)
            json.dump(cache, open(cache_p, "w"), indent=1)
    if not cache["scores"]:
        print(f"[early_stop] no checkpoint at or after epoch {a.min_epoch} yet: keep training")
        return 0
    eps = sorted(int(e) for e in cache["scores"])
    best = min(eps, key=lambda e: cache["scores"][str(e)])
    stop = eps[-1] - best >= a.patience
    cache.update(min_epoch=a.min_epoch, patience=a.patience, best_epoch=best,
                 newest_epoch=eps[-1], stop=stop)
    json.dump(cache, open(cache_p, "w"), indent=1)
    rows = [{"epoch": e, "ckpt": f"ckpt_{e:05d}.pt", "walkable_blind_mse": cache["scores"][str(e)]}
            for e in eps]
    out = os.path.join(ROOT, "check_outputs", "eval", f"select_{os.path.basename(run_dir)}_valid.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"run_dir": run_dir, "split": "valid", "scope": "walkable blind",
               "n_days": len(om.split_files("valid")), "rows": rows,
               "best": rows[eps.index(best)], "early_stopped": stop}, open(out, "w"), indent=2)
    print(f"[early_stop] best epoch {best} (RMSE {np.sqrt(cache['scores'][str(best)]):.5f}), "
          f"newest {eps[-1]}: {'STOP' if stop else 'continue'}", flush=True)
    return 3 if stop else 0


if __name__ == "__main__":
    sys.exit(main())
