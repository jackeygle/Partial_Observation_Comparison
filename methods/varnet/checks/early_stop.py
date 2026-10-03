"""early_stop.py — score a 4DVarNet run's new checkpoints on the validation split, and
decide whether training should stop (supervisor_evaluation/RESELECTION_PROTOCOL.md).

    python3 -m methods.varnet.checks.early_stop --run-dir runs/varnet_reselect_h32k3_s0

Scores every ckpt_<epoch>.pt at or after --min-epoch that is not yet in the run's
early_stop.json, with exactly the scoring of checks/select_checkpoint.py (blind walkable
cells, clipped, all validation days, the run's own observation seed and routes), and
caches the result. It then writes select_valid.json, so the selected checkpoint is
resolved by model_io.reported_ckpt like every other run, and exits with

    0  keep training
    3  stop: the best checkpoint is at least --patience epochs older than the newest

A training job calls this before it resumes, and does not resume (nor queue a
successor) on exit code 3.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

from crowdcore import navigation as nav
from crowdcore import observation_model as om
from methods.varnet.checks.model_io import load_solver
from methods.varnet.checks.select_checkpoint import FINAL_N_ITER, ROOT, candidates, score_day


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--min-epoch", type=int, default=80)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    run_dir = a.run_dir if os.path.isabs(a.run_dir) else os.path.join(ROOT, a.run_dir)
    cache_p = os.path.join(run_dir, "early_stop.json")
    cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {"scores": {}}
    cands = [(ep, p) for ep, p in candidates(run_dir, a.min_epoch)
             if str(ep) not in cache["scores"]]
    if cands:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        solvers, args_by_ckpt = {}, {}
        for ep, p in cands:
            sol, args, ck = load_solver(p, dev)
            if int(ck.get("n_iter_eff", -1)) != FINAL_N_ITER:
                raise SystemExit(f"{p} solves with {ck.get('n_iter_eff')} iterations, "
                                 f"not {FINAL_N_ITER}; raise --min-epoch")
            solvers[ep] = sol
            args_by_ckpt[p] = args
        walk = nav.build_valid_mask_from_config()
        files = om.split_files("valid")
        se = {ep: 0.0 for ep in solvers}
        n = {ep: 0 for ep in solvers}
        for fp in files:
            for ep, (s, c) in score_day(solvers, fp, args_by_ckpt, walk, dev, a.batch).items():
                se[ep] += s
                n[ep] += c
        for ep in solvers:
            cache["scores"][str(ep)] = se[ep] / max(n[ep], 1)
            print(f"[early_stop] epoch {ep}: valid blind walkable RMSE "
                  f"{np.sqrt(cache['scores'][str(ep)]):.5f}", flush=True)
        cache["days"] = [os.path.basename(x) for x in files]
    if not cache["scores"]:
        print(f"[early_stop] no checkpoint at or after epoch {a.min_epoch} yet: keep training")
        return 0
    eps = sorted(int(e) for e in cache["scores"])
    best = min(eps, key=lambda e: cache["scores"][str(e)])
    newest = eps[-1]
    stop = newest - best >= a.patience
    cache.update(min_epoch=a.min_epoch, patience=a.patience, best_epoch=best,
                 newest_epoch=newest, stop=stop)
    json.dump(cache, open(cache_p, "w"), indent=1)
    json.dump({"run_dir": run_dir, "split": "valid", "min_epoch": a.min_epoch,
               "n_iter": FINAL_N_ITER, "days": cache.get("days", []),
               "scope": "walkable (blind cells inside the walkable region, clipped)",
               "mse_by_epoch": cache["scores"], "selected_epoch": best,
               "selected_mse": cache["scores"][str(best)],
               "selected_ckpt": f"ckpt_{best:05d}.pt", "early_stopped": stop},
              open(os.path.join(run_dir, "select_valid.json"), "w"), indent=2)
    print(f"[early_stop] best epoch {best} (RMSE {np.sqrt(cache['scores'][str(best)]):.5f}), "
          f"newest {newest}: {'STOP' if stop else 'continue'}", flush=True)
    return 3 if stop else 0


if __name__ == "__main__":
    sys.exit(main())
