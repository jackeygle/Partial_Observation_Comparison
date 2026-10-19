"""Smoke test of the gap mask on the GPU scorers (route 2), on 2013-08-07 (first test day, 14% of
its frames excluded). Not a result: the checkpoints are whatever route-2 snapshots exist.

    DATA_VARIANT=all92 python3 -m supervisor_evaluation.checks.smoke_gapmask OUT_DIR SENSEIVER VARNET DINCAE_RUN

For compare5 (Senseiver + 4DVarNet), the DINCAE evaluator and 4DVarNet's score_day, run once with
the mask off and once on, and check: off scores every frame; on scores exactly the scorable frames;
the scores differ only through the dropped frames.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from crowdcore import frame_reliability as fr, navigation as nav, observation_model as om

out, sv, vn, drun = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
out.mkdir(parents=True, exist_ok=True)
fp = om.split_files("test")[0]
assert fr.day_stem(fp) == "atc-20130807", fp
import h5py
n = h5py.File(fp, "r")["grid"].shape[0]
fr.ENABLED = True
keep = fr.scorable_frames(fp, n)
print(f"[day] {fr.day_stem(fp)}: {n} frames, {int(keep.sum())} scorable ({100 * keep.mean():.1f}%)", flush=True)
ok = True


def check(name, cond, msg):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}: {msg}", flush=True)


def run(mod, args, mask):
    env = dict(os.environ, GAP_MASK=mask)
    subprocess.run([sys.executable, "-u", "-m", mod, *args], env=env, check=True)


# compare5: Senseiver-A + 4DVarNet, one day
res = {}
for mask in ("0", "1"):
    o = out / f"compare5_mask{mask}.json"
    run("compare.compare5", ["--senseiver", sv, "--varnet-checkpoints", f"MSE={vn}", "--arms", "",
                             "--no-with-ensemble", "--days", "1", "--out", str(o),
                             "--dincae-json", str(out / "none.json")], mask)
    res[mask] = json.loads(o.read_text())
d0, d1 = res["0"]["per_day"][0], res["1"]["per_day"][0]
lo = 1                                          # compare5's FRAME_LO
check("compare5 frames", d1["frames_scored"] == int(keep[lo:lo + d0["frames_scored"]].sum())
      and d1["frames_scored"] < d0["frames_scored"],
      f"off {d0['frames_scored']}, on {d1['frames_scored']}")
for k in ("Senseiver", "4DVarNet MSE"):
    print(f"  {k}: all-walkable RMSE off {res['0']['results'][k]['walkable_full']['rmse_pooled']:.5f} "
          f"on {res['1']['results'][k]['walkable_full']['rmse_pooled']:.5f}", flush=True)

# DINCAE evaluator, one day
dm = {}
for mask in ("0", "1"):
    o = out / f"dincae_mask{mask}"
    run("methods.dincae.checks.evaluate", ["--run-dir", drun, "--ckpt-glob", f"{drun}/last.pt",
                                           "--split", "test", "--days", "1", "--out", str(o)], mask)
    dm[mask] = json.loads((o / "dincae_metrics_test.json").read_text())
f0, f1 = dm["0"]["per_day"][0]["frames"], dm["1"]["per_day"][0]["frames"]
check("DINCAE frames", f0 == n - 2 and f1 == int(keep[1:n - 1].sum()), f"off {f0}, on {f1}")
print(f"  DINCAE all-walkable MSE off {dm['0']['walkable_all_mse_all_channels']:.5f} "
      f"on {dm['1']['walkable_all_mse_all_channels']:.5f}", flush=True)

# 4DVarNet score_day (selection / early stopping), in process
from methods.varnet.checks.model_io import load_solver
from methods.varnet.checks.select_checkpoint import score_day
sol, args, _ = load_solver(vn, "cuda"); sol.requires_grad_(False)
walk = nav.build_valid_mask_from_config()
r = {}
for mask in (False, True):
    fr.ENABLED = mask
    r[mask] = score_day({0: sol}, fp, {vn: args}, walk, "cuda", 8, "all_walkable")[0]
nw = n // 200
check("score_day cells", r[False][1] == nw * 200 * 4 * int(walk.sum())
      and r[True][1] == int(keep[:nw * 200].sum()) * 4 * int(walk.sum()),
      f"off {r[False][1]}, on {r[True][1]}")
print(f"  score_day MSE off {r[False][0] / r[False][1]:.5f} on {r[True][0] / r[True][1]:.5f}", flush=True)
print("[smoke] ALL PASS" if ok else "[smoke] FAILURES", flush=True)
sys.exit(0 if ok else 1)
