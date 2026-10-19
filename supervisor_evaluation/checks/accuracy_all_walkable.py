"""Per-channel RMSE on ALL walkable cells (observed and blind) from an evaluation's raw outputs.

    python3 -m supervisor_evaluation.checks.accuracy_all_walkable <output_dir>

Reads the same raw files as evaluate.collect (no model is run) and writes
<output_dir>/accuracy_all_walkable.csv. The pooled column must equal accuracy.csv's
all_walkable_rmse; the script stops if it does not.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

from crowdcore import frame_reliability as fr

CH = ("density", "vx", "vy", "variance")
out = Path(sys.argv[1]); raw = out / "raw"
acc = {r["method"]: r for r in csv.DictReader(open(out / "accuracy.csv"))}
rows = []


def add(method, per_channel_mse, pooled_mse):
    rows.append({"method": method, "all_walkable_rmse": float(np.sqrt(pooled_mse)),
                 **{f"{c}_rmse": float(np.sqrt(per_channel_mse[i])) for i, c in enumerate(CH)}})


for label, fname, key in (("Senseiver-A", "senseiver_a_accuracy.json", "Senseiver"),
                          ("Senseiver-G Temporal (ours)", "senseiver_g_accuracy.json", "Senseiver"),
                          ("4DVarNet MSE", "varnet_mse_accuracy.json", "4DVarNet MSE")):
    w = json.loads((raw / fname).read_text())["results"][key]["walkable_full"]
    pc = [w["per_channel"]["var" if c == "variance" else c] for c in CH]
    add(label, pc, w["rmse_pooled"] ** 2)
if (raw / "varnet_aughead_accuracy.json").exists():        # six-method evaluation
    w = json.loads((raw / "varnet_aughead_accuracy.json").read_text())["results"][
        "4DVarNet aughead_obs"]["walkable_full"]
    add("4DVarNet aughead_obs (single)",
        [w["per_channel"]["var" if c == "variance" else c] for c in CH], w["rmse_pooled"] ** 2)
d = json.loads((raw / "dincae_accuracy" / "dincae_metrics_test.json").read_text())
add("DINCAE", [d["walkable_all_mse"]["var" if c == "variance" else c] for c in CH],
    d["walkable_all_mse_all_channels"])
se = np.zeros(4); n = 0
for fp in sorted((raw / "enkf_exports").glob("atc-*.npz")):          # as evaluate.collect
    with np.load(fp) as z:
        mu, truth, walk = z["mean"], z["truth"], z["walkable"].astype(bool)
    warmup = int(json.loads((raw / "enkf" / f"{fp.stem}.json").read_text())["config"]["warmup"])
    keep = fr.scorable_after_warmup(fp, warmup, len(mu))      # tracker-gap blocks left out
    if not keep.all():
        mu, truth = mu[keep], truth[keep]
    delta = np.square(mu - truth)
    se += np.array([delta[:, c][:, walk].sum(dtype=np.float64) for c in range(4)])
    n += delta.shape[0] * int(walk.sum())
add("Localized EnKF (100 members)", se / n, se.sum() / (4 * n))

order = list(acc)
rows.sort(key=lambda r: order.index(r["method"]))
for r in rows:
    ref = float(acc[r["method"]]["all_walkable_rmse"])
    if abs(r["all_walkable_rmse"] - ref) > 1e-9:
        raise SystemExit(f"{r['method']}: pooled {r['all_walkable_rmse']} != accuracy.csv {ref}")
with open(out / "accuracy_all_walkable.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(f"{r['method'][:28]:28s} " + " ".join(f"{r[k]:.4f}" for k in r if k != "method"))
