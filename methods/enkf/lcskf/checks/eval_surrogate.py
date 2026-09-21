"""
eval_surrogate.py — does our retrained PedPred3 forecast as well as the vendored one?

The reproduction gate before training all ten networks. The original train.py instantiates
PedPred2 while the vendored checkpoint (apt-ibex_train_model_28D.pth) is a PedPred3, so the
recipe behind that checkpoint cannot be recovered from code; this measures instead. One-step
forecast x_t -> x_{t+1} on the VALID split, per channel RMSE over all cells, walkable cells and
occupied cells (true density > 0 -- the only cells where the original loss weights velocity and
variance), for the vendored model, each of our mean-arm members, and their ensemble mean.

    source sbatch/_env.sh && cd methods/enkf
    python3 -u -m methods.enkf.lcskf.checks.eval_surrogate --members "runs/surrogate_mean_s0/best.pt"
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import torch

from crowdcore import navigation as nav
from crowdcore import paths
from methods.enkf.lcskf.dynamics.data import PairFrames
from methods.enkf.lcskf.dynamics.model import CH, check_vendor_forward, load_pedpred3, mean_forecast

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # methods/enkf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="runs/surrogate_mean_s*/best.pt", help="glob of mean-arm checkpoints")
    ap.add_argument("--original", default=os.path.join(paths.enkf_vendor("enkf_lab"), "apt-ibex_train_model_28D.pth"))
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(HERE, "check_outputs", "eval", "surrogate_valid.json"))
    ap.add_argument("--allow-cpu", action="store_true")
    a = ap.parse_args()
    if not torch.cuda.is_available() and not a.allow_cpu:
        raise SystemExit("no GPU -- run on a GPU node; add --allow-cpu to force CPU")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    valid = PairFrames("valid", dev, a.days, a.max_frames)
    walk = torch.from_numpy(nav.build_valid_mask_from_config().astype(bool)).to(dev)
    members = sorted(glob.glob(a.members))
    nets = {"original (apt-ibex)": load_pedpred3(a.original, dev).eval()}
    nets |= {os.path.relpath(p, HERE): load_pedpred3(p, dev).eval() for p in members}
    check_vendor_forward(nets["original (apt-ibex)"], valid.batch(torch.arange(8, device=dev))[0])
    names = list(nets) + ([f"ensemble mean of {len(members)}"] if len(members) > 1 else [])
    scopes = ("all", "walkable", "occupied")
    se = {n: {s: torch.zeros(4, device=dev, dtype=torch.float64) for s in scopes} for n in names}
    count = {s: 0 for s in scopes}

    with torch.no_grad():
        for i in range(0, valid.n_pairs, a.batch):
            x, y = valid.batch(torch.arange(i, min(i + a.batch, valid.n_pairs), device=dev))
            preds = {n: mean_forecast(net, x) for n, net in nets.items()}
            if len(members) > 1:
                preds[names[-1]] = torch.stack([preds[os.path.relpath(p, HERE)] for p in members]).mean(0)
            occ = (y[:, :, 0:1] > 0).double()                       # (B,1,1,H,W), broadcast over channels
            for n, mu in preds.items():
                e2 = ((y - mu) ** 2).double()
                se[n]["all"] += e2.sum(dim=(0, 1, 3, 4))
                se[n]["walkable"] += e2[..., walk].sum(dim=(0, 1, 3))
                se[n]["occupied"] += (e2 * occ).sum(dim=(0, 1, 3, 4))
            count["all"] += len(x) * walk.numel()
            count["walkable"] += len(x) * int(walk.sum())
            count["occupied"] += int(occ.sum())

    res = {n: {s: {c: float(torch.sqrt(v[k] / count[s])) for k, c in enumerate(CH)} for s, v in d.items()}
           for n, d in se.items()}
    doc = {"split": "valid", "n_days": valid.n_days, "n_pairs": valid.n_pairs, "members": members, "rmse": res}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(doc, open(a.out, "w"), indent=2)

    for scope in scopes:
        print(f"\none-step RMSE, {scope} cells, valid ({valid.n_days} days, {valid.n_pairs:,} pairs)")
        print(f"{'model':<44}" + "".join(f"{c:>10}" for c in CH))
        for n in names:
            print(f"{n:<44}" + "".join(f"{res[n][scope][c]:>10.4f}" for c in CH))
    print(f"\n[out] {a.out}")


if __name__ == "__main__":
    main()
