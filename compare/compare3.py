"""
compare3.py — three-way per-channel comparison (Senseiver / 4DVarNet / EnKF),
full day, k=1, same clipping
=================================================================================

Why per-channel: the four channels differ by an order of magnitude, and a single
MSE is dominated by vx (in Senseiver's blind error, vx is 0.079 while density is
only 0.014) -- how well density is reconstructed gets completely hidden. The
header of `4dvarnet_enkf/checks/compare_channels.py` complains about the same
thing.

Convention (identical across all three methods):
  * All 7 held-out days, **full day** (not the 400-frame subset)
  * obs_every_k = 1
  * Blind = unobserved (channel, cell) pairs, same as eval_test_days.py's `mask<0.5`
  * All three methods get the EnKF's physical clipping applied (the EnKF is
    already clipped, so reapplying it is idempotent)

The EnKF's estimate is taken from `4dvarnet_enkf/check_outputs/enkf_k1_full/`
(the full-day k=1 set). Note `check_outputs/enkf/` is **k=4 with only 400
frames** -- `enkf_metrics.json`'s 0.0392 comes from there and **must not** be
placed side by side with the full-day numbers.

This script only reads 4dvarnet_enkf, never writes to any of its directories.

**This convention excludes DINCAE**: it is scored on all cells, where 88% of the
velocity channels' cells are placeholder 0s for empty cells, and DINCAE was never
trained on those cells. For the four-way comparison (including DINCAE), see
`compare4.py`, which brings the same three methods onto the "channel-defined
cells" convention and lists this script's convention alongside it.

Usage (GPU node):
    sbatch sbatch/submit_eval.sbatch --help   # see the argparse below
    srun -p gpu-debug --gres=gpu:1 -t 00:14:00 --mem=64G bash -c \
      'module load scicomp-pytorch-env/2026.1; python3 -u checks/compare3.py --days 2'
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from crowdcore import paths
import sys

import numpy as np
import torch

from methods.senseiver import dataset as ds
from methods.senseiver import sensors
from methods.senseiver.network import Senseiver

from methods.varnet.checks.model_io import load_solver                                    # noqa: E402

LO = np.array([0.0, -5.0, -5.0, 0.0], np.float32)
HI = np.array([5.0, 5.0, 5.0, 2.0], np.float32)


def clip_np(x):
    return np.clip(x, LO[None, :, None, None], HI[None, :, None, None])


class Acc:
    """Accumulates squared blind-cell error per channel."""

    def __init__(self, C):
        self.se = np.zeros(C); self.n = np.zeros(C)

    def add(self, pred, true, obs):
        """pred/true (T,C,H,W) np, obs (T,H,W) bool"""
        d2 = (pred - true) ** 2
        blind = ~obs
        for c in range(d2.shape[1]):
            self.se[c] += d2[:, c][blind].sum(); self.n[c] += blind.sum()

    def mse(self):
        return self.se / np.maximum(self.n, 1)

    def overall(self):
        return self.se.sum() / max(self.n.sum(), 1)


def run_senseiver(model, Y, Om, dev, batch, C, H, W):
    pe = model.pos_enc.cpu().numpy(); mu = model.in_mean.cpu().numpy(); sd = model.in_std.cpu().numpy()
    outs = []
    with torch.no_grad():
        for i in range(0, Y.shape[0], batch):
            sl = slice(i, i + batch)
            tok, pad, _ = sensors.build_batch(Y[sl], Om[sl], pe, mu, sd)
            outs.append(model.reconstruct(tok.to(dev), pad.to(dev)).cpu().numpy())
    return np.concatenate(outs, 0)


def run_varnet(solver, X, Y, Omc, X0, dT, dev, batch):
    from crowdcore import observation_model as om
    win = lambda a: om.to_windows(a, dT)
    Xw, Yw = win(X), win(Y)
    Mw, X0w = win(Omc.astype(np.float32)), win(X0)
    outs = []
    with torch.enable_grad():
        for i in range(0, Xw.shape[0], batch):
            xb = torch.from_numpy(X0w[i:i + batch]).float().to(dev)
            yb = torch.from_numpy(Yw[i:i + batch]).float().to(dev)
            mb = torch.from_numpy(Mw[i:i + batch]).float().to(dev)
            outs.append(solver(xb, yb, mb).detach().cpu().numpy())
    r = np.concatenate(outs, 0)                          # (nw, C, dT, H, W)
    nw, C, dt, H, W = r.shape
    return r.transpose(0, 2, 1, 3, 4).reshape(nw * dt, C, H, W), Xw.shape[0] * dT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--senseiver", default="runs/senseiver_A/best.pt")
    ap.add_argument("--varnet", default="a4_k1")
    ap.add_argument("--enkf-dir", default=paths.enkf_export("enkf_k1_full"))
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--varnet-batch", type=int, default=16)
    ap.add_argument("--out", default="check_outputs/eval/compare3.json")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C, H, W = ds.state_shape(); chans = ds.channels()

    sck = torch.load(args.senseiver, map_location=dev, weights_only=False)
    sm = Senseiver(**sck["hparams"]).to(dev); sm.load_state_dict(sck["model"]); sm.eval()
    solver, va, _ = load_solver(os.path.join(paths.runs(paths.VARNET), f"varnet_{args.varnet}", "varnet_best.pt"), dev)
    print(f"[model] Senseiver {sm.num_params:,} params | 4DVarNet {args.varnet} "
          f"dT={va['dT']} n_iter={solver.n_iter} | device={dev}", flush=True)

    days = ds.om.split_files("test")
    if args.days:
        days = days[:args.days]
    accs = {"Senseiver": Acc(C), f"4DVarNet {args.varnet}": Acc(C), "EnKF k1": Acc(C)}

    for d in days:
        stem = os.path.basename(d).split("_")[0]
        X, Y, Om = ds.load_day(d, stride=1, seed=0)
        n = X.shape[0]
        Xf = X.reshape(n, C, H, W); Yf = Y.reshape(n, C, H, W); Omf = Om.reshape(n, H, W)

        p = clip_np(run_senseiver(sm, Y, Om, dev, args.batch, C, H, W))
        accs["Senseiver"].add(p, Xf, Omf)

        Omc = np.repeat(Omf[:, None], C, axis=1)
        X0 = ds.om.fill_missing_state(Yf, Omc, method=ds.obs_config()["init_method"])
        pv, nkeep = run_varnet(solver, Xf, Yf, Omc, X0, va["dT"], dev, args.varnet_batch)
        accs[f"4DVarNet {args.varnet}"].add(clip_np(pv), Xf[:nkeep], Omf[:nkeep])

        ep = os.path.join(args.enkf_dir, f"est_{stem}.npz")
        if os.path.exists(ep):
            est = np.load(ep)["Est"].astype(np.float32)
            z = np.load(os.path.join(args.enkf_dir, f"obs_{stem}.npz"))
            m = est.shape[0]
            accs["EnKF k1"].add(clip_np(est), z["X_true"][:m].astype(np.float32),
                                z["Omega"][:m].astype(bool))
        print(f"  {stem}: {n} frames done", flush=True)

    print(f"\nBlind MSE (full day, k=1, same clipping, {len(days)} held-out days)\n")
    print(f"{'method':<18}" + "".join(f"{c:>11}" for c in chans) + f"{'total':>11}")
    print("-" * (18 + 11 * (C + 1)))
    rows = {}
    for k, a in accs.items():
        if a.n.sum() == 0:
            continue
        rows[k] = {"per_channel": {chans[i]: float(v) for i, v in enumerate(a.mse())},
                   "overall": float(a.overall())}
        print(f"{k:<18}" + "".join(f"{v:>11.4f}" for v in a.mse()) + f"{a.overall():>11.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"protocol": "full-day, obs_every_k=1, EnKF clip bounds, blind = ~Omega",
                   "n_days": len(days), "channels": chans, "results": rows}, f, indent=2)
    print(f"\n[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
