"""
diag_error_split.py  —  where does the blind-zone error actually live?

For a trained checkpoint, splits the blind-cell MSE of every channel by whether the
TRUE density in that cell is zero (empty) or non-zero (occupied), and reports how much
of the total error each bucket contributes. This decides what a loss change should
target: if the velocity error sits in EMPTY cells the model is inventing motion where
there is nobody (a weighting/focal problem); if it sits in OCCUPIED cells the velocity
values themselves are wrong (a capacity/prior problem).

Run on a GPU node:
    python3 checks/diag_error_split.py --ckpt runs/varnet_b0_k1/varnet_best.pt
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import config, navigation as nav, observation_model as om
from model_io import load_solver

CH = ["density", "vx", "vy", "var"]


def build_solver(ckpt, dev):
    s, a, _ = load_solver(ckpt, dev)
    return s, a


def clip_bounds(x):                                     # identical to eval_test_days.py
    x = x.copy()
    x[:, 0] = np.clip(x[:, 0], 0, 5); x[:, 1] = np.clip(x[:, 1], -5, 5)
    x[:, 2] = np.clip(x[:, 2], -5, 5); x[:, 3] = np.clip(x[:, 3], 0, 2)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/varnet_b0_k1/varnet_best.pt")
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--occ-thresh", type=float, default=0.0,
                    help="a cell counts as OCCUPIED if true density > this (default: >0)")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    solver, a = build_solver(args.ckpt, dev); dT = a["dT"]

    # accumulate squared error and cell counts, split by occupancy of the true density
    se_e = np.zeros(4); n_e = np.zeros(4)      # empty  cells
    se_o = np.zeros(4); n_o = np.zeros(4)      # occupied cells
    for day in om.split_files("test"):
        X, _ = om.load_state(day); X = np.asarray(X)[:args.frames]
        out = om.generate_observations(X, add_noise=True, valid_mask=nav.build_valid_mask_from_config(X))
        x0 = om.fill_missing_state(out["Y"], out["Omega_c"], method=config.get("observation", "init_method"))
        n = (X.shape[0] // dT) * dT
        win = lambda arr: om.to_windows(arr, dT)
        with torch.enable_grad():
            xr = solver(torch.from_numpy(win(x0)).float().to(dev),
                        torch.from_numpy(win(out["Y"])).float().to(dev),
                        torch.from_numpy(win(out["Omega_c"].astype(np.float32))).float().to(dev)
                        ).detach().cpu().numpy()
        xr = clip_bounds(om.from_windows(xr))
        Xn = X[:xr.shape[0]]
        unobs = ~out["Omega_c"][:xr.shape[0]].astype(bool)
        occ = Xn[:, 0] > args.occ_thresh                  # (N,H,W) true density non-zero
        for c in range(4):
            u = unobs[:, c]
            sq = (xr[:, c] - Xn[:, c]) ** 2
            se_e[c] += sq[u & ~occ].sum(); n_e[c] += (u & ~occ).sum()
            se_o[c] += sq[u & occ].sum();  n_o[c] += (u & occ).sum()
        print(f"  {os.path.basename(day)}: done", flush=True)

    tot_se = se_e.sum() + se_o.sum()
    print(f"\n{args.ckpt}   blind cells, split by TRUE density > {args.occ_thresh}")
    print(f"occupied cells = {n_o[0]/(n_o[0]+n_e[0])*100:.1f}% of blind cells\n")
    print(f"{'channel':8s} {'MSE_empty':>10} {'MSE_occ':>10} {'%err_empty':>11} {'%err_occ':>9}")
    for c in range(4):
        me = se_e[c]/max(n_e[c], 1); mo = se_o[c]/max(n_o[c], 1)
        print(f"{CH[c]:8s} {me:>10.4f} {mo:>10.4f} {se_e[c]/tot_se*100:>10.1f}% {se_o[c]/tot_se*100:>8.1f}%")
    print(f"\ntotal error share:  empty cells {se_e.sum()/tot_se*100:.1f}%   "
          f"occupied cells {se_o.sum()/tot_se*100:.1f}%")


if __name__ == "__main__":
    main()
