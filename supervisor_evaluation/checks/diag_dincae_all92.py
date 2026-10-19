"""Route-2 diagnostic: DINCAE's dev residual MSE on one Wednesday and one Sunday validation day,
for a given checkpoint, with the statistics of the current DATA_VARIANT.

    DATA_VARIANT=all92|'' DINCAE_VAR_TRANSFORM=none python3 -m supervisor_evaluation.checks.diag_dincae_all92 --ckpt X
Prints, per day, the residual MSE per channel in physical units (as train.py's dev metric),
the largest |error| in units of the channel std, and the share of values above 10 std.
"""
import argparse, os
import numpy as np, torch
from methods.dincae.state import StateStats, CHANNELS, NCH
from methods.dincae.dataset import ChunkedDays
from methods.dincae.losses import decode_target
from methods.dincae.checks.evaluate import load_models

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); a = ap.parse_args()
dev = torch.device("cuda")
stats = StateStats()
model = load_models(os.path.dirname(a.ckpt), a.ckpt, dev)[0][0]
model.eval()
days = {"Wed 20130612": "/scratch/work/zhangx29/data_all92/grid_cache/atc-20130612_corridor_1.0s.h5",
        "Sun 20130616": "/scratch/work/zhangx29/data/grid_cache/atc-20130616_corridor_1.0s.h5"}
for name, fp in days.items():
    se = np.zeros(NCH); n = np.zeros(NCH); worst = np.zeros(NCH); big = np.zeros(NCH)
    with torch.no_grad():
        for xb, tb in ChunkedDays([fp], stats, batch_size=64, stride=16, shuffle=False, seed=0,
                                  cache_dir=None, full_field=True):
            xb, tb = xb.to(dev), tb.to(dev)
            m_rec, _ = model(xb)[-1]
            m_true, _, mask = decode_target(tb)
            err = ((m_rec - m_true) * mask).abs()
            se += ((err ** 2).sum(dim=(0, 2, 3))).cpu().numpy()
            n += mask.sum(dim=(0, 2, 3)).cpu().numpy()
            worst = np.maximum(worst, err.amax(dim=(0, 2, 3)).cpu().numpy())
            big += ((err > 10) & (mask > 0)).sum(dim=(0, 2, 3)).cpu().numpy()
    mse = se / np.maximum(n, 1) * stats.std.astype(np.float64) ** 2
    print(f"{name}: " + "  ".join(f"{c} mse {m:.4g} worst {w:.1f}sd >10sd {b/max(k,1)*100:.3f}%"
                                  for c, m, w, b, k in zip(CHANNELS, mse, worst, big, n)), flush=True)
