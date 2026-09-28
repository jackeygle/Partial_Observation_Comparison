"""
kalman_contribution.py — how much does the Kalman update actually change the estimate?

For one day and one filter configuration, compares the forecast (before the update)
with the analysis (after it):

  * absorption: on observed cells, the fraction of the innovation (observation minus
    forecast mean) that the update moves the mean by,
        sum((analysis - forecast) * (y - forecast)) / sum((y - forecast)^2)
    0 = the observation is ignored, 1 = the analysis copies the observation;
  * error reduction: RMSE against the truth before and after the update, on observed
    walkable cells and on unobserved (blind) walkable cells.

Frames >= warmup, per channel and pooled.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

CH = ("density", "vx", "vy", "variance")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecast", required=True, help="--forecast-calibration-export npz")
    ap.add_argument("--analysis", required=True, help="--calibration-export npz")
    ap.add_argument("--obs", required=True, help="obs_<day>.npz given to the filter")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    f, an = np.load(a.forecast), np.load(a.analysis)
    xf, xa, truth = f["mean"].astype(np.float64), an["mean"].astype(np.float64), f["truth"]
    obs_mask, walk = f["observed"].astype(bool), f["walkable"].astype(bool)
    with np.load(a.obs) as z:
        y = z["Y"][a.warmup:a.warmup + len(xf)].astype(np.float64)
    assert np.allclose(an["truth"], truth) and y.shape == xf.shape
    observed = obs_mask & walk[None]
    blind = ~obs_mask & walk[None]
    res = {"label": a.label, "day": str(f["day"]), "frames": int(len(xf))}
    for key, idx in (("all", slice(None)),) + tuple((c, i) for i, c in enumerate(CH)):
        sel_o = np.broadcast_to(observed[:, None], xf.shape)[:, idx]
        sel_b = np.broadcast_to(blind[:, None], xf.shape)[:, idx]
        d = (y - xf)[:, idx][sel_o]
        inc = (xa - xf)[:, idx][sel_o]
        r = {"absorption": float((inc * d).sum() / (d * d).sum()),
             "forecast_spread_observed": float(f["spread"][:, idx][sel_o].mean())}
        for name, sel in (("observed", sel_o), ("blind", sel_b)):
            ef = (xf - truth)[:, idx][sel]; ea = (xa - truth)[:, idx][sel]
            rf, ra = float(np.sqrt((ef ** 2).mean())), float(np.sqrt((ea ** 2).mean()))
            r[f"rmse_forecast_{name}"], r[f"rmse_analysis_{name}"] = rf, ra
            r[f"error_reduction_{name}"] = 1 - ra / rf
        res[key] = r
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"[{a.label}] {res['day']}  frames {res['frames']}")
    print(f"{'':10}{'absorb':>8}{'fc spread':>11}{'obs: fc->an RMSE':>22}{'cut':>7}{'blind: fc->an RMSE':>22}{'cut':>7}")
    for key in ("all",) + CH:
        r = res[key]
        print(f"{key:10}{r['absorption']:8.1%}{r['forecast_spread_observed']:11.4f}"
              f"{r['rmse_forecast_observed']:12.4f} ->{r['rmse_analysis_observed']:7.4f}{r['error_reduction_observed']:7.1%}"
              f"{r['rmse_forecast_blind']:12.4f} ->{r['rmse_analysis_blind']:7.4f}{r['error_reduction_blind']:7.1%}")


if __name__ == "__main__":
    main()
