"""
eval_uncertainty_dincae.py — bring DINCAE's native sigma^2 onto the same metrics
as the other three methods
==============================================================================

Why this script is needed
------------------
DINCAE is the **only method besides 4DVarNet that natively outputs a variance**
(the second slice of the information form IS the precision 1/sigma^2), but its own
`evaluate.py` only reports `calibration` (predicted-SD-binned vs. actual RMS) and
`var_retention` -- that is the geoscience-inpainting community's convention, and
matches none of table B's CRPS / spread-skill / coverage / constant-sigma
baseline. So DINCAE had always been missing from table B, while a rough estimate
from its calibration block already puts its spread/skill at ~0.84, the closest to
1.0 of any method. Without filling this gap, the conclusion "uncertainty built
natively into the model vs. bolted on afterwards as a read-out head" is missing
its most important data point.

Which space the scoring happens in
------------------
**Normalised residual space** -- that is where the Gaussian assumption lives.
`predict_day`'s returned `mu_n`/`sd_n` are the mean and std in this space; the
target is `(fwd_channel(X) - mean)/std`.

Not converted to physical space, because density/var go through log1p, a
**nonlinear** transform: in physical space DINCAE's predictive distribution is
log-normal rather than Gaussian, the closed-form CRPS no longer applies, and a
sampling estimate would need drawing several samples for each of 6e7 points. More
importantly, it isn't necessary -- see below.

How to compare across spaces (the full version is in
`compare/score_uncertainty.py`'s module docstring)
--------------------------------------------------------------------
4DVarNet/EnKF are in physical space, DINCAE is in this transformed+standardised
space. So:

  * **CRPS absolute values are not comparable** (different units). Tables must
    state the space.
  * **CRPS / CRPS_const is comparable** -- numerator and denominator are in the
    same space, so the transform cancels. **The verdict column uses this.**
  * **spread/skill is comparable** -- likewise a ratio within the same space.
  * **coverage is strictly comparable** -- a monotonic transform preserves
    "whether the truth falls inside the interval," so log1p has no effect on it.

Why only the defined slice matters (don't use the all/blind rows)
------------------------------------------------------
Measured (ckpt_00070, 1 day): in normalised space, `channel_vy`'s all-cells RMSE
is **63.09**, while `defined_channel_vy` is only **1.02**. The difference is not
a bug -- `evaluate.py`'s physical metrics go through `clip_bounds` (physical
priors like density>=0, var in [0,2]), which is absent here, because clipping
would change mu without changing sigma, breaking the Gaussian assumption. So
DINCAE's unconstrained raw output on undefined cells goes straight into the
average, and a handful of extreme values dominate the whole slice.

This is itself the phenomenon table A records (DINCAE is 4.5x worse under the
all-cells convention), but it means the all/blind slices have no comparative
value for **uncertainty**. **The main table only uses `defined_*`**, the rest go
into the json for reference.

Main convention
------
`defined ∩ walkable ∩ blind`, the same cells as table A's main convention
(`channel_valid` ∩ `stats.valid`). all/observed and per-channel are also
reported, along with each slice's **own** constant-sigma baseline -- the baseline
must use that slice's own RMSE; using the full-field RMSE as the baseline for a
blind slice would put the metric and its baseline on two different cell sets.

Usage:
    source sbatch/_env.sh
    cd methods/dincae
    python3 -u -m methods.dincae.checks.eval_uncertainty_dincae \\
        --ckpt-glob "$PWD/runs/dincae_full/ckpt_00070.pt"
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from compare import score_uncertainty as su                          # noqa: E402
from crowdcore import observation_model as om                        # noqa: E402
from methods.dincae.checks.evaluate import load_models, predict_day   # noqa: E402
from methods.dincae.state import CHANNELS, NCH, StateStats, channel_valid, fwd_channel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.path.join(ROOT, "runs", "dincae_full"))
    ap.add_argument("--ckpt-glob", default="",
                    help="the glob is a **full path pattern**, not relative to "
                         "run-dir (matching evaluate.load_models's behaviour). "
                         "Leave empty to take every ckpt_*.pt under run-dir and "
                         "average their outputs.")
    ap.add_argument("--split", default="test", choices=["test", "valid"])
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default=os.path.join(ROOT, "check_outputs", "eval",
                                                  "uncertainty_dincae.json"))
    args = ap.parse_args()

    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cpu":
        print("!! no GPU -- do not run torch on the login node", flush=True)

    stats = StateStats()
    models, epochs, _ = load_models(args.run_dir, args.ckpt_glob, dev)
    print(f"[members] {len(models)} checkpoints: epochs {epochs}  dev={dev}", flush=True)

    # Slices: two conventions x three scopes, plus per-channel. The
    # constant-sigma baseline needs a second pass, so only the real metrics are
    # accumulated first.
    tags = ["all", "blind", "observed", "defined", "defined_blind", "defined_observed"]
    acc = {t: su.Accumulator() for t in tags}
    acc |= {f"channel_{c}": su.Accumulator() for c in CHANNELS}
    acc |= {f"defined_channel_{c}": su.Accumulator() for c in CHANNELS}
    # The baseline must use each slice's own RMSE, which is only known once the
    # run finishes -> accumulate each slice's squared-error sum and count
    files = om.split_files(args.split)[:args.days] if args.days else om.split_files(args.split)

    def slices(Xt, M):
        """Returns {tag: (NCH,n,H,W) bool}. cv/walk come from the same source as table A's main convention."""
        cv = channel_valid(Xt)                       # (NCH,n,H,W)
        walk = stats.valid[None][None]               # (1,1,H,W) -> broadcasts
        dfn = cv & walk
        blind = ~M.transpose(1, 0, 2, 3)             # (NCH,n,H,W)
        obs = ~blind
        one = np.ones_like(blind)
        return {"all": one, "blind": blind, "observed": obs,
                "defined": dfn, "defined_blind": dfn & blind, "defined_observed": dfn & obs}

    def day(fp):
        """One day -> (mu_n, sd, tgt, slices). Target moved to normalised space, same convention as encode_target."""
        Xt, _rec, mu_n, sd_n, M = predict_day(models, stats, fp, dev, args.frames, args.batch)
        tgt = np.empty_like(mu_n)
        for c in range(NCH):
            tgt[:, c] = ((fwd_channel(Xt[:, c].astype(np.float64), c)
                          - stats.mean[c][None]) / stats.std[c])
        return mu_n, np.maximum(sd_n, 1e-12), tgt, slices(Xt, M)   # floor on sd guards against log/divide-by-zero

    for fp in files:
        mu_n, sd, tgt, sl = day(fp)
        for t, m in sl.items():
            mm = m.transpose(1, 0, 2, 3) if m.shape[0] == NCH else m
            acc[t].add(mu_n[mm], sd[mm], tgt[mm])
        for c, nm in enumerate(CHANNELS):
            acc[f"channel_{nm}"].add(mu_n[:, c], sd[:, c], tgt[:, c])
            d = sl["defined_blind"][c]
            acc[f"defined_channel_{nm}"].add(mu_n[:, c][d], sd[:, c][d], tgt[:, c][d])
        print(f"  {os.path.basename(fp)}  {mu_n.shape[0]} frames", flush=True)
        del mu_n, sd, tgt, sl

    res = {t: a.result() for t, a in acc.items() if a.result()}

    # ---- Second pass: the constant-sigma null model. sigma = that slice's own
    # RMSE, so it is only known once the first pass finishes.
    #
    # Deliberately **rerunning inference** rather than caching the first pass's
    # arrays: each day is the full 39819 frames, and (mu, sd, tgt) as three
    # float32 arrays plus six bool masks is about 1.24 GB/day, 8.7 GB for seven
    # days. Rerunning inference costs only about 3 more minutes of GPU, cheaper
    # than carrying that memory around.
    base_tags = [t for t in ("all", "blind", "defined", "defined_blind") if t in res]
    bacc = {t: su.Accumulator() for t in base_tags}
    print("\n[pass 2] constant-sigma baseline, sigma = "
          + ", ".join(f"{t} {res[t]['rmse']:.4f}" for t in base_tags), flush=True)
    for fp in files:
        mu_n, _sd, tgt, sl = day(fp)
        for t in base_tags:
            m = sl[t]
            mm = m.transpose(1, 0, 2, 3) if m.shape[0] == NCH else m
            bacc[t].add(mu_n[mm], np.full(int(mm.sum()), res[t]["rmse"]), tgt[mm])
        del mu_n, _sd, tgt, sl
    for t in base_tags:
        res[f"{t}_constant_sigma_baseline"] = bacc[t].result()

    doc = {"method": "dincae", "space": "normalised residual (log1p on density/var, then "
                                        "per-channel standardisation) — CRPS absolute values "
                                        "are NOT comparable to the physical-space methods; "
                                        "the CRPS/baseline ratio, spread/skill and coverage are",
           "checkpoints": epochs, "split": args.split, "n_days": len(files),
           "results": res}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(doc, open(args.out, "w"), indent=2)

    print(f"\n{'split':<34}{'RMSE':>8}{'CRPS':>9}{'sigma':>8}{'sp/sk':>8}{'cov90':>8}")
    for t, r in res.items():
        print(f"{t:<34}{r['rmse']:>8.4f}{r['crps']:>9.4f}{r['sigma_mean']:>8.4f}"
              f"{r['spread_skill']:>8.3f}{100 * r['coverage'][90]:>7.1f}%")
    print(f"\n[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
