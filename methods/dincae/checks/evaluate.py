"""
evaluate.py — evaluate DINCAE on held-out days, and run the paper's three checks
========================================================

Three things, all from the paper:

1. **Multi-epoch output averaging** (1.0 Fig.3 / the reference implementation's
   `save_epochs = 200:10:epochs` is this default behaviour). Average the
   **outputs** (not the weights) of checkpoints saved every 10 epochs late in
   training -- better than any single epoch. sigma-hat^2 is averaged the same
   way. 1.0 also warns: ignoring the correlation between errors across epochs
   will **overestimate** sigma-hat.

2. **sigma-hat calibration** (2.0 sec.5.2 Fig.9b/10b)
   Bin cells into 10 bins by predicted SD (cut evenly between the 10%-90%
   percentiles of predicted SD), compute the actual RMS in each bin, plot
   "actual SD vs predicted SD". Ideally this falls on the diagonal.
   2.0 also applies a **global adjustment factor** to sigma-hat, matching the
   average RMS to the average predicted SD -- i.e. the raw sigma-hat's absolute
   scale is biased, and what's trustworthy is its structure and ordering. Both
   before and after adjustment are reported here.

3. **Variance retention** (1.0 Fig.8 / 2.0 Table 3)
   The reconstructed field's standard deviation vs. the ground truth's. RMSE-type
   metrics favour a smooth field (double penalty, 1.0 cites Gilleland 2009 /
   Ebert 2013), so variance must be reported separately, or "smoother" gets
   misread as "better".

**Both MSE conventions are reported**, because they lead to different conclusions
(a pitfall already hit once in the 4dvarnet_enkf project):

  * `ours`  : computed only on cells where **that channel is defined** (velocity
              requires density>0, var requires vel_var>0), restricted to
              walkable. This is our training convention.
  * `v4dvar`: exactly follows `4dvarnet_enkf/checks/eval_test_days.py`'s
              convention -- the raw field, all cells (including non-walkable),
              four channels unweighted, blind = `mask < 0.5`, with the same
              physical clipping as the EnKF (density[0,5], vx/vy[-5,5], var[0,2]).
              Note this convention scores cells **we were never trained on**
              (non-walkable, empty-cell velocity placeholders of 0), so it is
              unfavourable to us; it is reported for comparability, not because
              it is more correct.

Usage (**GPU node**):
    sbatch sbatch/submit_eval.sbatch                 # defaults to the test split
    # or
    srun -p gpu-debug --gres=gpu:1 -t 00:14:00 bash -c \
      'module load scicomp-pytorch-env/2026.1; python3 -u evaluate.py --split valid --days 1 --frames 4000'
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np
import torch

# Scripts in checks/ import source modules from the project root; the root must
# be inserted at the front (so this directory's losses.py takes priority),
# 4dvarnet_enkf can only be appended (it also has a losses.py, and inserting it
# at the front would shadow this directory's).
from crowdcore import observation_model as om                                    # noqa: E402
from crowdcore import paths as repo_paths   # aliased: `paths` is a local variable below

from methods.dincae.state import (CHANNELS, StateStats, NCH, channel_valid,  # noqa: E402
                         fwd_channel, inv_channel)
from methods.dincae.dataset import obs_config                                     # noqa: E402
from methods.dincae.encoding import (FRESH_OFFSETS, N_IN, N_STATIC, observed_pair,  # noqa: E402
                      static_channels)
from methods.dincae.model import DINCAE                                            # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Exactly matches 4dvarnet_enkf/checks/eval_test_days.py:clip_bounds (the EnKF's physical bounds)
CLIP = ((0.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (0.0, 2.0))


#: The checkpoint every PUBLISHED DINCAE number was produced with -- single
#: model, picked on DINCAE's own validation set. Both `compare5_final.json`
#: (via check_outputs/eval_single_00070/) and `uncertainty_dincae.json`
#: (`checkpoints: [70]`) use exactly this file. Multi-checkpoint averaging is
#: the reference implementation's behaviour and scores BETTER (-1.7% RMSE),
#: but the main table reports single models for every method, because none of
#: the four papers ensembles and mixing the two would not be comparable.
PUBLISHED_CKPT = "ckpt_00070.pt"


def load_models(run_dir, ckpt_glob, dev, allow_average=False):
    """Loads the checkpoint(s) to evaluate.

    `allow_average=False` (the default) **refuses to silently average**: if the
    glob matches more than one checkpoint it raises instead of quietly
    returning the reference implementation's multi-checkpoint average, which is
    a different configuration from every published number (see PUBLISHED_CKPT).

    That failure mode is why the guard exists: averaging N checkpoints produces
    a perfectly plausible number roughly 1.7% (RMSE) away from the reported
    one, with no error and nothing in the output to say which configuration
    produced it. Pass `allow_average=True` (CLI: `--average-checkpoints`) to
    reproduce the reference implementation's averaging deliberately.
    """
    paths = sorted(glob.glob(ckpt_glob or os.path.join(run_dir, "ckpt_*.pt")))
    if not paths:                                    # fall back to last.pt if there are no intermediate checkpoints
        paths = [os.path.join(run_dir, "last.pt")]
    if len(paths) > 1 and not allow_average:
        raise SystemExit(
            f"[load_models] {len(paths)} checkpoints matched under {run_dir} and averaging was not "
            f"requested.\n"
            f"  Published configuration : --ckpt-glob '{os.path.join(run_dir, PUBLISHED_CKPT)}'\n"
            f"  Reference-implementation averaging (a DIFFERENT number, ~1.7% better RMSE):\n"
            f"                            --average-checkpoints\n"
            f"  Matched: {[os.path.basename(p) for p in paths[:6]]}"
            + (" ..." if len(paths) > 6 else ""))
    models, epochs = [], []
    for p in paths:
        st = torch.load(repo_paths.require_ckpt(p, "DINCAE checkpoint"), map_location=dev)
        a = st["args"]
        m = DINCAE(N_IN, NCH, enc_internal=tuple(a["enc"]),
                   loss_weights=tuple(a["loss_weights"]), pool=a["pool"]).to(dev)
        m.load_state_dict(st["model"]); m.eval()
        models.append(m); epochs.append(int(st["epoch"]))
    return models, epochs, paths


def build_inputs(scaled, invvar, t_unix, idx, H, W):
    """Assembles the input using the same layout as dataset.encode_day (this is
    the evaluation path, processed in chunks to save memory)."""
    out = np.empty((len(idx), N_IN, H, W), dtype=np.float32)
    out[:, :N_STATIC] = static_channels(t_unix[idx], H, W)
    o = N_STATIC
    for dt in FRESH_OFFSETS:
        j = idx + dt
        out[:, o:o + NCH] = scaled[j]
        out[:, o + NCH:o + 2 * NCH] = invvar[j]
        o += 2 * NCH
    assert o == N_IN, (o, N_IN)
    return out


@torch.no_grad()
def predict_day(models, stats, fp, dev, frames=0, batch=256):
    """Reconstruction for one day (averaged over multiple checkpoints' outputs).

    Returns (Xt, rec, mu_n, sd_n, M):
      Xt   (n,NCH,H,W) ground truth, **raw physical units**
      rec  (n,NCH,H,W) reconstruction, **raw physical units** (denormalised,
           the per-cell mean added back, and inverse-transformed; log1p channels
           take the **median**, expm1(mu), without the log-normal mean
           correction -- see `state.inv_channel` for why)
      mu_n (n,NCH,H,W) predicted mean in normalised residual space
      sd_n (n,NCH,H,W) predicted standard deviation in normalised residual space
      M    (n,NCH,H,W) bool observation mask

    sigma-hat calibration is done in **normalised space**: the Gaussian
    assumption lives there, and calibration looks at the actual/predicted ratio,
    which is itself dimensionless, so doing it in either space gives the same
    result -- doing it in the model's own space is cleanest (especially for
    log1p channels).
    """
    oc = obs_config()
    with h5py.File(fp, "r") as f:
        T_all = f["grid"].shape[0]
        T = min(T_all, frames) if frames else T_all
        X = f["grid"][:T]
        t_unix = f["time"][:T]
    obs = om.generate_observations(
        X, sensing_range=oc["sensing_range"], num_agents=oc["num_agents"],
        add_noise=oc["add_noise"], seed=oc["seed"], valid_mask=stats.valid,
        obs_std=oc["obs_std"], obs_every_k=oc["obs_every_k"])
    Y, M = obs["Y"][:, :NCH], obs["Omega_c"][:, :NCH]
    scaled, invvar = observed_pair(Y, M, stats.mean, stats.std)

    lo, hi = -min(FRESH_OFFSETS), T - max(FRESH_OFFSETS)
    idx_all = np.arange(lo, hi, dtype=np.int64)
    H, W = X.shape[2], X.shape[3]
    std = stats.std.reshape(1, NCH, 1, 1)
    mean = stats.mean[None]

    mu_n = np.empty((len(idx_all), NCH, H, W), dtype=np.float32)
    s2_n = np.empty_like(mu_n)
    for b in range(0, len(idx_all), batch):
        idx = idx_all[b:b + batch]
        xb = torch.from_numpy(build_inputs(scaled, invvar, t_unix, idx, H, W)).to(dev)
        m_sum = torch.zeros(len(idx), NCH, H, W, device=dev)
        s2_sum = torch.zeros_like(m_sum)
        for m in models:                             # output averaging (1.0 Fig.3)
            mo, s2 = m(xb)[-1]                       # take the last level (post-refinement output)
            m_sum += mo; s2_sum += s2
        k = len(models)
        mu_n[b:b + len(idx)] = (m_sum / k).cpu().numpy()
        s2_n[b:b + len(idx)] = (s2_sum / k).cpu().numpy()

    # Normalised residual -> absolute value in transformed space -> raw physical value
    x_sp = mu_n * std + mean                                 # transformed space (log1p channels are still log)
    rec = np.empty_like(x_sp)
    for c in range(NCH):
        # Uses the **median**, expm1(mu), without the log-normal mean correction
        # -- the latter blows up when sigma-hat is uncalibrated (measured: var's
        # physical-space MSE reaching 10^22). See `state.inv_channel` for why.
        rec[:, c] = inv_channel(x_sp[:, c], c)
    return X[idx_all], rec, mu_n, np.sqrt(np.maximum(s2_n, 0.0)), M[idx_all]


def clip_bounds(x):
    x = x.copy()
    for c, (lo, hi) in enumerate(CLIP):
        np.clip(x[:, c], lo, hi, out=x[:, c])
    return x


def accumulate(acc, Xt, rec, mu_n, sd_n, M, stats):
    """Accumulates one day's statistics into acc (per channel; both MSE conventions + calibration + variance).

    MSE and variance are in **physical space**; calibration is in **normalised
    space** (where the Gaussian assumption lives, see predict_day).
    """
    cv = channel_valid(Xt)                                    # (NCH,n,H,W)
    walk = stats.valid[None]
    blind = ~M                                                # unobserved
    rec_clip = clip_bounds(rec)

    for c in range(NCH):
        d2 = (rec[:, c] - Xt[:, c]) ** 2
        d2c = (rec_clip[:, c] - Xt[:, c]) ** 2
        # --- ours: that channel is defined ∩ walkable. **Physical clipping also applied** ---
        # The physical bounds (density>=0, var in [0,2], ...) are known priors and
        # apply to both conventions. Why clipping is necessary: in information
        # form `m = x1*sigma-hat^2` and sigma-hat^2 is capped at 1/mu = 1000
        # (Eq.6 clamping), so an unconverged model can output mu~1000, and after
        # inverting a log1p channel that becomes an astronomical number, letting
        # a handful of cells dominate the whole MSE. `*_noclip` is kept alongside
        # so this pathology stays visible instead of being quietly hidden by clipping.
        ours = cv[c] & walk
        for tag, sel in (("ours_blind", ours & blind[:, c]), ("ours_all", ours)):
            a = acc[tag][c]
            a["se"] += float(d2c[sel].sum()); a["n"] += int(sel.sum())
        for tag, sel in (("ours_blind_noclip", ours & blind[:, c]),
                         ("ours_all_noclip", ours)):
            a = acc[tag][c]
            a["se"] += float(d2[sel].sum()); a["n"] += int(sel.sum())
        # --- v4dvar: all cells, clipped ---
        for tag, sel in (("v4dvar_blind", blind[:, c]),
                         ("v4dvar_all", np.ones_like(blind[:, c]))):
            a = acc[tag][c]
            a["se"] += float(d2c[sel].sum()); a["n"] += int(sel.sum())
        # --- variance retention (1.0 Fig.8): compare standard deviations on the ours-convention cells ---
        a = acc["var_retention"][c]
        a["st"] += float(Xt[:, c][ours].sum()); a["st2"] += float((Xt[:, c][ours] ** 2).sum())
        a["sr"] += float(rec_clip[:, c][ours].sum())          # clipped, same as above
        a["sr2"] += float((rec_clip[:, c][ours] ** 2).sum())
        a["n"] += int(ours.sum())
        # --- calibration: blind ∩ defined, collect (predicted SD, squared error) in normalised space ---
        sel = ours & blind[:, c]
        tgt_n = (fwd_channel(Xt[:, c].astype(np.float64), c)
                 - stats.mean[c][None]) / stats.std[c]
        acc["calib"][c]["sd"].append(sd_n[:, c][sel].astype(np.float32))
        acc["calib"][c]["se"].append(((mu_n[:, c] - tgt_n) ** 2)[sel].astype(np.float32))


def calibration_table(sd, se, nbin=10):
    """2.0 sec.5.2's approach: bin evenly by predicted SD between p10 and p90 into nbin bins, compute the actual RMS in each."""
    if len(sd) == 0:
        return []
    lo, hi = np.percentile(sd, [10, 90])
    edges = np.linspace(lo, hi, nbin + 1)
    rows = []
    for i in range(nbin):
        m = (sd >= edges[i]) & (sd < edges[i + 1] if i < nbin - 1 else sd <= edges[i + 1])
        if m.sum() < 100:
            continue
        rows.append({"pred_sd": float(sd[m].mean()),
                     "actual_sd": float(np.sqrt(se[m].mean())),
                     "n": int(m.sum())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.path.join(ROOT, "runs", "dincae_full"))
    ap.add_argument("--ckpt-glob", default="",
                    help=f"which checkpoint(s) to evaluate. Default: every ckpt_*.pt under "
                         f"--run-dir, which is more than one and therefore requires "
                         f"--average-checkpoints. For the PUBLISHED configuration pass "
                         f"--ckpt-glob '<run-dir>/{PUBLISHED_CKPT}'")
    ap.add_argument("--average-checkpoints", action="store_true",
                    help="average the outputs of every matched checkpoint (1.0 Fig.3, the "
                         "reference implementation's behaviour). **Off by default**: it scores "
                         "~1.7%% better on RMSE but is NOT the configuration any reported number "
                         "uses, and averaging silently would make the two indistinguishable in "
                         "the output. Kept because the gain is itself a reportable quantity.")
    ap.add_argument("--split", default="test", choices=["test", "valid"])
    ap.add_argument("--days", type=int, default=0, help="use only the first N days (debug)")
    ap.add_argument("--frames", type=int, default=0, help="use only the first N frames per day (debug)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--calib-sample", type=int, default=4_000_000,
                    help="max number of points kept for the calibration histogram (random subsample)")
    ap.add_argument("--out", default=os.path.join(ROOT, "check_outputs", "eval"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cpu":
        print("!! no GPU -- do not run torch on the login node (see README)", flush=True)

    stats = StateStats()
    models, epochs, paths = load_models(args.run_dir, args.ckpt_glob, dev,
                                        allow_average=args.average_checkpoints)
    print(f"{'output averaging over' if len(models) > 1 else 'single checkpoint:'} "
          f"{len(models) if len(models) > 1 else ''} epochs {epochs}"
          + ("   [NOT the published configuration]" if len(models) > 1 else ""))

    files = om.split_files(args.split)
    if args.days:
        files = files[: args.days]
    print(f"{args.split} split: {len(files)} days")

    mk = lambda: [{"se": 0.0, "n": 0} for _ in range(NCH)]
    TAGS = ("ours_blind", "ours_all", "ours_blind_noclip", "ours_all_noclip",
            "v4dvar_blind", "v4dvar_all")
    acc = {t: mk() for t in TAGS}
    acc["var_retention"] = [{"st": 0.0, "st2": 0.0, "sr": 0.0, "sr2": 0.0, "n": 0}
                            for _ in range(NCH)]
    acc["calib"] = [{"sd": [], "se": []} for _ in range(NCH)]
    per_day = []

    for fp in files:
        Xt, rec, mu_n, sd_n, M = predict_day(models, stats, fp, dev, args.frames, args.batch)
        # this day's ours_blind MSE (recorded per day, to see stability)
        cv = channel_valid(Xt); walk = stats.valid[None]
        day_mse = {}
        for c in range(NCH):
            sel = cv[c] & walk & (~M[:, c])
            day_mse[CHANNELS[c]] = (float(((clip_bounds(rec)[:, c] - Xt[:, c]) ** 2)[sel].mean())
                                    if sel.any() else None)
        per_day.append({"day": os.path.splitext(os.path.basename(fp))[0],
                        "frames": int(Xt.shape[0]), "ours_blind_mse": day_mse})
        print(f"  {per_day[-1]['day']}  " +
              "  ".join(f"{k} {v:.5f}" for k, v in day_mse.items() if v is not None),
              flush=True)
        accumulate(acc, Xt, rec, mu_n, sd_n, M, stats)
        del Xt, rec, mu_n, sd_n, M

    rng = np.random.default_rng(0)
    result = {"run_dir": args.run_dir, "split": args.split,
              "checkpoints": [os.path.basename(p) for p in paths], "epochs": epochs,
              "n_days": len(files), "per_day": per_day, "channels": list(CHANNELS)}

    for tag in TAGS:
        result[tag + "_mse"] = {CHANNELS[c]: (acc[tag][c]["se"] / acc[tag][c]["n"]
                                             if acc[tag][c]["n"] else None)
                                for c in range(NCH)}
        # unweighted merge of the four channels (this is the merged number 4dvarnet_enkf reports)
        se = sum(acc[tag][c]["se"] for c in range(NCH))
        n = sum(acc[tag][c]["n"] for c in range(NCH))
        result[tag + "_mse_all_channels"] = se / n if n else None

    result["var_retention"] = {}
    for c in range(NCH):
        a = acc["var_retention"][c]
        n = max(a["n"], 1)
        st = np.sqrt(max(a["st2"] / n - (a["st"] / n) ** 2, 0.0))
        sr = np.sqrt(max(a["sr2"] / n - (a["sr"] / n) ** 2, 0.0))
        result["var_retention"][CHANNELS[c]] = {
            "truth_sd": float(st), "rec_sd": float(sr),
            "ratio": float(sr / st) if st > 0 else None}

    result["calibration"] = {}
    for c in range(NCH):
        sd = np.concatenate(acc["calib"][c]["sd"]) if acc["calib"][c]["sd"] else np.array([])
        se = np.concatenate(acc["calib"][c]["se"]) if acc["calib"][c]["se"] else np.array([])
        if len(sd) > args.calib_sample:
            j = rng.choice(len(sd), args.calib_sample, replace=False)
            sd, se = sd[j], se[j]
        rows = calibration_table(sd, se)
        # global adjustment factor (2.0 sec.5.2): matches average predicted SD to actual RMS
        adj = (float(np.sqrt(se.mean()) / sd.mean()) if len(sd) and sd.mean() > 0 else None)
        result["calibration"][CHANNELS[c]] = {
            "bins": rows, "global_adjust_factor": adj,
            "mean_pred_sd": float(sd.mean()) if len(sd) else None,
            "actual_rms": float(np.sqrt(se.mean())) if len(se) else None,
            "n": int(len(sd))}

    path = os.path.join(args.out, f"dincae_metrics_{args.split}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== MSE (both conventions) ===")
    print(f"{'convention':16s} " + "  ".join(f"{c:>9s}" for c in CHANNELS) + "   4-channel merged")
    for tag in TAGS:
        vals = "  ".join(f"{result[tag + '_mse'][c]:9.5f}"
                         if result[tag + "_mse"][c] is not None else "        -"
                         for c in CHANNELS)
        print(f"{tag:16s} {vals}   {result[tag + '_mse_all_channels']:.5f}")

    print("\n=== Variance retention (reconstruction SD / truth SD, 1 is best; <1 = smoothed out) ===")
    for c in CHANNELS:
        v = result["var_retention"][c]
        print(f"  {c:8s} truth {v['truth_sd']:.4f}  rec {v['rec_sd']:.4f}  "
              f"ratio {v['ratio']:.3f}" if v["ratio"] else f"  {c}: -")

    print("\n=== sigma-hat calibration (blind, normalised space; ideally actual ≈ predicted) ===")
    for c in CHANNELS:
        v = result["calibration"][c]
        if not v["bins"]:
            print(f"  {c}: not enough samples"); continue
        print(f"  {c:8s} mean predicted SD {v['mean_pred_sd']:.4f}  actual RMS {v['actual_rms']:.4f}  "
              f"global adjustment factor {v['global_adjust_factor']:.3f}  (n={v['n']:,})")
        for r in v["bins"]:
            print(f"      pred {r['pred_sd']:.4f} -> actual {r['actual_sd']:.4f}  "
                  f"(n={r['n']:,})")

    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
