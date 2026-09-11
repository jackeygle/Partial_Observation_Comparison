"""
compare5.py — per-channel comparison across four methods (+2 uncertainty designs),
four cell-scope
conventions side by side
==============================================

Four reconstruction methods plus our two 4DVarNet uncertainty designs (all on the
full day of 7 held-out days, obs_every_k=1, seed=0, the same physical clipping):

    Senseiver                     sparse sensors -> field
    4DVarNet MSE   single/ensemble  Eq.14's plain squared loss
    4DVarNet NLL   single/ensemble  Gaussian NLL + sigma-hat read-out head ("the uncertainty head")
    DINCAE                        convolutional autoencoder inpainting (single checkpoint, epoch 70)
    EnKF k1                       localised EnKF, a vendor copy of Partial_observation

The MSE arm and NLL arm are **identical weight-for-weight** in prior/solver
architecture (hidden 32, kt 3, lstm_hidden 64, GENN 9,474); the only difference is
the NLL side's extra `grad_net.out_var` -- a 692,000-parameter sigma-hat read-out
head. So the loss function's effect is a clean comparison at both the
single-model and ensemble level.

IN PROGRESS as of 2026-09-10, not yet reflected in the numbers this script
reports: the capacity sweep (runs/varnet_a4_k1) found hidden=96/kt=5 beats this
32/3 configuration by ~7% and is being adopted as the new baseline, with the
three arms above being retrained at 96/5 to match (runs/varnet_{mse5,vsb0,aug0}_h96_s*)
so the headline model and the uncertainty comparison share one backbone. Update
this figure and re-point --arms once that retrain has validation-selected
checkpoints -- until then it would describe runs this script is not reading.

Why four conventions
----------------------
**Rankings flip depending on the convention.** On the same predictions, under the
"all cells" convention Senseiver comes first and DINCAE comes last, 4.5x worse;
switch to "channel-defined cells" and DINCAE comes first, Senseiver third.

The cause is in the velocity channels: an empty cell has no people and hence no
velocity, and the `0` the data pipeline stores there is a placeholder, not a
measurement -- and 88.4% of blind cells are empty. The three methods trained on
the full field learned to output 0 on empty cells and get that part for free;
DINCAE was only ever trained on defined cells and gets crushed by this term -- its
`vy` is 0.1026 under its own convention and jumps to 1.6489 (16x) under the
all-cells convention, while `vx` barely moves (0.546 -> 0.571). This asymmetry is
a convention artifact, not a property of the model.

So all four are computed side by side, making the difference visible in one piece
of output rather than making the reader compare several jsons themselves:

    (1) defined        blind ∩ defined ∩ walkable      the information form's own scope
    (2) defined_full   full field ∩ defined ∩ walkable includes observed cells
    (3) walkable       blind ∩ walkable                **the main result**: every cell of the
                                                       physical domain, empty ones included,
                                                       map-obstacle cells excluded
    (4) walkable_full  full field ∩ walkable           includes observed cells
    (5) allcells       blind, all cells                the removed three-way script's convention
    (6) full           full field, all cells           the whole grid, obstacles included

Each is further multiplied by "clipped/unclipped" and "pooled/day-averaged",
because those two axes have each been mixed up before too: `test_metrics_*.json`
reports the day average, the removed three-way script reported pooled -- the two are
not the same quantity.

Convention details
--------
  * Channel-defined cells: density is defined everywhere; vx/vy require
    density>0; var requires var>0. The single definition lives in
    `methods/dincae/state.py:channel_valid()`, imported directly by this script --
    **the rule is never copied**.
  * ∩ walkable. Verified cell by cell that
    `methods/dincae/artifacts/state_stats.npz["valid_mask"]` and
    `nav.build_valid_mask_from_config()` are exactly identical (290/432 cells), so
    both sides feed `generate_observations` the same `valid_mask`, and the blind
    set is consistent by construction (there is an assertion for this in the
    script).
  * Frame range `[1, T-1)`: DINCAE's `FRESH_OFFSETS=(-1,0,1)` forces it to drop the
    first and last frame; every other method drops them too, otherwise the
    denominators would differ.
  * Pooling: accumulate squared error and cell count per channel first, divide
    once at the end.

The DINCAE row
-------------
Read from `methods/dincae/check_outputs/eval_single_00070/dincae_metrics_test.json`
-- the single epoch-70 checkpoint every published DINCAE number uses, not the
16-checkpoint average in `check_outputs/eval/`. Its own evaluate.py happens to
report exactly these four conventions, matching one-to-one
(`ours_blind` / `ours_all` / `v4dvar_blind` / `v4dvar_all`). When combining into
the "total", **this script's own per-channel cell counts** are used, not the
`all_channels` value in its json -- the latter's denominator uses its own frame
range.

Fairness note: this used to read the 16-checkpoint average, which gave DINCAE an
inherent ensembling advantage over every other row. It now reads the single
epoch-70 checkpoint, so all rows are single models and directly comparable.

Reproducibility floor
--------
4DVarNet's numbers have a ~4e-4 relative jitter: `GradSolver` also runs an
autograd backward pass **at inference time**, and conv backward's atomicAdd
reduction order differs every run. Senseiver (no_grad), EnKF (numpy reading npz),
and DINCAE (reading json) are all bit-reproducible. See
`refactor_baseline/README.md`. **4DVarNet's 4th decimal place is noise -- do not
report a difference at that scale.**

Usage (GPU node, from the repo root)
--------------------------
    source sbatch/_env.sh
    python3 -m compare.compare5
    python3 -m compare.compare5 --days 1 --ensembles ""     # smoke test, skip the ensembles
"""
from __future__ import annotations

import argparse
import json
import os
from crowdcore import paths
import sys

import numpy as np
import torch

from methods.senseiver import dataset as ds
from methods.senseiver import sensors
from methods.senseiver.network import Senseiver

from methods.varnet.checks.model_io import load_solver
from crowdcore import navigation as nav
# The single definition of the "channel-defined cells" rule. Before the
# 2026-09-03 refactor this was an importlib.spec_from_file_location hack --
# because dincae and senseiver both have dataset.py/losses.py/model.py, putting
# dincae's root on sys.path would shadow this project's modules. After
# packaging, a plain import works fine.
from methods.dincae.state import StateStats, channel_valid

# DINCAE's FRESH_OFFSETS = (-1, 0, 1), see dincae_crowd/encoding.py:30. Hardcoded
# and asserted here so we don't have to import that file's encoding.py (which
# would collide with the local one) just for one constant.
FRESH_OFFSETS = (-1, 0, 1)
FRAME_LO = -min(FRESH_OFFSETS)
FRAME_HI_PAD = max(FRESH_OFFSETS)

LO = np.array([0.0, -5.0, -5.0, 0.0], np.float32)
HI = np.array([5.0, 5.0, 5.0, 2.0], np.float32)

# Three cell-scope choices x clipped/unclipped. `full` is "the full field" --
# observed + blind, all cells, the same quantity as eval_test_days.py's full_mse
# and the removed three-way script's rmse_all; it is included so those scripts'
# reported numbers can be put next to this one's.
#
# eval_test_days.py used to compute the same two quantities and disagreed with
# this script by a constant +0.0027 on b0_k1. Rather than reconcile two
# implementations of one metric, it was removed on 2026-09-09 (archived at tag
# archive-2026-09-09) and everything now reads this one.
# Why `_noclip` exists: the removed three-way script did not clip, test_metrics does -- that is the
# only known convention difference between them.
BASE_CONVENTIONS = ("defined", "defined_full", "walkable", "walkable_full",
                    "allcells", "full")
CONVENTIONS = BASE_CONVENTIONS + tuple(f"{k}_noclip" for k in BASE_CONVENTIONS)


def clip_np(x):
    return np.clip(x, LO[None, :, None, None], HI[None, :, None, None])


class Acc:
    """Accumulates squared error and cell count per channel, per convention (for pooling)."""

    def __init__(self, C):
        self.se = {k: np.zeros(C) for k in CONVENTIONS}
        self.n = {k: np.zeros(C, dtype=np.int64) for k in CONVENTIONS}

    def add(self, pred_clip, pred_raw, true, sel):
        """pred/true (T,C,H,W); sel is a dict {convention: (T,C,H,W) bool} of masks.

        `*_noclip` conventions are scored using the unclipped pred_raw, the rest
        use pred_clip.
        """
        d2c = (pred_clip - true) ** 2
        d2r = (pred_raw - true) ** 2
        for k, m in sel.items():
            d2 = d2r if k.endswith("_noclip") else d2c
            for c in range(d2.shape[1]):
                self.se[k][c] += float(d2[:, c][m[:, c]].sum())
                self.n[k][c] += int(m[:, c].sum())

    def mse(self, k):
        return self.se[k] / np.maximum(self.n[k], 1)

    def overall(self, k):
        return self.se[k].sum() / max(self.n[k].sum(), 1)

    def merge(self, other):
        for k in CONVENTIONS:
            self.se[k] += other.se[k]
            self.n[k] += other.n[k]


def varnet_ckpt(run_name, ckpt_name):
    """Which checkpoint to report for runs/varnet_<run_name>, and why.

    Default (ckpt_name empty): the epoch chosen on the VALIDATION split by
    methods/varnet/checks/select_checkpoint.py, recorded per run in
    select_valid.json. Each run gets its own epoch; all of them solve with 20
    iterations, the final stage of the paper's Sec. 3.4 curriculum, because the
    selector refuses candidates that do not.

    This is not a preference, it is what makes the arms comparable. varnet_best.pt
    is selected inside train.py on the first --n-eval windows of --split, and
    --split defaults to train -- so it is model selection on training data, and it
    lands each arm at a different epoch AND a different solver depth. It also put
    4DVarNet out of step with the other two methods, which both select on
    validation (Senseiver in its own train.py, DINCAE in its select_checkpoint.py).
    """
    rd = os.path.join(paths.runs(paths.VARNET), f"varnet_{run_name}")
    if ckpt_name:
        return os.path.join(rd, ckpt_name)
    sel = os.path.join(rd, "select_valid.json")
    if not os.path.exists(sel):
        # Fall back rather than refuse: every headline run this README reports
        # (mse5_s*, vsb0_s*, aug0_s*) predates checks/select_checkpoint.py and has no
        # select_valid.json, so a hard failure here would break the documented
        # reproduction command (`sbatch sbatch/submit_compare5.sbatch`) for every
        # number currently in compare5_final.json. varnet_best.pt is exactly the
        # checkpoint those numbers were produced from -- see the README's "Which
        # file backs which published number" table -- so falling back to it
        # reproduces the shipped table rather than reporting a different one.
        print(f"[ckpt] {sel} not found -- falling back to varnet_best.pt for "
              f"varnet_{run_name} (train-split selection; see train.py's --split "
              f"default). Run methods.varnet.checks.select_checkpoint on this run "
              f"to switch it to the validation-selected epoch.", flush=True)
        return os.path.join(rd, "varnet_best.pt")
    with open(sel) as f:
        j = json.load(f)
    return os.path.join(rd, j["selected_ckpt"])


def run_senseiver(model, Y, Om, dev, batch):
    pe = model.pos_enc.cpu().numpy()
    mu = model.in_mean.cpu().numpy()
    sd = model.in_std.cpu().numpy()
    outs = []
    with torch.no_grad():
        for i in range(0, Y.shape[0], batch):
            sl = slice(i, i + batch)
            tok, pad, _ = sensors.build_batch(Y[sl], Om[sl], pe, mu, sd)
            outs.append(model.reconstruct(tok.to(dev), pad.to(dev)).cpu().numpy())
    return np.concatenate(outs, 0)


def run_varnet(solver, Y, Omc, X0, dT, dev, batch):
    """Returns (reconstruction (nw*dT, C, H, W), number of frames covered). Same
    path as the removed compare3.run_varnet."""
    from crowdcore import observation_model as om
    win = lambda a: om.to_windows(a, dT)
    Yw, Mw, X0w = win(Y), win(Omc.astype(np.float32)), win(X0)
    outs = []
    with torch.enable_grad():
        for i in range(0, Yw.shape[0], batch):
            yb = torch.from_numpy(Yw[i:i + batch]).float().to(dev)
            mb = torch.from_numpy(Mw[i:i + batch]).float().to(dev)
            xb = torch.from_numpy(X0w[i:i + batch]).float().to(dev)
            outs.append(solver(xb, yb, mb).detach().cpu().numpy())
    r = np.concatenate(outs, 0)                          # (nw, C, dT, H, W)
    nw, C, dt, H, W = r.shape
    return r.transpose(0, 2, 1, 3, 4).reshape(nw * dt, C, H, W), nw * dt


def build_masks(X, Omf, walk, C):
    """The blind masks for both conventions, (T,C,H,W) bool.

    X    (T,C,H,W) ground truth (raw physical values -- channel_valid judges on raw values)
    Omf  (T,H,W)   observation mask (all four channels are observed together, so
                   this is per-cell, not per-channel)
    walk (H,W)     walkable
    """
    blind = ~np.repeat(Omf[:, None], C, axis=1)                  # (T,C,H,W)
    cv = np.moveaxis(channel_valid(X), 0, 1)                     # (NCH,T,H,W) -> (T,C,H,W)
    w = walk[None, None]
    base = {"defined": blind & cv & w,          # blind ∩ defined ∩ walkable
            "defined_full": cv & w,                # defined ∩ walkable, observed cells count too
            "walkable": blind & w,                 # blind ∩ walkable -- every cell of the
                                                   # physical domain, empty ones included
            "walkable_full": w & np.ones_like(blind),  # walkable, observed cells count too
            "allcells": blind,                     # blind, all cells
            "full": np.ones_like(blind)}           # full field = observed + blind, all cells
    base.update({f"{k}_noclip": v for k, v in base.items()})      # same cells, scored unclipped
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--senseiver",
                    default=os.path.join(paths.runs(paths.SENSEIVER),
                                         "senseiver_A", "best.pt"))
    ap.add_argument("--varnet", default="",
                    help="comma-separated 4dvarnet run names to add as extra single-run rows, "
                         "e.g. 'a4_k1'. Empty by default: the three arms below already cover "
                         "every reported 4DVarNet number, and a2/a4/b0 are exploratory "
                         "capacity ablations that are not part of any reported comparison.")
    ap.add_argument("--arms", default="MSE=mse5_s{},NLL=vsb0_s{},AUG=aug0_s{}",
                    help="comma-separated `label=run-name-template` entries. Each "
                         "arm produces N single-seed rows, plus one cross-seed "
                         "mean+/-std row. Leave empty to skip all arms.")
    ap.add_argument("--arm-seeds", default="0,1,2,3,4")
    ap.add_argument("--ckpt-name", default="",
                    help="DIAGNOSTIC ONLY: take this one checkpoint name from every run "
                         "directory instead of the per-run choice recorded in "
                         "select_valid.json. Empty (the default) = use the selection, which "
                         "is what every reported number comes from.\n"
                         "Do not pass varnet_best.pt to produce a reported number. train.py "
                         "selects it on the first --n-eval windows of --split, and --split "
                         "defaults to TRAIN, so it is model selection on training data -- and "
                         "on the emptiest part of it. It also lands the arms at different "
                         "points of the curriculum: measured on the hidden=32 runs, aug0's "
                         "best fell at epoch 16-40 with the solver still at 5-10 iterations "
                         "while mse5's fell at 78-143 at 20, so those two checkpoints differ "
                         "in solver depth as well as in loss.")
    ap.add_argument("--with-ensemble", action="store_true",
                    help="additionally compute one row averaging N members' "
                         "reconstructions into an ensemble. **Off by default** -- "
                         "the paper never mentions ensembling, so the main table "
                         "reports single models. This switch is kept because the "
                         "ensembling gain is itself a reportable quantity "
                         "(measured: 4.2%% for the MSE arm, 1.7%% for the NLL arm).")
    ap.add_argument("--enkf-dir", default=paths.enkf_export("enkf_k1_full"))
    ap.add_argument("--dincae-json",
                    default=os.path.join(paths.method(paths.DINCAE), "check_outputs",
                                         "eval_single_00070", "dincae_metrics_test.json"),
                    help="the PUBLISHED DINCAE row: the single epoch-70 checkpoint. "
                         "check_outputs/eval/ holds the 16-checkpoint average instead, "
                         "which is a different configuration from every other row here.")
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--varnet-batch", type=int, default=16)
    # Output lands in compare/'s own directory: this quantity is cross-method and
    # does not belong to any single method's check_outputs. Before the refactor
    # it was written under senseiver_crowd/check_outputs/eval/, making people
    # think it was a Senseiver metric.
    ap.add_argument("--out", default=os.path.join(paths.COMPARE, "results",
                                                  "compare5.json"),
                    help="a rerun writes here; the PUBLISHED artefact is "
                         "compare5_final.json, which the figure and README read. "
                         "Separate files on purpose: a rerun must not silently "
                         "overwrite the numbers the write-up quotes.")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C, H, W = ds.state_shape()
    chans = ds.channels()

    # walkable: the two projects' masks were already verified identical; assert
    # it once here so it can never silently drift.
    walk = StateStats(os.path.join(paths.method(paths.DINCAE),
                                   "artifacts", "state_stats.npz")).valid
    walk_4d = nav.build_valid_mask_from_config()
    assert walk.shape == walk_4d.shape and (walk == walk_4d).all(), \
        "dincae's walkable no longer matches 4dvarnet's -- the premise for all four conventions is broken"

    sck = torch.load(args.senseiver, map_location=dev, weights_only=False)
    sm = Senseiver(**sck["hparams"]).to(dev)
    sm.load_state_dict(sck["model"])
    sm.eval()

    vnames = [z.strip() for z in args.varnet.split(",") if z.strip()]
    vsolvers = {}
    for vn in vnames:
        sol, va, _ = load_solver(varnet_ckpt(vn, args.ckpt_name), dev)
        vsolvers[vn] = (sol, va)

    # Deep ensemble. The point estimate is the mean of the members'
    # reconstructions (the paper's Sec 2.4), so "ensemble" and "single-model
    # mean+/-std" are two different quantities -- both are reported, otherwise
    # only reporting the ensemble would conflate "the benefit of averaging" with
    # "the effect of the loss function."
    #
    # Why the ensembles are shown side by side: mse5, vsb0 and aug0 are **identical
    # weight-for-weight** in prior/solver architecture (hidden 32, kt 3,
    # lstm_hidden 64, GENN 9,474); the only differences are vsb0's extra
    # grad_net.out_var (a 692,000-parameter sigma-hat read-out head) and aug0's
    # augmented state. So MSE-vs-NLL is a clean comparison at both the
    # single-model and ensemble level. (In progress: being retrained at
    # hidden=96/kt=5 to match the capacity sweep's winner -- see the module
    # docstring. --arms still points at the 32/3 runs until that finishes.)
    #
    # The MSE ensemble is not decoration: it is Lakshminarayanan et al.'s own
    # Table 2 baseline, M networks trained on the squared loss whose uncertainty
    # is the EMPIRICAL VARIANCE across members rather than a learnt sigma^2.
    # Dropping it would leave the two learnt-sigma designs comparable only to each
    # other, with nothing to test the paper's central claim against.
    ens_members = [z.strip() for z in args.arm_seeds.split(",") if z.strip()]
    ensembles = {}                                   # label -> [solver, ...]
    picked = {}                                      # label -> [(run, ckpt, epoch, n_iter)]
    for spec in (z.strip() for z in args.arms.split(",") if z.strip()):
        label, _, fmt = spec.partition("=")
        assert fmt, f"each --ensembles entry must be written as label=run-name-template, got {spec!r}"
        sols, edT = [], None
        for m in ens_members:
            cp = varnet_ckpt(fmt.format(m), args.ckpt_name)
            sol, va, ck = load_solver(cp, dev)
            # State the epoch each member came from. Per-run validation selection means the
            # five members of an ensemble are NOT all from the same epoch -- legitimate under
            # Lakshminarayanan et al. (M independently initialised networks; nothing requires
            # a shared epoch), but it has to be visible in the log rather than inferred.
            picked.setdefault(label, []).append(
                (fmt.format(m), os.path.basename(cp), int(ck.get("epoch", -1)),
                 int(ck.get("n_iter_eff", -1))))
            sols.append(sol)
            edT = va["dT"]
        ensembles[label] = (sols, edT)
    print(f"[model] Senseiver {sm.num_params:,} params | "
          + " | ".join(f"4DVarNet {vn} dT={va['dT']} n_iter={sol.n_iter}"
                       for vn, (sol, va) in vsolvers.items())
          + f" | walkable {int(walk.sum())}/{walk.size} | device={dev}", flush=True)
    # Which checkpoint each ensemble member came from. Per-run validation selection puts the
    # members at different epochs, so print it rather than leaving the reader to assume they
    # are all epoch 149. n_iter is printed too because it is the quantity that has to be
    # constant across members and arms for the comparison to mean anything.
    for label, rows in picked.items():
        depths = {d for *_, d in rows}
        note = "" if len(depths) == 1 else "   <-- MIXED SOLVER DEPTH, arms are not comparable"
        print(f"[ckpt] {label}: "
              + ", ".join(f"{r}={c}(ep{e},{d}it)" for r, c, e, d in rows) + note, flush=True)

    days = ds.om.split_files("test")
    if args.days:
        days = days[:args.days]

    names = ["Senseiver"] + [f"4DVarNet {vn}" for vn in vnames]
    for label, (sols, _) in ensembles.items():
        if args.with_ensemble:
            names.append(f"4DVarNet {label} ens{len(sols)}")
        names += [f"4DVarNet {label} s{m}" for m in ens_members]
    names.append("EnKF k1")
    accs = {k: Acc(C) for k in names}
    # Per-day overall, used for the "day-averaged" convention. test_metrics_*.json
    # / eval_test_days.py report this; the removed three-way script reported pooled --
    # the two are not the same quantity, so both are given here.
    day_overall = {k: {c: [] for c in CONVENTIONS} for k in names}
    per_day = []

    for d in days:
        stem = os.path.basename(d).split("_")[0]
        X, Y, Om = ds.load_day(d, stride=1, seed=0)
        n = X.shape[0]
        Xf = X.reshape(n, C, H, W)
        Yf = Y.reshape(n, C, H, W)
        Omf = Om.reshape(n, H, W)

        lo, hi = FRAME_LO, n - FRAME_HI_PAD                  # the frame range DINCAE can be scored on
        sel_all = build_masks(Xf, Omf, walk, C)
        cut = lambda m, a, b: {k: v[a:b] for k, v in m.items()}
        dacc = {k: Acc(C) for k in names}                    # accumulated separately for this day, to compute the day average

        p = run_senseiver(sm, Y, Om, dev, args.batch)
        dacc["Senseiver"].add(clip_np(p[lo:hi]), p[lo:hi], Xf[lo:hi], cut(sel_all, lo, hi))
        del p

        Omc = np.repeat(Omf[:, None], C, axis=1)
        X0 = ds.om.fill_missing_state(Yf, Omc, method=ds.obs_config()["init_method"])
        for vn in vnames:
            sol, va = vsolvers[vn]
            pv, nkeep = run_varnet(sol, Yf, Omc, X0, va["dT"], dev, args.varnet_batch)
            b = min(hi, nkeep)                               # the tail dropped when dT doesn't divide evenly
            dacc[f"4DVarNet {vn}"].add(clip_np(pv[lo:b]), pv[lo:b], Xf[lo:b],
                                       cut(sel_all, lo, b))
            del pv

        for label, (sols, edT) in ensembles.items():
            ens = None
            for m, sol in zip(ens_members, sols):
                pv, nkeep = run_varnet(sol, Yf, Omc, X0, edT, dev, args.varnet_batch)
                b = min(hi, nkeep)
                dacc[f"4DVarNet {label} s{m}"].add(clip_np(pv[lo:b]), pv[lo:b], Xf[lo:b],
                                                   cut(sel_all, lo, b))
                if args.with_ensemble:
                    ens = pv if ens is None else ens + pv   # accumulate rather than keeping 5 whole-day arrays at once
                del pv
            if args.with_ensemble:
                ens /= len(sols)
                b = min(hi, ens.shape[0])
                dacc[f"4DVarNet {label} ens{len(sols)}"].add(
                    clip_np(ens[lo:b]), ens[lo:b], Xf[lo:b], cut(sel_all, lo, b))
            del ens

        ep = os.path.join(args.enkf_dir, f"est_{stem}.npz")
        if os.path.exists(ep):
            est = np.load(ep)["Est"].astype(np.float32)
            b = min(hi, est.shape[0])
            dacc["EnKF k1"].add(clip_np(est[lo:b]), est[lo:b], Xf[lo:b], cut(sel_all, lo, b))
            del est
        else:
            print(f"  [warn] no EnKF export for {stem}", flush=True)

        rec = {"day": stem, "frames_total": int(n), "frames_scored": int(hi - lo)}
        for k, a in dacc.items():
            if a.n["full"].sum() == 0:
                continue
            accs[k].merge(a)
            for c in CONVENTIONS:
                day_overall[k][c].append(float(a.overall(c)))
            rec[k] = {"full_mse": float(a.overall("full")),
                      "defined_blind_mse": float(a.overall("defined"))}
        per_day.append(rec)
        print(f"  {stem}: {n} frames, scored [{lo},{hi})  "
              + "  ".join(f"{k} full={rec[k]['full_mse']:.4f}" for k in dacc if k in rec),
              flush=True)

    # --- DINCAE: read its already-computed per-channel MSE on channel-defined cells ---
    dincae_pc = None
    if os.path.exists(args.dincae_json):
        with open(args.dincae_json) as f:
            dj = json.load(f)
        # DINCAE's own evaluate.py happens to report exactly these four
        # conventions, matching this script's four one-to-one
        dincae_by_conv = {
            "defined":      dj.get("ours_blind_mse"),      # defined ∩ walkable ∩ blind
            "defined_full": dj.get("ours_all_mse"),        # defined ∩ walkable ∩ full field
            "walkable":      dj.get("walkable_blind_mse"),  # walkable ∩ blind (the main scope)
            "walkable_full": dj.get("walkable_all_mse"),    # walkable ∩ full field
            "allcells":     dj.get("v4dvar_blind_mse"),    # all cells ∩ blind
            "full":         dj.get("v4dvar_all_mse"),      # all cells ∩ full field
        }
        dincae_pc = dincae_by_conv["defined"]
        if int(dj.get("n_days", 0)) != len(days):
            print(f"\n  [warn] the DINCAE row is pooled over {dj.get('n_days')} days, "
                  f"this run only covered {len(days)} -- the two are not comparable, "
                  f"this table is only useful to check the code runs.", flush=True)
    else:
        print(f"  [warn] {args.dincae_json} not found, leaving the DINCAE row blank", flush=True)
        dincae_by_conv = {}

    # --- Output -------------------------------------------------------------------
    res = {"protocol": {
        "cells": "blind ∩ channel-defined ∩ walkable ('defined'), alongside blind ('allcells')",
        "channel_defined": "density: everywhere; vx/vy: density>0; var: var>0"
                           " (dincae_crowd/state.py:channel_valid)",
        "walkable_cells": int(walk.sum()), "grid_cells": int(walk.size),
        "frame_range": f"[{FRAME_LO}, T-{FRAME_HI_PAD})  (aligned with DINCAE's FRESH_OFFSETS)",
        "obs_every_k": 1, "seed": 0, "clip": "EnKF's physical bounds",
        "ckpt": args.ckpt_name,
        "pooling": "accumulate squared error and cell count per channel, divide once at the end",
        "n_days": len(days)},
        "per_day": per_day, "channels": chans, "results": {}}

    for k, a in accs.items():
        if a.n["defined"].sum() == 0:
            continue
        res["results"][k] = {
            conv: {"per_channel": {chans[i]: float(v) for i, v in enumerate(a.mse(conv))},
                   "n_per_channel": {chans[i]: int(v) for i, v in enumerate(a.n[conv])},
                   "overall": float(a.overall(conv)),
                   # Two summaries of the same numbers, placed side by side so
                   # they cannot be mixed up again
                   "overall_pooled": float(a.overall(conv)),
                   "overall_mean_of_days": float(np.mean(day_overall[k][conv])),
                   "rmse_pooled": float(np.sqrt(a.overall(conv))),
                   "rmse_mean_of_days": float(np.mean(
                       [np.sqrt(v) for v in day_overall[k][conv]]))}
            for conv in CONVENTIONS}

    # --- Cross-seed summary: the number that should be reported for a single model ---
    #
    # The main table reports single models (the paper never mentions
    # ensembling). But "single model" is not one number -- there are 5 seeds, so
    # the honest way to write it is mean +/- std, the same way Lakshminarayanan
    # reports the mean and spread across folds. Reporting the best-scoring seed
    # would be systematically over-optimistic; reporting s0 would be arbitrary.
    #
    # This spread has a second use: judging whether the gap between the two arms
    # exceeds seed noise. Measured: MSE arm 0.1239+/-0.0033, NLL arm
    # 0.1506+/-0.0009, a gap of 0.0267 = 8.2x the MSE arm's seed std, so "MSE
    # beats NLL" already holds at the single-model level, without relying on
    # ensembling.
    res["seed_summary"] = {}
    for label in ensembles:
        keys = [f"4DVarNet {label} s{m}" for m in ens_members if f"4DVarNet {label} s{m}" in accs]
        if not keys:
            continue
        entry = {}
        for conv in CONVENTIONS:
            ov = [float(accs[k].overall(conv)) for k in keys]
            pc = {c: [float(accs[k].mse(conv)[i]) for k in keys]
                  for i, c in enumerate(chans)}
            entry[conv] = {
                "n_seeds": len(keys),
                "overall_mean": float(np.mean(ov)), "overall_std": float(np.std(ov)),
                "overall_min": float(np.min(ov)), "overall_max": float(np.max(ov)),
                "rmse_mean": float(np.mean([np.sqrt(v) for v in ov])),
                "per_channel_mean": {c: float(np.mean(v)) for c, v in pc.items()},
                "per_channel_std": {c: float(np.std(v)) for c, v in pc.items()},
            }
        res["seed_summary"][label] = entry

    # DINCAE's total is recomputed using this script's own cell counts, see the file header for why
    if dincae_pc:
        ref = next(iter(accs.values()))
        entry = {}
        for conv, pc in dincae_by_conv.items():
            if not pc:
                continue
            nn = ref.n[conv]
            se = sum(pc[c] * nn[i] for i, c in enumerate(chans))
            ov = float(se / max(nn.sum(), 1))
            # Fields aligned with the other methods, so downstream plotting code
            # can treat them uniformly. **No** rmse_mean_of_days: the DINCAE row
            # is a pooled value read from its own json, and we don't have its
            # per-day data.
            entry[conv] = {"per_channel": {c: float(pc[c]) for c in chans},
                           "n_per_channel": {chans[i]: int(v) for i, v in enumerate(nn)},
                           "overall": ov, "overall_pooled": ov,
                           "rmse_pooled": float(np.sqrt(ov)),
                           "source": os.path.relpath(args.dincae_json, paths.ROOT)}
        res["results"]["DINCAE"] = entry

    print("\n\nFull-field RMSE (observed + blind, all cells) -- pooled vs day-averaged vs unclipped\n")
    print(f"{'method':<24}{'pooled':>10}{'day-avg':>12}{'pooled,noclip':>14}{'day-avg,noclip':>16}")
    print("-" * 76)
    for k, q in res["results"].items():
        if "full" not in q:
            continue
        cell = lambda c, f: (f"{q[c][f]:.4f}" if c in q and f in q[c] else "—")
        print(f"{k:<24}{cell('full','rmse_pooled'):>10}"
              f"{cell('full','rmse_mean_of_days'):>12}"
              f"{cell('full_noclip','rmse_pooled'):>14}"
              f"{cell('full_noclip','rmse_mean_of_days'):>16}")

    for conv, title in (
            ("defined", "(1) defined ∩ walkable ∩ blind  -- comparable across all five, the main result"),
            ("defined_full", "(2) defined ∩ walkable ∩ full field (includes observed cells)"),
            ("walkable", "(3) walkable ∩ blind  -- the main result: the physical domain, "
                         "empty cells included, map obstacles excluded"),
            ("walkable_full", "(4) walkable ∩ full field  (includes observed cells)"),
            ("allcells", "(5) all cells ∩ blind  -- the removed three-way script's convention, for reference"),
            ("full", "(4) all cells ∩ full field  -- the same quantity as eval_test_days's full_mse")):
        print(f"\n{title}\n")
        print(f"{'method':<24}" + "".join(f"{c:>11}" for c in chans)
              + f"{'total':>11}{'total RMSE':>12}")
        print("-" * (24 + 11 * (len(chans) + 1) + 12))
        for k, q in res["results"].items():
            if conv not in q:
                continue
            pc, ov = q[conv]["per_channel"], q[conv]["overall"]
            print(f"{k:<24}" + "".join(f"{pc[c]:>11.4f}" for c in chans)
                  + f"{ov:>11.4f}{np.sqrt(ov):>12.4f}")
        for label, e in res.get("seed_summary", {}).items():
            if conv not in e:
                continue
            q = e[conv]
            print(f"{'4DVarNet ' + label + ' single model':<24}"
                  + "".join(f"{q['per_channel_mean'][c]:>11.4f}" for c in chans)
                  + f"{q['overall_mean']:>11.4f}{np.sqrt(q['overall_mean']):>12.4f}"
                  + f"   ± {q['overall_std']:.4f} ({q['n_seeds']} seeds)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
