"""Measure reconstruction error by time since a currently blind cell was last observed.

This is an evaluation-only diagnostic.  It answers where a temporal Senseiver
actually gains: cells seen very recently, cells seen earlier in its window, or
cells for which the window contains no direct observation.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from compare.compare5 import FRAME_HI_PAD, FRAME_LO, clip_np
from crowdcore import navigation as nav
from methods.senseiver import dataset as ds
from methods.senseiver import sensors
from methods.senseiver.network import Senseiver


AGE_BINS = (
    ("age_1", lambda a: a == 1),
    ("age_2_4", lambda a: (a >= 2) & (a <= 4)),
    ("age_5_8", lambda a: (a >= 5) & (a <= 8)),
    ("age_9_15", lambda a: (a >= 9) & (a <= 15)),
    ("age_16_plus", lambda a: a >= 16),
    ("never_seen", lambda a: a < 0),
    ("inside_k16", lambda a: (a >= 1) & (a <= 15)),
    ("outside_k16", lambda a: (a < 0) | (a >= 16)),
    ("all_blind", lambda a: np.ones_like(a, dtype=bool)),
)


def parse_model(spec):
    if "=" not in spec:
        raise argparse.ArgumentTypeError("model must be LABEL=CHECKPOINT")
    label, path = spec.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("model must be LABEL=CHECKPOINT")
    return label, path


def load_model(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    model = Senseiver(**ck["hparams"]).to(dev)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, ck


def observation_age(omega):
    """(T,H,W) -> 0 observed now; n>0 seconds since last seen; -1 never seen."""
    tmax, h, w = omega.shape
    age = np.full((tmax, h, w), -1, dtype=np.int32)
    last = np.full((h, w), -1, dtype=np.int32)
    for t in range(tmax):
        seen = omega[t]
        known = (~seen) & (last >= 0)
        age[t, known] = t - last[known]
        age[t, seen] = 0
        last[seen] = t
    return age


@torch.no_grad()
def infer(model, y, omega, dev, batch):
    pe = model.pos_enc.detach().cpu().numpy()
    mean = model.in_mean.detach().cpu().numpy()
    std = model.in_std.detach().cpu().numpy()
    out = []
    for start in range(0, y.shape[0], batch):
        stop = min(start + batch, y.shape[0])
        if model.time_window > 1:
            k = model.time_window
            windows = [(y[max(0, t - k + 1):t + 1],
                        omega[max(0, t - k + 1):t + 1])
                       for t in range(start, stop)]
            tok, pad, dt, _, cell = sensors.build_batch_temporal(
                windows, pe, mean, std, return_cell_idx=True)
            pred = model.reconstruct(tok.to(dev), pad.to(dev), dt.to(dev), cell.to(dev))
        else:
            tok, pad, _ = sensors.build_batch(y[start:stop], omega[start:stop],
                                               pe, mean, std)
            pred = model.reconstruct(tok.to(dev), pad.to(dev))
        out.append(pred.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def empty_acc(channels):
    return {name: {"se": np.zeros(channels, np.float64),
                   "n": np.zeros(channels, np.int64)} for name, _ in AGE_BINS}


def add_day(acc, pred, truth, omega, age, walk):
    # The official comparison ignores the warm-up and final padded interval.
    lo, hi = FRAME_LO, truth.shape[0] - FRAME_HI_PAD
    pred, truth = clip_np(pred[lo:hi]), truth[lo:hi]
    omega, age = omega[lo:hi], age[lo:hi]
    blind_walk = (~omega) & walk[None]
    se = (pred - truth) ** 2
    channels = truth.shape[1]
    for name, select_age in AGE_BINS:
        spatial = blind_walk & select_age(age)
        for c in range(channels):
            vals = se[:, c][spatial]
            acc[name]["se"][c] += vals.sum(dtype=np.float64)
            acc[name]["n"][c] += vals.size


def finish(acc, channel_names):
    result = {}
    for name, a in acc.items():
        mse_c = np.divide(a["se"], a["n"], out=np.full_like(a["se"], np.nan),
                          where=a["n"] > 0)
        total_n = int(a["n"].sum())
        overall = float(a["se"].sum() / total_n) if total_n else float("nan")
        result[name] = {
            "cell_frames": int(a["n"][0]),
            "channel_values": total_n,
            "mse": overall,
            "rmse": float(np.sqrt(overall)),
            "mse_per_channel": {c: float(v) for c, v in zip(channel_names, mse_c)},
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", type=parse_model, required=True,
                    help="repeatable LABEL=CHECKPOINT")
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    labels = [label for label, _ in args.model]
    if len(set(labels)) != len(labels):
        raise SystemExit("model labels must be unique")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = {}
    for label, path in args.model:
        model, ck = load_model(path, dev)
        loaded[label] = (model, path, ck.get("epoch"))
        print(f"[model] {label}: {path}, epoch={ck.get('epoch')}, "
              f"k={model.time_window}, params={model.num_params:,}", flush=True)

    days = ds.om.split_files("test")
    if args.days:
        days = days[:args.days]
    channels = ds.channels()
    c, h, w = ds.state_shape()
    walk = nav.build_valid_mask_from_config()
    accs = {label: empty_acc(c) for label in labels}
    per_day = []
    for day in days:
        x, y, omega = ds.load_day(day, stride=1, seed=ds.om.day_seed(day))
        truth = x.reshape(-1, c, h, w)
        omega = omega.reshape(-1, h, w)
        age = observation_age(omega)
        stem = os.path.basename(day).split("_")[0]
        day_row = {"day": stem}
        for label, (model, _, _) in loaded.items():
            pred = infer(model, y, omega.reshape(len(omega), -1), dev, args.batch)
            add_day(accs[label], pred, truth, omega, age, walk)
            # A concise progress number; final output remains pooled over days.
            tmp = finish(accs[label], channels)["all_blind"]["rmse"]
            day_row[label] = tmp
            del pred
        per_day.append(day_row)
        print(f"[day] {stem}: " + "  ".join(f"{k} pooled={v:.4f}"
              for k, v in day_row.items() if k != "day"), flush=True)

    models = {label: {"checkpoint": path, "checkpoint_epoch": epoch,
                      "metrics": finish(accs[label], channels)}
              for label, (_, path, epoch) in loaded.items()}
    out = {"split": "test", "days": len(days), "frame_lo": FRAME_LO,
           "frame_hi_pad": FRAME_HI_PAD, "models": models, "progress": per_day}
    if len(labels) >= 2:
        ref = labels[0]
        comparisons = {}
        for label in labels[1:]:
            comparisons[label] = {}
            for bin_name, _ in AGE_BINS:
                a = models[ref]["metrics"][bin_name]["rmse"]
                b = models[label]["metrics"][bin_name]["rmse"]
                comparisons[label][bin_name] = {
                    "reference": ref, "rmse_diff": b - a,
                    "rmse_change_pct": (b - a) / a * 100.0,
                }
        out["comparisons"] = comparisons

    print("\n[pooled blind-walkable RMSE by age]")
    print(f"{'age bin':<16}" + "".join(f"{label:>14}" for label in labels))
    for bin_name, _ in AGE_BINS:
        print(f"{bin_name:<16}" + "".join(
            f"{models[label]['metrics'][bin_name]['rmse']:>14.5f}" for label in labels))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
