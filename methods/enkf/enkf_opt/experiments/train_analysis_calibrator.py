"""One-step analysis-aware training of forecast-ensemble calibration.

Forecast snapshots are detached from the preceding sequence.  Gradients flow
through anomaly scaling, sample covariance, the ensemble-space Kalman solve and
the resulting analysis loss, but not through PedPred or earlier filter steps.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as Fnn

from methods.dincae.state import channel_valid
from methods.enkf.enkf_opt.experiments.train_spread_calibrator import (
    SpreadMLP, gaussian_crps_torch, soft_reliability_loss,
)


H, W, F = 36, 12, 4
TOTAL, STATE_DIM = H * W, F * H * W


def clip_bounds(x):
    y = x.reshape(-1, F, H, W)
    out = torch.empty_like(y)
    out[:, 0] = y[:, 0].clamp(0, 5)
    out[:, 1:3] = y[:, 1:3].clamp(-5, 5)
    out[:, 3] = y[:, 3].clamp(0, 2)
    return out.reshape(-1, STATE_DIM)


def localization_weights(obs_idx, radius, dtype):
    device = obs_idx.device
    cell = obs_idx.remainder(TOTAL)
    ro, co = torch.div(cell, W, rounding_mode="floor"), cell.remainder(W)
    rr = torch.arange(H, device=device)[:, None].expand(H, W).reshape(-1)
    cc = torch.arange(W, device=device)[None, :].expand(H, W).reshape(-1)
    dist2 = ((rr[None] - ro[:, None]).square()
             + (cc[None] - co[:, None]).square()).to(dtype)
    inside = dist2 <= radius * radius
    return torch.where(inside, torch.exp(-dist2 / (2 * radius * radius)),
                       torch.zeros((), device=device, dtype=dtype))


def load_banks(paths):
    rows = []
    for path in paths:
        with np.load(path) as z:
            members = z["members"].astype(np.float32)
            truth = z["truth"].astype(np.float32)
            observations = z["observations"].astype(np.float32)
            observed = z["observed"].astype(bool)
            age = z["observation_age"].astype(np.int32)
            obs_std = z["obs_std"].astype(np.float32)
            walkable = z["walkable"].astype(bool)
            frames = z["frame"].astype(np.int64)
        valid = np.ascontiguousarray(channel_valid(truth).transpose(1, 0, 2, 3))
        valid &= walkable[None, None]
        for i in range(len(members)):
            rows.append((members[i], truth[i], observations[i], observed[i], age[i],
                         obs_std, walkable, int(frames[i])))
    return rows


def raw_features(xf, observed, age):
    members = xf.reshape(xf.shape[0], F, H, W)
    mean = members.mean(0)
    spread = members.std(0, correction=0).clamp_min(1e-8)
    obs = observed.to(mean.dtype)
    local = Fnn.avg_pool2d(obs[None, None], 3, stride=1, padding=1,
                           count_include_pad=True)[0, 0]
    common = torch.stack((
        mean.sign() * torch.log1p(mean.abs()), spread.log(),
        torch.log1p(age.to(mean.dtype))[None].expand(F, -1, -1),
        obs[None].expand(F, -1, -1), local[None].expand(F, -1, -1),
    ), dim=-1)
    onehot = torch.eye(F, device=xf.device, dtype=xf.dtype)[:, None, None]
    onehot = onehot.expand(-1, H, W, -1)
    return torch.cat((common, onehot), dim=-1), mean


def fit_normalization(rows, device):
    sums = torch.zeros(3, device=device, dtype=torch.float64)
    squares = torch.zeros_like(sums)
    count = 0
    with torch.no_grad():
        for members, _, _, observed, age, *_ in rows:
            xf = torch.from_numpy(members).to(device)
            feat, _ = raw_features(xf, torch.from_numpy(observed).to(device),
                                   torch.from_numpy(age).to(device))
            x = feat[..., :3].reshape(-1, 3).double()
            sums += x.sum(0); squares += x.square().sum(0); count += len(x)
    center = sums / count
    scale = (squares / count - center.square()).clamp_min(1e-12).sqrt()
    return center.float(), scale.float()


def calibrate(model, xf, observed, age, walkable, center, norm):
    features, mean = raw_features(xf, observed, age)
    features[..., :3] = (features[..., :3] - center) / norm
    log_scale = model(features.reshape(-1, 9)).reshape(F, H, W)
    scale = log_scale.exp()
    eligible = (walkable & ~observed & (age >= 1))[None]
    scale = torch.where(eligible, scale, torch.ones_like(scale))
    members = xf.reshape(xf.shape[0], F, H, W)
    calibrated = mean[None] + scale[None] * (members - mean[None])
    return clip_bounds(calibrated.reshape(xf.shape[0], STATE_DIM)), log_scale, features


def analysis_update(xf, observations, observed, obs_std, cross, radius, seed):
    cells = torch.nonzero(observed.reshape(-1), as_tuple=False).flatten()
    if not len(cells):
        return xf
    obs_idx = (torch.arange(F, device=xf.device)[:, None] * TOTAL + cells[None])
    obs_idx = obs_idx.T.reshape(-1)
    y_obs = observations.reshape(-1)[obs_idx]
    yf = xf[:, obs_idx]
    ym, xm = yf.mean(0), xf.mean(0)
    xa0, ya = xf - xm, yf - ym
    rdiag = obs_std.repeat_interleave(TOTAL)[obs_idx].square()
    d = rdiag + 1e-3
    md = (ya / (xf.shape[0] - 1)) / d
    amat = md @ ya.T
    amat.diagonal().add_(1.0)
    z = torch.linalg.solve(amat, md)
    loc = localization_weights(obs_idx, radius, xf.dtype)
    obs_channel = torch.div(obs_idx, TOTAL, rounding_mode="floor")
    gain = ((xa0.T @ z).reshape(F, TOTAL, -1) * loc.T[None]
            * cross[:, obs_channel][:, None]).reshape(STATE_DIM, -1)
    # Fixed antithetic perturbations give deterministic gradients and exactly
    # zero observation-noise mean for the 100-member ensemble.
    generator = torch.Generator(device=xf.device).manual_seed(seed)
    half = xf.shape[0] // 2
    eps = torch.randn((half, len(obs_idx)), generator=generator, device=xf.device)
    eps = torch.cat((eps, -eps), 0) * rdiag.sqrt()
    xa = xf + (y_obs[None] + eps - yf) @ gain.T
    mean = xa.mean(0)
    xa = mean + 1.02 * (xa - mean)
    return clip_bounds(xa)


def snapshot_loss(xa, truth, valid, blind, features, log_scale,
                  raw_reference=None, reliability_weight=.1):
    mean = xa.mean(0).reshape(F, H, W)
    spread = xa.std(0, correction=0).reshape(F, H, W).clamp_min(1e-8)
    mse, crps = [], []
    for channel in range(F):
        mask = valid[channel] & blind
        if mask.any():
            mse.append((mean[channel][mask] - truth[channel][mask]).square().mean())
            crps.append(gaussian_crps_torch(
                mean[channel][mask], spread[channel][mask], truth[channel][mask]).mean())
        else:
            # Some low-density frames have no defined velocity/variance cells.
            # Keep fixed four-channel output shape and exclude these entries
            # from both training ratios and epoch aggregation.
            mse.append(torch.full((), float("nan"), device=xa.device))
            crps.append(torch.full((), float("nan"), device=xa.device))
    mse, crps = torch.stack(mse), torch.stack(crps)
    if raw_reference is None:
        return mse.detach(), crps.detach(), mean, spread
    raw_mse, raw_crps = raw_reference
    keep_mse = torch.isfinite(mse) & torch.isfinite(raw_mse)
    keep_crps = torch.isfinite(crps) & torch.isfinite(raw_crps)
    score = (mse[keep_mse] / raw_mse[keep_mse].clamp_min(1e-8)).mean()
    score = score + .5 * (
        crps[keep_crps] / raw_crps[keep_crps].clamp_min(1e-8)).mean()
    flat_mask = valid & blind[None]
    reliability = soft_reliability_loss(
        mean[flat_mask], spread[flat_mask], truth[flat_mask],
        features[flat_mask], temperature=.08)
    regularizer = log_scale[flat_mask].square().mean()
    return score + reliability_weight * reliability + .01 * regularizer, mse, crps


def evaluate(model, rows, center, norm, cross, radius, device):
    totals = {"raw_mse": [], "cal_mse": [], "raw_crps": [], "cal_crps": []}
    model.eval()
    with torch.no_grad():
        for members, truth, observations, observed, age, obs_std, walkable, frame in rows:
            xf = torch.from_numpy(members).to(device)
            yt = torch.from_numpy(truth).to(device)
            ob = torch.from_numpy(observed).to(device)
            ag = torch.from_numpy(age).to(device)
            ws = torch.from_numpy(walkable).to(device)
            osd = torch.from_numpy(obs_std).to(device)
            yo = torch.from_numpy(observations).to(device)
            valid = torch.from_numpy(np.ascontiguousarray(
                channel_valid(truth[None]).transpose(1, 0, 2, 3)[0])).to(device) & ws[None]
            blind = ws & ~ob
            raw_a = analysis_update(xf, yo, ob, osd, cross, radius, frame + 17)
            ref = snapshot_loss(raw_a, yt, valid, blind, None, None)
            cal_xf, log_s, feat = calibrate(model, xf, ob, ag, ws, center, norm)
            cal_a = analysis_update(cal_xf, yo, ob, osd, cross, radius, frame + 17)
            _, cm, cc = snapshot_loss(cal_a, yt, valid, blind, feat, log_s, ref[:2], 0)
            for key, value in (("raw_mse", ref[0]), ("raw_crps", ref[1]),
                               ("cal_mse", cm), ("cal_crps", cc)):
                totals[key].append(value.cpu())
    return {key: torch.nanmean(torch.stack(value), dim=0).tolist()
            for key, value in totals.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--max-scale", type=float, default=1.5)
    ap.add_argument("--reliability-weight", type=float, default=.1)
    ap.add_argument("--radius", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val = load_banks(args.train), load_banks(args.val)
    center, norm = fit_normalization(train, device)
    model = SpreadMLP(9, args.hidden, args.max_scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    cross = torch.tensor(((1, .5, .5, .1), (.5, 1, .5, .1),
                          (.5, .5, 1, .1), (.1, .1, .1, 1)),
                         device=device, dtype=torch.float32)
    history, best, best_state = [], float("inf"), None
    for epoch in range(args.epochs):
        model.train(); random.shuffle(train); losses = []
        for members, truth, observations, observed, age, obs_std, walkable, frame in train:
            xf = torch.from_numpy(members).to(device)
            yt = torch.from_numpy(truth).to(device)
            yo = torch.from_numpy(observations).to(device)
            ob = torch.from_numpy(observed).to(device)
            ag = torch.from_numpy(age).to(device)
            osd = torch.from_numpy(obs_std).to(device)
            ws = torch.from_numpy(walkable).to(device)
            valid = torch.from_numpy(np.ascontiguousarray(
                channel_valid(truth[None]).transpose(1, 0, 2, 3)[0])).to(device) & ws[None]
            blind = ws & ~ob
            with torch.no_grad():
                raw_a = analysis_update(xf, yo, ob, osd, cross, args.radius, frame + 17)
                ref = snapshot_loss(raw_a, yt, valid, blind, None, None)[:2]
            cal_xf, log_s, feat = calibrate(model, xf, ob, ag, ws, center, norm)
            cal_a = analysis_update(cal_xf, yo, ob, osd, cross, args.radius, frame + 17)
            loss, _, _ = snapshot_loss(cal_a, yt, valid, blind, feat, log_s, ref,
                                       args.reliability_weight)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step(); losses.append(float(loss.detach()))
        metrics = evaluate(model, val, center, norm, cross, args.radius, device)
        objective = float(np.mean(np.asarray(metrics["cal_mse"]) /
                                  np.maximum(metrics["raw_mse"], 1e-8))
                          + .5 * np.mean(np.asarray(metrics["cal_crps"]) /
                                         np.maximum(metrics["raw_crps"], 1e-8)))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                        "val_objective": objective, "metrics": metrics})
        print(f"epoch={epoch+1:02d} train={np.mean(losses):.6f} "
              f"val_objective={objective:.6f}", flush=True)
        if objective < best:
            best = objective
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("training produced no finite validation checkpoint")
    model.load_state_dict(best_state)
    final = evaluate(model, val, center, norm, cross, args.radius, device)
    os.makedirs(args.out, exist_ok=True)
    saved = {"state_dict": best_state, "normalization_center": center.cpu().numpy(),
             "normalization_scale": norm.cpu().numpy(), "config": vars(args)}
    torch.save(saved, os.path.join(args.out, "model.pt"))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"config": vars(args), "n_train": len(train), "n_val": len(val),
                   "best_objective": best, "final": final, "history": history}, f, indent=2)
    print(json.dumps({"best_objective": best, "final": final}, indent=2))


if __name__ == "__main__":
    main()
