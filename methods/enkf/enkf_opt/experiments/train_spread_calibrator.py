"""Train a tiny residual MLP to calibrate posterior EnKF spread.

The model never sees truth as an input.  Truth is used only in the proper-score
loss and evaluation masks.  Input normalization is fitted on training exports;
validation exports are held out by file/day.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random

import numpy as np
import torch
from scipy.ndimage import uniform_filter

from methods.dincae.state import channel_valid


CHANNELS = ("density", "vx", "vy", "var")
SQRT_PI = math.sqrt(math.pi)


def gaussian_crps_torch(mu, sigma, truth):
    z = (truth - mu) / sigma
    pdf = torch.exp(-0.5 * z.square()) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    return sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / SQRT_PI)


def gaussian_crps_numpy(mu, sigma, truth):
    from scipy.special import erf
    z = (truth - mu) / sigma
    pdf = np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + erf(z / math.sqrt(2)))
    return sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / SQRT_PI)


def soft_reliability_loss(mu, sigma, truth, features, temperature):
    """Differentiable 10--90% coverage error, pooled and per channel.

    The final four feature columns are the channel one-hot encoding.  Penalizing
    channels separately prevents a too-wide channel from cancelling a too-narrow
    one in the pooled reliability curve.
    """
    z = (truth - mu).abs() / sigma.clamp_min(1e-8)
    normal = torch.distributions.Normal(
        torch.zeros((), device=z.device), torch.ones((), device=z.device))
    losses = []
    for nominal in torch.arange(.1, 1, .1, device=z.device):
        threshold = normal.icdf((1 + nominal) / 2)
        hit = torch.sigmoid((threshold - z) / temperature)
        losses.append((hit.mean() - nominal).square())
        for channel in range(4):
            mask = features[:, 5 + channel] > .5
            if mask.any():
                losses.append((hit[mask].mean() - nominal).square())
    return torch.stack(losses).mean()


class SpreadMLP(torch.nn.Module):
    def __init__(self, n_features, hidden=32, max_scale=4.0):
        super().__init__()
        self.max_log_scale = math.log(max_scale)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_features, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, 1),
        )
        # Exact identity calibration at initialization.
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.max_log_scale * torch.tanh(self.net(x).squeeze(-1))


def load_points(paths, max_per_stratum, seed, blind_only=False):
    """Load a balanced sample over channel x observation-age strata."""
    rng = np.random.default_rng(seed)
    buckets = []
    for path in paths:
        with np.load(path) as z:
            mu = z["mean"].astype(np.float32)
            sig = np.maximum(z["spread"].astype(np.float32), 1e-8)
            truth = z["truth"].astype(np.float32)
            observed = z["observed"].astype(bool)
            age = z["observation_age"].astype(np.float32)
            walkable = z["walkable"].astype(bool)
        valid = np.ascontiguousarray(channel_valid(truth).transpose(1, 0, 2, 3))
        valid &= walkable[None, None]
        # Causal local observation density; no truth enters this feature.
        local_obs = uniform_filter(observed.astype(np.float32), size=(1, 3, 3), mode="constant")
        for channel in range(4):
            for lo, hi in ((0, 0), (1, 10), (11, 50), (51, None)):
                if blind_only and lo == 0:
                    continue
                cell = age >= lo
                if hi is not None:
                    cell &= age <= hi
                idx = np.flatnonzero(valid[:, channel].reshape(-1) & cell.reshape(-1))
                if not len(idx):
                    continue
                if len(idx) > max_per_stratum:
                    idx = rng.choice(idx, max_per_stratum, replace=False)
                m = mu[:, channel].reshape(-1)[idx]
                s = sig[:, channel].reshape(-1)[idx]
                y = truth[:, channel].reshape(-1)[idx]
                a = age.reshape(-1)[idx]
                o = observed.reshape(-1)[idx].astype(np.float32)
                n = local_obs.reshape(-1)[idx]
                onehot = np.eye(4, dtype=np.float32)[np.full(len(idx), channel)]
                # Signed log mean handles velocities; other inputs are naturally bounded/logged.
                base = np.column_stack((np.sign(m) * np.log1p(np.abs(m)),
                                        np.log(s), np.log1p(a), o, n)).astype(np.float32)
                buckets.append((np.concatenate((base, onehot), axis=1), m, s, y,
                                np.full(len(idx), channel, np.int64), a))
    if not buckets:
        raise RuntimeError("no valid calibration samples")
    fields = [np.concatenate([row[i] for row in buckets]) for i in range(6)]
    order = rng.permutation(len(fields[1]))
    return tuple(field[order] for field in fields)


def calibration_metrics(mu, sigma, truth, channel, age):
    def one(mask):
        if not mask.any():
            return {"n": 0, "crps": None, "spread_skill": None,
                    "coverage": {}, "reliability_mae": None}
        m, s, y = mu[mask], np.maximum(sigma[mask], 1e-8), truth[mask]
        rmse = float(np.sqrt(np.mean((y - m) ** 2)))
        coverage = {}
        errors = []
        from scipy.stats import norm
        for nominal in np.arange(.1, 1, .1):
            hit = float((np.abs(y - m) <= norm.ppf((1 + nominal) / 2) * s).mean())
            coverage[str(int(round(100 * nominal)))] = hit
            errors.append(abs(hit - nominal))
        return {"n": int(mask.sum()), "crps": float(gaussian_crps_numpy(m, s, y).mean()),
                "spread_skill": float(s.mean() / rmse), "coverage": coverage,
                "reliability_mae": float(np.mean(errors))}
    out = {"all": one(np.ones(len(mu), bool)), "per_channel": {}, "observation_age": {}}
    for c, name in enumerate(CHANNELS):
        out["per_channel"][name] = one(channel == c)
    for name, lo, hi in (("observed", 0, 0), ("blind_1_10", 1, 10),
                         ("blind_11_50", 11, 50), ("blind_51_plus", 51, None)):
        mask = age >= lo
        if hi is not None:
            mask &= age <= hi
        out["observation_age"][name] = one(mask)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-stratum", type=int, default=30000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--max-scale", type=float, default=4.0)
    ap.add_argument("--reliability-weight", type=float, default=0.0)
    ap.add_argument("--reliability-temperature", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blind-only", action="store_true",
                    help="train and validate only on currently unobserved cells")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    train = load_points(args.train, args.max_per_stratum, args.seed, args.blind_only)
    val = load_points(args.val, args.max_per_stratum, args.seed + 1, args.blind_only)
    xtr = train[0]
    center, scale = xtr[:, :3].mean(0), np.maximum(xtr[:, :3].std(0), 1e-6)
    for data in (train, val):
        data[0][:, :3] = (data[0][:, :3] - center) / scale
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpreadMLP(train[0].shape[1], args.hidden, args.max_scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    tx = torch.from_numpy(train[0]); tm = torch.from_numpy(train[1])
    ts = torch.from_numpy(train[2]); ty = torch.from_numpy(train[3])
    vx = torch.from_numpy(val[0]).to(device)
    vm = torch.from_numpy(val[1]).to(device); vs = torch.from_numpy(val[2]).to(device)
    vy = torch.from_numpy(val[3]).to(device)
    best, best_state, history = float("inf"), None, []
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(tx))
        losses = []
        for start in range(0, len(tx), args.batch_size):
            idx = order[start:start + args.batch_size]
            xb, mb, sb, yb = tx[idx].to(device), tm[idx].to(device), ts[idx].to(device), ty[idx].to(device)
            log_scale = model(xb)
            calibrated = sb * log_scale.exp()
            crps_loss = gaussian_crps_torch(mb, calibrated, yb).mean()
            rel_loss = soft_reliability_loss(
                mb, calibrated, yb, xb, args.reliability_temperature)
            loss = (crps_loss + args.reliability_weight * rel_loss
                    + 0.002 * log_scale.square().mean())
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val_sigma = vs * model(vx).exp()
            val_crps = float(gaussian_crps_torch(vm, val_sigma, vy).mean())
            val_rel = float(soft_reliability_loss(
                vm, val_sigma, vy, vx, args.reliability_temperature))
            val_objective = val_crps + args.reliability_weight * val_rel
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                        "val_crps": val_crps, "val_reliability_loss": val_rel,
                        "val_objective": val_objective})
        print(f"epoch={epoch+1:02d} train={np.mean(losses):.6f} "
              f"val_crps={val_crps:.6f} val_rel={val_rel:.6f}", flush=True)
        if val_objective < best:
            best = val_objective
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        val_scale = model(vx).exp().cpu().numpy()
    raw = calibration_metrics(val[1], val[2], val[3], val[4], val[5])
    calibrated = calibration_metrics(val[1], val[2] * val_scale, val[3], val[4], val[5])
    result = {"config": vars(args), "device": str(device), "n_train": len(train[1]),
              "n_val": len(val[1]), "normalization": {"center": center.tolist(), "scale": scale.tolist()},
              "best_val_objective": best, "scale_summary": {"mean": float(val_scale.mean()),
              "p05": float(np.quantile(val_scale, .05)), "p50": float(np.quantile(val_scale, .5)),
              "p95": float(np.quantile(val_scale, .95))}, "raw": raw,
              "calibrated": calibrated, "history": history}
    os.makedirs(args.out, exist_ok=True)
    torch.save({"state_dict": best_state, "normalization_center": center,
                "normalization_scale": scale, "config": vars(args)}, os.path.join(args.out, "model.pt"))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"raw": raw["all"], "calibrated": calibrated["all"],
                      "scale": result["scale_summary"]}, indent=2))


if __name__ == "__main__":
    main()
