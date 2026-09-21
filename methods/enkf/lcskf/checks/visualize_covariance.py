"""Export and visualize learned state-conditional covariance on held-out frames."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from crowdcore import config, navigation
from methods.enkf.lcskf.filter import load_models
from methods.enkf.lcskf.dynamics.data import PairFrames
from methods.enkf.lcskf.dynamics.model import CH, surrogate_mean


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--candidates", type=int, default=2048)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--outdir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def masked(array, walkable):
    return np.where(walkable, array, np.nan)


def add_map(ax, array, walkable, title, cmap="viridis", symmetric=False,
            vmin=None, vmax=None, anchors=()):
    image = masked(array, walkable)
    finite = image[np.isfinite(image)]
    if symmetric and finite.size:
        limit = np.quantile(np.abs(finite), 0.99)
        limit = max(float(limit), 1e-12)
        vmin, vmax = -limit, limit
    im = ax.imshow(image, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    for row, col in anchors:
        ax.scatter(col, row, s=34, facecolors="none", edgecolors="white", linewidths=1.2)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)


def choose_anchors(mean_density, walkable, number=3, min_distance=6.0):
    candidates = np.argwhere(walkable)
    order = np.argsort(mean_density[walkable])[::-1]
    selected = []
    for index in order:
        point = tuple(int(v) for v in candidates[index])
        if all(np.linalg.norm(np.asarray(point) - np.asarray(other)) >= min_distance
               for other in selected):
            selected.append(point)
            if len(selected) == number:
                return selected
    for index in order:
        point = tuple(int(v) for v in candidates[index])
        if point not in selected:
            selected.append(point)
            if len(selected) == number:
                break
    return selected


def dense_parts(factor, diagonal):
    channels, height, width = diagonal.shape
    u = factor.permute(1, 2, 3, 0).reshape(channels * height * width, factor.shape[0])
    d = diagonal.reshape(-1)
    covariance = u @ u.T + torch.diag(d)
    variance = covariance.diagonal().reshape(channels, height, width)
    return u, d, covariance, variance


def covariance_row(covariance, variance_flat, channel, row, col, height, width):
    index = channel * height * width + row * width + col
    cov = covariance[:, index]
    corr = cov / torch.sqrt((variance_flat * variance_flat[index]).clamp_min(1e-20))
    return index, cov, corr


def plot_state_variance(record, walkable, anchors, out):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    for channel, name in enumerate(CH):
        add_map(axes[0, channel], record["mean"][channel], walkable,
                f"forecast mean: {name}", anchors=anchors)
        add_map(axes[1, channel], np.sqrt(record["variance"][channel]), walkable,
                f"prior std: {name}", cmap="magma", vmin=0, anchors=anchors)
    fig.suptitle(f"{record['label']} | activity={record['activity']:.2f}")
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_anchor_correlations(record, walkable, anchors, out):
    height, width = walkable.shape
    variance = record["variance_t"].reshape(-1)
    covariance = record["covariance_t"]
    fig, axes = plt.subplots(len(anchors), 4, figsize=(15, 3.5 * len(anchors)),
                             constrained_layout=True)
    for anchor_number, (row, col) in enumerate(anchors):
        _, _, correlation = covariance_row(
            covariance, variance, 0, row, col, height, width)
        maps = correlation.reshape(4, height, width).cpu().numpy()
        for target_channel, name in enumerate(CH):
            add_map(axes[anchor_number, target_channel], maps[target_channel], walkable,
                    f"density@({row},{col}) → {name}", cmap="RdBu_r",
                    symmetric=True, anchors=[(row, col)])
    fig.suptitle(f"Correlation maps | {record['label']}")
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_cross_channel(record, walkable, anchor, out):
    height, width = walkable.shape
    variance = record["variance_t"].reshape(-1)
    covariance = record["covariance_t"]
    fig, axes = plt.subplots(4, 4, figsize=(14, 13), constrained_layout=True)
    row, col = anchor
    for source_channel, source_name in enumerate(CH):
        _, _, correlation = covariance_row(
            covariance, variance, source_channel, row, col, height, width)
        maps = correlation.reshape(4, height, width).cpu().numpy()
        for target_channel, target_name in enumerate(CH):
            add_map(axes[source_channel, target_channel], maps[target_channel], walkable,
                    f"{source_name} → {target_name}", cmap="RdBu_r", symmetric=True,
                    anchors=[anchor])
    fig.suptitle(f"Cross-channel correlations at anchor {anchor} | {record['label']}")
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_gain(record, walkable, anchor, obs_variance, out):
    height, width = walkable.shape
    covariance = record["covariance_t"]
    variance = record["variance_t"].reshape(-1)
    row, col = anchor
    _, column, _ = covariance_row(covariance, variance, 0, row, col, height, width)
    source_index = row * width + col
    gain = column / (variance[source_index] + obs_variance[0])
    maps = gain.reshape(4, height, width).cpu().numpy()
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.5), constrained_layout=True)
    for channel, name in enumerate(CH):
        add_map(axes[channel], maps[channel], walkable,
                f"K: density obs → {name}", cmap="RdBu_r", symmetric=True,
                anchors=[anchor])
    fig.suptitle(f"Single-observation Kalman influence | {record['label']}")
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_lowrank_modes(record, walkable, out, count=4):
    u = record["u_t"].double()
    gram = u.T @ u
    values, vectors = torch.linalg.eigh(gram)
    order = torch.argsort(values, descending=True)[:count]
    modes = []
    selected_values = []
    for index in order:
        value = values[index].clamp_min(1e-30)
        modes.append((u @ vectors[:, index] / value.sqrt()).reshape(4, *walkable.shape))
        selected_values.append(float(value))
    fig, axes = plt.subplots(count, 4, figsize=(15, 3.2 * count), constrained_layout=True)
    for mode_number, mode in enumerate(modes):
        maps = mode.cpu().numpy()
        for channel, name in enumerate(CH):
            add_map(axes[mode_number, channel], maps[channel], walkable,
                    f"mode {mode_number + 1}: {name}", cmap="RdBu_r", symmetric=True)
    fig.suptitle(f"Low-rank covariance modes | eigenvalues={selected_values}")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return selected_values


def plot_state_comparison(records, walkable, anchor, out):
    height, width = walkable.shape
    density_limit = max(
        float(np.quantile(record["latest"][0][walkable], 0.99))
        for record in records
    )
    correlation_maps = []
    for record in records:
        variance = record["variance_t"].reshape(-1)
        _, _, correlation = covariance_row(
            record["covariance_t"], variance, 0, *anchor, height, width)
        correlation_maps.append(correlation.reshape(4, height, width)[0].cpu().numpy())
    correlation_limit = max(
        float(np.quantile(np.abs(correlation_map[walkable]), 0.99))
        for correlation_map in correlation_maps
    )
    correlation_limit = max(correlation_limit, 1e-12)
    fig, axes = plt.subplots(2, len(records), figsize=(5 * len(records), 7),
                             constrained_layout=True)
    for column, (record, corr_map) in enumerate(zip(records, correlation_maps)):
        add_map(axes[0, column], record["latest"][0], walkable,
                f"{record['label']} density", vmin=0, vmax=density_limit,
                anchors=[anchor])
        add_map(axes[1, column], corr_map, walkable,
                "density correlation", cmap="RdBu_r",
                vmin=-correlation_limit, vmax=correlation_limit, anchors=[anchor])
    fig.suptitle(
        f"Same anchor {anchor}, different activity states (shared row-wise color scales)")
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    if args.candidates < 3 or args.batch < 1:
        raise SystemExit("need at least three candidates and a positive batch")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    mean_net, cov_net, train_args, mean_path = load_models(args.checkpoint, device)
    history = int(train_args.get("input_frames", 1))
    data = PairFrames("valid", device, input_frames=history)
    number = min(args.candidates, data.n_pairs)
    rows = torch.linspace(0, data.n_pairs - 1, number, device=device).round().long().unique()
    walkable_t = torch.from_numpy(navigation.build_valid_mask_from_config()).to(device).bool()
    walkable = walkable_t.cpu().numpy()
    activities = []
    density_sum = torch.zeros_like(data.frames[0, 0], dtype=torch.float64)
    with torch.inference_mode():
        for begin in range(0, len(rows), args.batch):
            x, _ = data.batch(rows[begin:begin + args.batch])
            latest_density = x[:, -1, 0]
            activities.append(latest_density[:, walkable_t].sum(dim=1).cpu())
            density_sum += latest_density.double().sum(dim=0)
    activities = torch.cat(activities)
    order = torch.argsort(activities)
    quantiles = (("low", 0.10), ("medium", 0.50), ("high", 0.90))
    selected_rows = []
    for label, quantile in quantiles:
        position = round(quantile * (len(order) - 1))
        selected_rows.append((label, rows[order[position]], float(activities[order[position]])))
    anchors = choose_anchors((density_sum / len(rows)).cpu().numpy(), walkable)
    obs_std = torch.tensor(config.get("observation", "obs_std"), device=device)
    obs_variance = obs_std.square()

    records = []
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()), "mean_checkpoint": mean_path,
        "input_frames": history, "candidate_count": len(rows), "anchors": anchors,
        "frames": [],
    }
    with torch.inference_mode():
        for label, row_index, activity in selected_rows:
            x, truth = data.batch(row_index.view(1))
            mean = surrogate_mean(mean_net, x)[:, 0]
            factor, diagonal = cov_net(x[:, -1], mean, walkable_t)
            u, d, covariance, variance = dense_parts(factor[0], diagonal[0])
            record = {
                "label": label, "row": int(row_index), "target_index": int(data.index[row_index]),
                "activity": activity, "latest": x[0, -1].cpu().numpy(),
                "mean": mean[0].cpu().numpy(), "truth": truth[0, 0].cpu().numpy(),
                "variance": variance.cpu().numpy(), "u_t": u,
                "covariance_t": covariance, "variance_t": variance,
            }
            records.append(record)
            np.savez_compressed(
                outdir / f"covariance_{label}.npz",
                covariance=covariance.float().cpu().numpy(), factor=factor[0].cpu().numpy(),
                diagonal=diagonal[0].cpu().numpy(), latest=record["latest"],
                mean=record["mean"], truth=record["truth"], anchors=np.asarray(anchors),
                validation_row=int(row_index), target_index=record["target_index"])
            plot_state_variance(record, walkable, anchors, outdir / f"01_state_variance_{label}.png")
            plot_anchor_correlations(record, walkable, anchors,
                                     outdir / f"02_anchor_correlations_{label}.png")
            plot_cross_channel(record, walkable, anchors[0],
                               outdir / f"03_cross_channel_{label}.png")
            plot_gain(record, walkable, anchors[0], obs_variance,
                      outdir / f"04_kalman_influence_{label}.png")
            eigenvalues = plot_lowrank_modes(record, walkable,
                                             outdir / f"05_lowrank_modes_{label}.png")
            summary["frames"].append({
                "label": label, "validation_row": int(row_index),
                "target_index": record["target_index"], "activity": activity,
                "covariance_trace": float(variance.double().sum()),
                "lowrank_trace_fraction": float(factor[0].square().double().sum()
                                                / variance.double().sum()),
                "top_lowrank_eigenvalues": eigenvalues,
            })
    plot_state_comparison(records, walkable, anchors[0], outdir / "06_state_comparison.png")
    pairwise = {}
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            a, b = records[left], records[right]
            difference = torch.linalg.norm(a["covariance_t"] - b["covariance_t"])
            reference = 0.5 * torch.linalg.norm(a["covariance_t"] + b["covariance_t"])
            pairwise[f"{a['label']}_vs_{b['label']}"] = float(difference / reference)
    summary["pairwise_relative_covariance_frobenius_difference"] = pairwise
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
