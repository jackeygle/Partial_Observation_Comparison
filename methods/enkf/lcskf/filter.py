"""Run a trained covariance U-Net as a sequential differentiable Kalman filter."""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from crowdcore import navigation
from methods.enkf.lcskf.kalman import analysis_update, posterior_diag
from methods.enkf.lcskf.covariance import (CovarianceUNet, STATE_SCALE,
                                            load_covariance_state)
from methods.enkf.lcskf.dynamics.model import (CLIP_HI, CLIP_LO, EMPTY_DENSITY,
                                          load_pedpred3, surrogate_mean)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def project_state(x):
    lo = x.new_tensor(CLIP_LO).view(1, 4, 1, 1)
    hi = x.new_tensor(CLIP_HI).view(1, 4, 1, 1)
    x = torch.maximum(torch.minimum(x, hi), lo)
    empty = x[:, 0:1] < EMPTY_DENSITY
    return torch.cat((x[:, 0:1], torch.where(empty, torch.zeros_like(x[:, 1:]), x[:, 1:])), dim=1)


def load_models(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device)
    args = ckpt["args"]
    cov = CovarianceUNet(
        rank=int(args["rank"]), width=int(args["width"]),
        diagonal_floor=float(args.get("diagonal_floor", 1e-5)),
        diagonal_floor_fraction=float(args.get("diagonal_floor_fraction", 0.0)),
        observation_mask=bool(args.get("cov_obs_mask", False)),
    ).to(device)
    load_covariance_state(cov, ckpt["model"])
    cov.eval()
    # Joint end-to-end checkpoints carry the fine-tuned PedPred3 alongside the
    # covariance model; frozen-mean ones intentionally do not and must read the file
    # their `mean_ckpt` names.  That path was recorded as the training job saw it, so
    # it can be relative to methods/enkf -- when the weights travel inside the
    # checkpoint anyway, do not make loading depend on finding it.
    mean_path = ckpt.get("mean_ckpt", args["mean_ckpt"])
    if "mean_model" in ckpt:
        mean = load_pedpred3(None, device).eval()
        mean.load_state_dict(ckpt["mean_model"])
        mean_path = f"{checkpoint}::mean_model"
    else:
        mean = load_pedpred3(mean_path, device).eval()
    return mean, cov, args, mean_path


def run_file(path, mean_net, cov_net, static_mask, device, frames, covariance_mode,
             factor_covariance_scale, diagonal_covariance_scale, input_frames=1,
             blind_diagonal_std_scale=1.0,
             blind_empty_diagonal_std_scale=1.0):
    """Filter one exported observation file. Thin loader over :func:`run_arrays`."""
    with np.load(path) as z:
        y_np, mask_np = z["Y"], z["Omega"]
        x0_np, obs_std_np = z["X0"], z["obs_std"]
    return run_arrays(y_np, mask_np, x0_np, obs_std_np, mean_net, cov_net, static_mask,
                      device, frames, covariance_mode, factor_covariance_scale,
                      diagonal_covariance_scale, input_frames,
                      blind_diagonal_std_scale, blind_empty_diagonal_std_scale)


def run_arrays(y_np, mask_np, x0_np, obs_std_np, mean_net, cov_net, static_mask, device,
               frames, covariance_mode, factor_covariance_scale, diagonal_covariance_scale,
               input_frames=1, blind_diagonal_std_scale=1.0,
               blind_empty_diagonal_std_scale=1.0):
    """The sequential filter itself, on in-memory observations.

    Separate from :func:`run_file` so callers that simulate observations on the fly
    (``checks/export_backgrounds.py``) need not persist them first.
    """
    total = min(len(y_np), frames) if frames else len(y_np)
    y_np, mask_np, x0_np = y_np[:total], mask_np[:total], x0_np[:total]
    obs_std = torch.from_numpy(obs_std_np).to(device)
    state = torch.from_numpy(x0_np[0:1]).to(device)
    # At the beginning of a day there are not yet ``input_frames`` analyses.
    # Repeat X0 for warm-up, then replace entries with each new analysis.
    history = [state] * input_frames
    estimates = np.empty((total, *state.shape[1:]), dtype=np.float32)
    spreads = np.empty_like(estimates)
    estimates[0] = state[0].cpu().numpy()
    # X0 is an observation-based fill rather than a covariance-model forecast.  A zero here
    # would make the uncertainty scorer's NLL explode for an implementation artifact, so use
    # the measured per-channel state scale for this single initialization frame.
    spreads[0] = np.asarray(STATE_SCALE, dtype=np.float32)[:, None, None]
    tic = time.time()
    with torch.inference_mode():
        for t in range(1, total):
            mean = surrogate_mean(mean_net, torch.stack(history[-input_frames:], dim=1))[:, 0]
            # ``state`` is the analysis for t-1, so the mask that goes with it is the
            # robots' footprint at t-1 -- the same pairing training uses.
            x_observed = (torch.from_numpy(mask_np[t - 1:t]).to(device)
                          if cov_net.observation_mask else None)
            factor, diagonal = cov_net(state, mean, static_mask, x_observed)
            if covariance_mode == "diagonal":
                factor = torch.zeros_like(factor)
            elif covariance_mode == "fixed-diagonal":
                channel_scale = factor.new_tensor(STATE_SCALE).view(1, 4, 1, 1)
                diagonal = channel_scale.square().expand_as(diagonal)
            # Scale B = alpha_U U U^T + alpha_d diag(d) without materializing B.
            factor = factor * factor_covariance_scale ** 0.5
            diagonal = diagonal * diagonal_covariance_scale
            observation = torch.from_numpy(y_np[t:t + 1]).to(device)
            mask = torch.from_numpy(mask_np[t:t + 1]).to(device)
            if blind_diagonal_std_scale != 1.0:
                blind_walkable = static_mask[None].bool() & ~mask.bool()
                diagonal = torch.where(
                    blind_walkable[:, None],
                    diagonal * blind_diagonal_std_scale ** 2,
                    diagonal)
            if blind_empty_diagonal_std_scale != 1.0:
                blind_walkable = static_mask[None].bool() & ~mask.bool()
                predicted_empty = mean[:, 0:1] < EMPTY_DENSITY
                empty_blind = blind_walkable[:, None] & predicted_empty
                diagonal = torch.where(
                    empty_blind.expand_as(diagonal),
                    diagonal * blind_empty_diagonal_std_scale ** 2,
                    diagonal)
            if covariance_mode == "mean":
                state = project_state(mean)
                variance = diagonal + factor.square().sum(dim=1)
            else:
                state = project_state(analysis_update(
                    mean, factor, diagonal, observation, mask, obs_std.square()))
                variance = posterior_diag(
                    mean, factor, diagonal, observation, mask, obs_std.square())
            history.append(state)
            estimates[t] = state[0].cpu().numpy()
            spreads[t] = variance[0].clamp_min(0).sqrt().cpu().numpy()
            if t in (9, 99, 999) or (t + 1) % 5000 == 0:
                print(f"  frame {t+1}/{total}: {(time.time()-tic)/(t+1)*1000:.1f} ms/frame", flush=True)
    return estimates, spreads, time.time() - tic


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--obs-dir", default="check_outputs/enkf_valid_k1")
    p.add_argument("--outdir", default="check_outputs/lowrank_unetkf_valid")
    p.add_argument("--only", default="")
    p.add_argument("--frames", type=int, default=0)
    p.add_argument("--covariance-mode", choices=("full", "diagonal", "fixed-diagonal", "mean"),
                   default="full", help="ablate the learned covariance at inference")
    p.add_argument("--covariance-scale", type=float, default=1.0,
                   help="positive multiplicative inflation/deflation applied to U U^T + diag(d)")
    p.add_argument("--factor-covariance-scale", type=float, default=None,
                   help="override --covariance-scale for the low-rank U U^T term")
    p.add_argument("--diagonal-covariance-scale", type=float, default=None,
                   help="override --covariance-scale for the diagonal d term")
    p.add_argument("--blind-diagonal-std-scale", type=float, default=1.0,
                   help="extra std multiplier for diag(d) at blind walkable cells")
    p.add_argument("--blind-empty-diagonal-std-scale", type=float, default=1.0,
                   help="extra std multiplier where blind and predicted density is empty")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.covariance_scale <= 0:
        p.error("--covariance-scale must be positive")
    factor_covariance_scale = (args.covariance_scale if args.factor_covariance_scale is None
                               else args.factor_covariance_scale)
    diagonal_covariance_scale = (
        args.covariance_scale if args.diagonal_covariance_scale is None
        else args.diagonal_covariance_scale)
    if factor_covariance_scale < 0:
        p.error("--factor-covariance-scale must be non-negative")
    if diagonal_covariance_scale <= 0:
        p.error("--diagonal-covariance-scale must be positive")
    if args.blind_diagonal_std_scale <= 0:
        p.error("--blind-diagonal-std-scale must be positive")
    if args.blind_empty_diagonal_std_scale <= 0:
        p.error("--blind-empty-diagonal-std-scale must be positive")
    device = torch.device(args.device)
    files = sorted(f for f in glob.glob(os.path.join(args.obs_dir, "obs_*.npz"))
                   if args.only in os.path.basename(f))
    if not files:
        raise SystemExit(f"no observation files in {args.obs_dir!r} matching {args.only!r}")
    os.makedirs(args.outdir, exist_ok=True)
    mean_net, cov_net, train_args, mean_path = load_models(args.checkpoint, device)
    static_mask = torch.from_numpy(navigation.build_valid_mask_from_config()).to(device)
    with open(os.path.join(args.outdir, "run_config.json"), "w") as f:
        json.dump({**vars(args), "mean_ckpt": mean_path, "train_args": train_args}, f, indent=2)
    for path in files:
        stem = os.path.basename(path)[4:-4]
        print(f"[{stem}] {path}", flush=True)
        estimate, spread, seconds = run_file(
            path, mean_net, cov_net, static_mask, device, args.frames,
            args.covariance_mode, factor_covariance_scale, diagonal_covariance_scale,
            int(train_args.get("input_frames", 1)), args.blind_diagonal_std_scale,
            args.blind_empty_diagonal_std_scale)
        np.savez_compressed(os.path.join(args.outdir, f"est_{stem}.npz"),
                            Est=estimate, Spread=spread)
        link = os.path.join(args.outdir, f"obs_{stem}.npz")
        if not os.path.exists(link):
            os.symlink(os.path.relpath(path, args.outdir), link)
        with open(os.path.join(args.outdir, f"timing_{stem}.json"), "w") as f:
            json.dump({"day": stem, "n_frames": len(estimate), "total_s": seconds,
                       "per_frame_s": seconds / max(len(estimate), 1), "device": str(device),
                       "covariance_scale": args.covariance_scale,
                       "factor_covariance_scale": factor_covariance_scale,
                       "diagonal_covariance_scale": diagonal_covariance_scale,
                       "blind_diagonal_std_scale": args.blind_diagonal_std_scale,
                       "blind_empty_diagonal_std_scale":
                           args.blind_empty_diagonal_std_scale}, f, indent=2)
    print("[done]")


if __name__ == "__main__":
    main()
