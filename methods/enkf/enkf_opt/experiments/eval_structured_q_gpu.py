"""CUDA implementation of the structured-Q 100-member localized EnKF.

Unlike merely moving PedPred3 to CUDA, this keeps the ensemble, process-noise
sampling, ensemble-space Kalman solve, localization and RTPS on the GPU.  Only
the per-frame posterior mean/spread are copied to host for final scoring.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as torch_f

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
OPT = os.path.join(ROOT, "methods", "enkf", "enkf_opt")
sys.path.insert(0, OPT)

from crowdcore import navigation, paths  # noqa: E402
from methods.enkf.checks.diag_proc_scale_sweep import (  # noqa: E402
    CH, F, H, INIT_STD, PROC_STD, STATE_DIM, TOTAL, W, score_arm,
)
from compare import score_uncertainty as su  # noqa: E402
from methods.enkf.checks.diag_proc_scale_sweep import _defined  # noqa: E402
from methods.enkf.enkf_opt.experiments.train_spread_calibrator import SpreadMLP  # noqa: E402
from methods.enkf.lcskf.dynamics.model import load_pedpred3, surrogate_mean  # noqa: E402


def clip_bounds(x):
    y = x.reshape(-1, F, H, W)
    out = torch.empty_like(y)
    out[:, 0] = y[:, 0].clamp(0, 5)
    out[:, 1:3] = y[:, 1:3].clamp(-5, 5)
    out[:, 3] = y[:, 3].clamp(0, 2)
    return out.reshape(-1, STATE_DIM)


def localization_weights(obs_idx, radius, dtype):
    """Correct four-channel localization, shape (observations, H*W)."""
    device = obs_idx.device
    cell = obs_idx.remainder(TOTAL)
    ro, co = torch.div(cell, W, rounding_mode="floor"), cell.remainder(W)
    rr = torch.arange(H, device=device)[:, None].expand(H, W).reshape(-1)
    cc = torch.arange(W, device=device)[None, :].expand(H, W).reshape(-1)
    dr = rr[None] - ro[:, None]
    dc = cc[None] - co[:, None]
    dist2 = (dr * dr + dc * dc).to(dtype)
    inside = dist2 <= radius * radius
    return torch.where(inside, torch.exp(-dist2 / (2.0 * radius * radius)),
                       torch.zeros((), device=device, dtype=dtype))


def detailed_diagnostics(est, spread, truth, omega, warmup):
    """Per-channel calibration plus calibration versus time since last observation."""
    mu, sig, x = est[warmup:], np.maximum(spread[warmup:].astype(np.float64), 1e-12), truth[warmup:]
    om = omega[warmup:].astype(bool)
    dfn = _defined(x)

    def metrics(sel):
        acc = su.Accumulator()
        acc.add(mu[sel], sig[sel], x[sel])
        return acc.result()

    out = {"per_channel": {}}
    for c, name in enumerate(CH):
        def channel_metrics(mask):
            acc = su.Accumulator()
            acc.add(mu[:, c][mask], sig[:, c][mask], x[:, c][mask])
            return acc.result()
        out["per_channel"][name] = {
            "defined": channel_metrics(dfn[:, c]),
            "defined_blind": channel_metrics(dfn[:, c] & ~om),
            "defined_observed": channel_metrics(dfn[:, c] & om),
        }

    # Density calibration conditioned on the model estimate.  These bins are
    # diagnostic only (truth is used for scoring, never for Q scaling) and expose
    # whether collapse comes from forecast false negatives or populated cells.
    out["density_prediction_bins"] = {}
    density_mu = mu[:, 0]
    density_defined_blind = dfn[:, 0] & ~om
    for name, lo, hi in (("zero_005", 0.0, 0.005),
                         ("low_005_05", 0.005, 0.05),
                         ("mid_05_2", 0.05, 0.2),
                         ("occupied_2_5", 0.2, 0.5),
                         ("high_5_plus", 0.5, None)):
        sel = density_defined_blind & (density_mu >= lo)
        if hi is not None:
            sel &= density_mu < hi
        acc = su.Accumulator()
        acc.add(mu[:, 0][sel], sig[:, 0][sel], x[:, 0][sel])
        out["density_prediction_bins"][name] = acc.result()

    false_negative = density_defined_blind & (density_mu < 0.05) & (x[:, 0] >= 0.2)
    acc = su.Accumulator()
    acc.add(mu[:, 0][false_negative], sig[:, 0][false_negative], x[:, 0][false_negative])
    out["density_prediction_bins"]["false_negative_mu_lt05_truth_ge2"] = acc.result()

    # Age is computed over the complete prefix so the warmup boundary does not reset history.
    age = np.zeros_like(omega, dtype=np.int32)
    for t in range(1, len(omega)):
        age[t] = np.where(omega[t], 0, age[t - 1] + 1)
    age = age[warmup:]
    out["observation_age"] = {}
    for name, lo, hi in (("observed", 0, 0), ("blind_1_10", 1, 10),
                         ("blind_11_50", 11, 50), ("blind_51_plus", 51, None)):
        cell = age >= lo
        if hi is not None:
            cell &= age <= hi
        sel = dfn & cell[:, None]
        out["observation_age"][name] = metrics(sel)
    return out


def density_calibration_grid(est, spread, truth, omega, warmup, scores):
    """Validation grid for a report-only, causal density spread calibration.

    The posterior mean and filter trajectory are untouched.  We replace only
    density sigma on blind walkable cells and reconstruct pooled CRPS/coverage
    from sufficient sums, so the other channels are not repeatedly rescored.
    """
    from scipy.ndimage import maximum_filter

    mu4, sig4, x4 = est[warmup:], np.maximum(spread[warmup:], 1e-12), truth[warmup:]
    om = omega[warmup:].astype(bool)
    sel = _defined(x4)[:, 0] & ~om
    mu, sig, x = mu4[:, 0][sel], sig4[:, 0][sel], x4[:, 0][sel]
    old_crps = float(su.crps_gaussian(mu, sig, x).mean())
    old_hit90 = float((np.abs(x - mu) <= 1.6448536269514722 * sig).mean())
    n_sub = int(sel.sum())
    db_n, all_n = scores["defined_blind"]["n"], scores["all"]["n"]
    risks = {0: mu4[:, 0]}
    risks[1] = maximum_filter(mu4[:, 0], size=(1, 3, 3), mode="constant", cval=0.0)

    out = {}
    for radius, risk4 in risks.items():
        risk = risk4[sel]
        for threshold in (0.1, 0.2, 0.4):
            ramp = np.clip(risk / threshold, 0, 1)
            for gain in (0.25, 0.5, 0.75, 1.0, 1.5):
                calibrated = sig * (1 + gain * ramp)
                crps = float(su.crps_gaussian(mu, calibrated, x).mean())
                hit90 = float((np.abs(x - mu) <=
                               1.6448536269514722 * calibrated).mean())
                key = f"r{radius}_t{threshold:g}_g{gain:g}"
                out[key] = {
                    "radius": radius,
                    "threshold": threshold,
                    "gain": gain,
                    "density_db_crps": crps,
                    "density_db_sigma_mean": float(calibrated.mean()),
                    "density_db_spread_skill": float(
                        calibrated.mean() / np.sqrt(np.mean((x - mu) ** 2))),
                    "density_db_coverage90": hit90,
                    "defined_blind_crps": float(
                        (scores["defined_blind"]["crps"] * db_n
                         + (crps - old_crps) * n_sub) / db_n),
                    "defined_blind_coverage90": float(
                        (scores["defined_blind"]["coverage"][90] * db_n
                         + (hit90 - old_hit90) * n_sub) / db_n),
                    "all_crps": float(
                        (scores["all"]["crps"] * all_n
                         + (crps - old_crps) * n_sub) / all_n),
                }
    return out


def apply_density_report_calibration(est, spread, omega, gain, threshold, radius):
    """Calibrate posterior density sigma without changing the EnKF trajectory."""
    from scipy.ndimage import maximum_filter

    risk = est[:, 0]
    if radius:
        risk = maximum_filter(risk, size=(1, 2 * radius + 1, 2 * radius + 1),
                              mode="constant", cval=0.0)
    factor = 1 + gain * np.clip(risk / threshold, 0, 1)
    walkable = navigation.build_valid_mask_from_config()
    blind_walkable = ~omega.astype(bool) & walkable[None]
    calibrated = spread.copy()
    calibrated[:, 0][blind_walkable] *= factor[blind_walkable]
    return calibrated


class OnlineForecastCalibrator:
    """Causal anomaly rescaling before the EnKF observation update."""
    def __init__(self, checkpoint, device, strength, min_age, channels):
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        cfg = saved["config"]
        self.model = SpreadMLP(9, cfg["hidden"], cfg["max_scale"]).to(device)
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()
        self.center = torch.as_tensor(saved["normalization_center"], device=device)
        self.norm = torch.as_tensor(saved["normalization_scale"], device=device)
        self.strength = strength
        self.min_age = min_age
        self.channels = torch.as_tensor(channels, device=device, dtype=torch.bool)

    @torch.inference_mode()
    def __call__(self, ensemble, observed, age, walkable):
        members = ensemble.reshape(ensemble.shape[0], F, H, W)
        mean = members.mean(0)
        spread = members.std(0, correction=0).clamp_min(1e-8)
        obs2 = observed.reshape(H, W).to(mean.dtype)
        local_obs = torch_f.avg_pool2d(
            obs2[None, None], kernel_size=3, stride=1, padding=1,
            count_include_pad=True)[0, 0]
        common = torch.stack((
            mean.sign() * torch.log1p(mean.abs()),
            spread.log(),
            torch.log1p(age.reshape(H, W).to(mean.dtype))[None].expand(F, -1, -1),
            obs2[None].expand(F, -1, -1),
            local_obs[None].expand(F, -1, -1),
        ), dim=-1)
        common[..., :3] = (common[..., :3] - self.center) / self.norm
        channel = torch.eye(F, device=mean.device, dtype=mean.dtype)[:, None, None]
        channel = channel.expand(-1, H, W, -1)
        features = torch.cat((common, channel), dim=-1).reshape(-1, 9)
        learned_scale = self.model(features).exp().reshape(F, H, W)
        scale = 1 + self.strength * (learned_scale - 1)
        # The successful offline constraint: never alter currently observed
        # cells or obstacles.  Only blind walkable state anomalies are scaled.
        blind = (walkable.reshape(H, W) & ~observed.reshape(H, W)
                 & (age.reshape(H, W).to(torch.int32) >= self.min_age))
        eligible = blind[None] & self.channels[:, None, None]
        scale = torch.where(eligible, scale, torch.ones_like(scale))
        calibrated = mean[None] + scale[None] * (members - mean[None])
        return clip_bounds(calibrated.reshape(ensemble.shape[0], STATE_DIM)), scale


class TorchEnKF:
    def __init__(self, model, ensemble, obs_std, radius, seed, device,
                 cross_channel_matrix, input_frames=1):
        self.model = model
        self.n = ensemble
        self.radius = radius
        self.device = device
        self.dtype = torch.float32
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.cross_channel = torch.as_tensor(
            cross_channel_matrix, device=device, dtype=self.dtype).reshape(F, F)
        self.bias = torch.zeros(STATE_DIM, device=device, dtype=self.dtype)
        self.proc = torch.as_tensor(PROC_STD, device=device, dtype=self.dtype).repeat_interleave(TOTAL)
        self.obs_std = torch.as_tensor(obs_std, device=device,
                                       dtype=self.dtype).repeat_interleave(TOTAL)
        rng = np.random.RandomState(0)
        init = np.zeros((ensemble, STATE_DIM), np.float32)
        for f, std in enumerate(PROC_STD):
            init[:, f * TOTAL:(f + 1) * TOTAL] = rng.normal(
                0, std, size=(ensemble, TOTAL))
        self.x = clip_bounds(torch.from_numpy(init).to(device))
        self.input_frames = int(input_frames)
        self.history = self.x.reshape(self.n, 1, F, H, W).repeat(
            1, self.input_frames, 1, 1, 1)

    @torch.inference_mode()
    def forecast(self):
        pred = surrogate_mean(self.model, self.history, horizon=1)[:, 0]
        pred = pred.reshape(self.n, STATE_DIM) - self.bias[None]
        return clip_bounds(pred)

    @torch.inference_mode()
    def commit_analysis(self):
        """Append the posterior state to the causal dynamics history."""
        newest = self.x.reshape(self.n, 1, F, H, W)
        self.history = torch.cat((self.history[:, 1:], newest), dim=1)

    @torch.inference_mode()
    def update(self, xf, obs_idx, y_obs, regularization=1e-3):
        yf = xf[:, obs_idx]
        ym = yf.mean(0)
        xm = xf.mean(0)
        obs_bias = ym - y_obs
        nz = obs_bias != 0
        # obs_idx is unique for this observation generator, so direct indexed EMA matches scatter.
        idx = obs_idx[nz]
        self.bias[idx] = 0.95 * self.bias[idx] + 0.05 * obs_bias[nz]
        xa0 = xf - xm
        ya = yf - ym
        rdiag = self.obs_std[obs_idx].square()
        d = rdiag + regularization
        md = (ya / (self.n - 1)) / d
        amat = md @ ya.T
        amat.diagonal().add_(1.0)
        z = torch.linalg.solve(amat, md)
        loc = localization_weights(obs_idx, self.radius, xf.dtype)
        gain0 = xa0.T @ z
        obs_channel = torch.div(obs_idx, TOTAL, rounding_mode="floor")
        cross = self.cross_channel[:, obs_channel]
        gain = (gain0.reshape(F, TOTAL, -1) * loc.T[None] *
                cross[:, None]).reshape(STATE_DIM, -1)
        noise = torch.randn((self.n, len(obs_idx)), generator=self.gen,
                            device=self.device, dtype=self.dtype) * rdiag.sqrt()
        xa = xf + (y_obs[None] + noise - yf) @ gain.T
        mean = xa.mean(0)
        xa = mean + 1.02 * (xa - mean)
        self.x = clip_bounds(xa)

    @torch.inference_mode()
    def rtps(self, xf, weight):
        if weight <= 0:
            return
        mean = self.x.mean(0)
        ano = self.x - mean
        sf = xf.std(0, correction=0)
        sa = self.x.std(0, correction=0)
        target = (1 - weight) * sa + weight * sf
        factor = torch.ones_like(sa)
        good = sa > 1e-10
        factor[good] = (target[good] / sa[good]).clamp(max=10)
        self.x = clip_bounds(mean + ano * factor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise-kind", choices=("gaussian", "residual"), required=True)
    ap.add_argument("--density-noise-space", choices=("physical", "log1p"),
                    default="physical")
    ap.add_argument("--residual-conditioning", choices=("none", "density4"),
                    default="none",
                    help="training-only bank selection from forecast-density quartiles")
    ap.add_argument("--log-density-mean-correction", choices=("none", "ensemble"),
                    default="none",
                    help="remove the physical-space ensemble-mean shift after expm1")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--rtps", type=float, default=0.0)
    ap.add_argument("--frames", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ensemble", type=int, default=100)
    ap.add_argument("--model-checkpoint", default=os.path.join(
        ROOT, "methods", "enkf", "runs", "pedpred3_5to5_clip_s0", "dyn150_best.pt"))
    ap.add_argument("--input-frames", type=int, default=5)
    ap.add_argument("--radius", type=int, default=7)
    ap.add_argument("--cross-channel-matrix", type=float, nargs=16,
                    default=(1, 1, 1, 1,
                             1, 1, 1, 1,
                             1, 1, 1, 1,
                             1, 1, 1, 1),
                    metavar="W",
                    help="row-major state-channel x observation-channel gain taper")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--day", default="atc-20130811")
    ap.add_argument("--obs-dir", default=paths.enkf_export("enkf_k1_full"))
    ap.add_argument("--temporal-rho", type=float, default=0.0,
                    help="AR(1) persistence of process perturbations; variance is normalized")
    ap.add_argument("--channel-scales", type=float, nargs=4, default=(1, 1, 1, 1))
    ap.add_argument("--blind-channel-scales", type=float, nargs=4,
                    default=(1, 1, 1, 1),
                    help="extra causal Q multiplier on currently unobserved walkable cells")
    ap.add_argument("--density-state-gain", type=float, default=0.0,
                    help="maximum extra density-Q multiplier is 1 + this gain")
    ap.add_argument("--density-state-threshold", type=float, default=0.2,
                    help="forecast density where the state-dependent multiplier saturates")
    ap.add_argument("--density-risk-radius", type=int, default=0,
                    help="max-pool radius used to propagate forecast density risk to neighbours")
    ap.add_argument("--density-calibration-grid", action="store_true",
                    help="score a validation-only posterior density-sigma calibration grid")
    ap.add_argument("--density-report-gain", type=float, default=0.0,
                    help="report-only posterior density-sigma gain; does not feed back")
    ap.add_argument("--density-report-threshold", type=float, default=0.4)
    ap.add_argument("--density-report-radius", type=int, default=0)
    ap.add_argument("--bank", default=os.path.join(OPT, "experiments", "outputs",
                                                    "pedpred3_train_residual_q_8192.npz"))
    ap.add_argument("--calibration-export", default=None,
                    help="optional NPZ containing posterior mean/spread and causal calibration features")
    ap.add_argument("--forecast-calibration-export", default=None,
                    help="optional NPZ of the raw pre-update forecast for calibrator training")
    ap.add_argument("--analysis-snapshot-out", default=None,
                    help="optional NPZ of sampled forecast ensembles for differentiable one-step training")
    ap.add_argument("--analysis-snapshot-count", type=int, default=256)
    ap.add_argument("--forecast-calibrator", default=None,
                    help="checkpoint used to rescale forecast anomalies before every EnKF update")
    ap.add_argument("--forecast-calibration-strength", type=float, default=1.0,
                    help="damping alpha: effective scale = 1 + alpha * (network scale - 1)")
    ap.add_argument("--forecast-calibration-min-age", type=int, default=1,
                    help="minimum causal observation age at which online scaling is applied")
    ap.add_argument("--forecast-calibration-channels", type=int, nargs=4,
                    default=(1, 1, 1, 1), metavar="ON",
                    help="four 0/1 switches for density, vx, vy and variance")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA backend requested, but torch.cuda.is_available() is false")
    if a.warmup >= a.frames:
        raise ValueError("--warmup must be smaller than --frames")
    if not 0 <= a.temporal_rho < 1:
        raise ValueError("--temporal-rho must be in [0, 1)")
    if min(a.channel_scales) <= 0 or min(a.blind_channel_scales) <= 0:
        raise ValueError("channel scales must be positive")
    if a.density_state_gain < 0 or a.density_state_threshold <= 0:
        raise ValueError("density state gain must be nonnegative and threshold positive")
    if a.density_risk_radius < 0:
        raise ValueError("density risk radius must be nonnegative")
    if a.residual_conditioning != "none" and a.noise_kind != "residual":
        raise ValueError("residual conditioning requires --noise-kind residual")
    if (a.density_report_gain < 0 or a.density_report_threshold <= 0
            or a.density_report_radius < 0):
        raise ValueError("invalid density report calibration parameters")
    if not 0 <= a.forecast_calibration_strength <= 1:
        raise ValueError("forecast calibration strength must be in [0, 1]")
    if a.forecast_calibration_min_age < 1:
        raise ValueError("forecast calibration minimum age must be at least 1")
    if any(value not in (0, 1) for value in a.forecast_calibration_channels):
        raise ValueError("forecast calibration channel switches must be 0 or 1")
    if a.analysis_snapshot_count <= 0:
        raise ValueError("analysis snapshot count must be positive")
    device = torch.device("cuda")

    obs_path = os.path.join(a.obs_dir, f"obs_{a.day}.npz")
    with np.load(obs_path) as z:
        obs_std = z["obs_std"]
        T = min(a.frames, len(z["Omega"]))
        omega, x_true, y = z["Omega"][:T], z["X_true"][:T], z["Y"][:T]
    if a.input_frames < 1:
        raise ValueError("--input-frames must be positive")
    model = load_pedpred3(a.model_checkpoint, device).eval()
    cross = np.asarray(a.cross_channel_matrix, dtype=np.float32).reshape(F, F)
    if not np.isfinite(cross).all() or (cross < 0).any() or (cross > 1).any():
        raise ValueError("cross-channel matrix entries must be finite and in [0, 1]")
    filt = TorchEnKF(model, a.ensemble, obs_std, a.radius, a.seed, device, cross,
                     input_frames=a.input_frames)
    noise_gen = torch.Generator(device=device).manual_seed(a.seed + 104729)
    bank = None
    bank_bins = None
    conditioning_edges = None
    if a.noise_kind == "residual":
        with np.load(a.bank) as z:
            b = z["residuals"].astype(np.float32)
            density_log = (z["density_log_residuals"].astype(np.float32)
                           if a.density_noise_space == "log1p" else None)
            bank_context = (z["forecast_density_mean"].astype(np.float32)
                            if a.residual_conditioning == "density4" else None)
        b4 = b.reshape(len(b), F, TOTAL)
        b4 *= (np.asarray(PROC_STD, np.float32) /
               np.maximum(b4.std(axis=(0, 2)), 1e-12))[None, :, None]
        if density_log is not None:
            if density_log.shape != (len(b), TOTAL):
                raise ValueError(f"unexpected density log-residual shape {density_log.shape}")
            density_log *= np.float32(PROC_STD[0] / max(
                density_log.std(dtype=np.float64), 1e-12))
            b4[:, 0] = density_log
        bank = torch.from_numpy(b).to(device)
        if bank_context is not None:
            if bank_context.shape != (len(b),):
                raise ValueError(f"unexpected bank conditioning shape {bank_context.shape}")
            conditioning_edges = np.quantile(bank_context, (.25, .5, .75)).astype(np.float32)
            group = np.searchsorted(conditioning_edges, bank_context, side="right")
            bank_bins = [torch.as_tensor(np.flatnonzero(group == i), device=device,
                                         dtype=torch.long) for i in range(4)]
            if any(len(indices) == 0 for indices in bank_bins):
                raise ValueError("empty residual conditioning bin")

    est = np.zeros((T, F, H, W), np.float32)
    spread = np.zeros_like(est)
    forecast_est = (np.zeros_like(est) if a.forecast_calibration_export else None)
    forecast_spread = (np.zeros_like(est) if a.forecast_calibration_export else None)
    snapshot_frames = (set(np.linspace(
        a.warmup, T - 1, min(a.analysis_snapshot_count, T - a.warmup),
        dtype=np.int64).tolist()) if a.analysis_snapshot_out else set())
    snapshot_members = []
    snapshot_indices = []
    failures = 0
    q_memory = torch.zeros((a.ensemble, STATE_DIM), device=device)
    channel_scales = torch.as_tensor(a.channel_scales, device=device,
                                     dtype=torch.float32).repeat_interleave(TOTAL)
    blind_channel_scales = torch.as_tensor(
        a.blind_channel_scales, device=device, dtype=torch.float32)
    walkable = torch.as_tensor(navigation.build_valid_mask_from_config().reshape(-1),
                               device=device, dtype=torch.bool)
    observation_age = np.zeros_like(omega, dtype=np.uint16)
    for t in range(1, T):
        observation_age[t] = np.where(
            omega[t], 0, np.minimum(observation_age[t - 1] + 1, 65535))
    online_calibrator = (OnlineForecastCalibrator(
        a.forecast_calibrator, device, a.forecast_calibration_strength,
        a.forecast_calibration_min_age, a.forecast_calibration_channels)
                         if a.forecast_calibrator else None)
    calibration_scale_sum = torch.zeros((), device=device)
    calibration_scale_count = 0
    log_negative_sum = torch.zeros((), device=device)
    log_raw_shift_sum = torch.zeros((), device=device)
    t0 = time.time()
    with torch.inference_mode():
        for t in range(T):
            if t and t % 1000 == 0:
                torch.cuda.synchronize()
                elapsed = time.time() - t0
                print(f"[{a.noise_kind}] {t}/{T}, {elapsed/t:.4f} s/frame", flush=True)
            xf0 = filt.forecast()
            if a.noise_kind == "gaussian":
                q = torch.randn(xf0.shape, generator=noise_gen, device=device,
                                dtype=xf0.dtype) * filt.proc
            else:
                if bank_bins is None:
                    ids = torch.randint(len(bank), (a.ensemble,), generator=noise_gen,
                                        device=device)
                else:
                    forecast_context = xf0.reshape(a.ensemble, F, TOTAL)[:, 0]
                    forecast_context = forecast_context[:, walkable].mean(1)
                    edges = torch.as_tensor(conditioning_edges, device=device,
                                            dtype=forecast_context.dtype)
                    groups = torch.bucketize(forecast_context, edges)
                    ids = torch.empty(a.ensemble, device=device, dtype=torch.long)
                    for group_i, candidates in enumerate(bank_bins):
                        member = torch.nonzero(groups == group_i, as_tuple=False).flatten()
                        if len(member):
                            draw = torch.randint(len(candidates), (len(member),),
                                                 generator=noise_gen, device=device)
                            ids[member] = candidates[draw]
                q = bank[ids].clone()
            if a.temporal_rho:
                q_memory.mul_(a.temporal_rho).add_(q, alpha=(1 - a.temporal_rho ** 2) ** 0.5)
                # Scale the perturbation used this step, not the AR state itself.  Mutating
                # q_memory here changes the next step's effective AR coefficient to
                # rho*scale and can make persistent runs explode when scale > 1.
                q = q_memory.clone()
            q.mul_(a.scale).mul_(channel_scales)
            # This is causal: omega[t] is the current observation footprint, and
            # walkable is a fixed map/training-union mask.  No truth or future
            # observations enter the spatial Q scaling.
            use_blind_scale = tuple(a.blind_channel_scales) != (1, 1, 1, 1)
            if use_blind_scale or a.density_state_gain:
                observed = torch.as_tensor(omega[t].reshape(-1), device=device,
                                           dtype=torch.bool)
                blind_walkable = walkable & ~observed
            if use_blind_scale:
                spatial_scale = torch.where(
                    blind_walkable[None], blind_channel_scales[:, None],
                    torch.ones((), device=device, dtype=q.dtype))
                q.reshape(a.ensemble, F, TOTAL).mul_(spatial_scale[None])
            if a.density_state_gain:
                # Forecast-conditional density Q.  The ensemble forecast mean is
                # available before the current observation update, so this remains
                # causal.  Optional max pooling gives cells next to a predicted
                # crowd some arrival-risk spread without consulting the truth.
                density_risk = xf0.reshape(a.ensemble, F, H, W)[:, 0].mean(0)
                if a.density_risk_radius:
                    radius = a.density_risk_radius
                    kernel = 2 * radius + 1
                    density_risk = torch_f.max_pool2d(
                        density_risk[None, None], kernel, stride=1,
                        padding=radius)[0, 0]
                state_scale = 1 + a.density_state_gain * (
                    density_risk / a.density_state_threshold).clamp(0, 1)
                density_scale = torch.where(
                    blind_walkable.reshape(H, W), state_scale,
                    torch.ones((), device=device, dtype=q.dtype))
                q.reshape(a.ensemble, F, H, W)[:, 0].mul_(density_scale[None])
            q.sub_(q.mean(0, keepdim=True))
            if a.density_noise_space == "log1p":
                xf4 = (xf0 + q).reshape(a.ensemble, F, H, W)
                xf04 = xf0.reshape(a.ensemble, F, H, W)
                density = torch.expm1(torch.log1p(xf04[:, 0]) +
                                      q.reshape(a.ensemble, F, H, W)[:, 0])
                raw_shift = (density - xf04[:, 0]).mean(0, keepdim=True)
                log_raw_shift_sum.add_(raw_shift.abs().mean())
                if a.log_density_mean_correction == "ensemble":
                    density = density - raw_shift
                log_negative_sum.add_((density < 0).to(xf0.dtype).mean())
                xf4[:, 0] = density
                xf = clip_bounds(xf4.reshape(a.ensemble, STATE_DIM))
            else:
                xf = clip_bounds(xf0 + q)
            if forecast_est is not None:
                forecast_est[t] = xf.mean(0).reshape(F, H, W).cpu().numpy()
                forecast_spread[t] = xf.std(0, correction=0).reshape(F, H, W).cpu().numpy()
            if t in snapshot_frames:
                # Float16 keeps a 100-member snapshot bank compact.  Training
                # casts back to float32 before covariance/Kalman operations.
                snapshot_members.append(xf.cpu().numpy().astype(np.float16))
                snapshot_indices.append(t)
            if online_calibrator is not None:
                observed_t = torch.as_tensor(omega[t].reshape(-1), device=device,
                                             dtype=torch.bool)
                age_t = torch.as_tensor(observation_age[t].reshape(-1), device=device)
                xf, calibration_scale = online_calibrator(
                    xf, observed_t, age_t, walkable)
                blind_t = walkable & ~observed_t
                calibration_scale_sum.add_(calibration_scale.reshape(F, TOTAL)[:, blind_t].sum())
                calibration_scale_count += F * int(blind_t.sum())
            cells = np.flatnonzero(omega[t].reshape(-1))
            if len(cells):
                # Same row order as build_C: cell-major, then feature.
                obs_idx_np = (np.arange(F)[:, None] * TOTAL + cells[None]).T.reshape(-1)
                obs_idx = torch.as_tensor(obs_idx_np, device=device, dtype=torch.long)
                y_flat = torch.as_tensor(y[t].reshape(-1), device=device)
                try:
                    filt.update(xf, obs_idx, y_flat[obs_idx])
                    filt.rtps(xf, a.rtps)
                except RuntimeError as exc:
                    if "linalg" not in str(exc).lower():
                        raise
                    failures += 1
                    filt.x = xf
            else:
                filt.x = xf
            filt.commit_analysis()
            est[t] = filt.x.mean(0).reshape(F, H, W).cpu().numpy()
            spread[t] = filt.x.std(0, correction=0).reshape(F, H, W).cpu().numpy()
    torch.cuda.synchronize()
    inference_seconds = time.time() - t0
    raw_scores = None
    if a.density_report_gain:
        raw_scores = score_arm(est, spread, x_true, omega, a.warmup)
        spread = apply_density_report_calibration(
            est, spread, omega, a.density_report_gain,
            a.density_report_threshold, a.density_report_radius)
    scores = score_arm(est, spread, x_true, omega, a.warmup)
    diagnostics = detailed_diagnostics(est, spread, x_true, omega, a.warmup)
    calibration_grid = (density_calibration_grid(est, spread, x_true, omega,
                                                  a.warmup, scores)
                        if a.density_calibration_grid else None)
    result = {"config": vars(a), "backend": "torch_cuda", "fix_localization": True,
              "bias_ema": True, "inflation": 1.02, "proc_std": list(PROC_STD),
              "frames_used": T, "n_analysis_failures": failures,
              "seconds": round(time.time() - t0, 2),
              "inference_seconds": inference_seconds,
              "gpu": torch.cuda.get_device_name(0),
              "scores": scores, "diagnostics": diagnostics}
    if online_calibrator is not None:
        result["online_forecast_calibration"] = True
        result["mean_applied_blind_scale"] = float(
            (calibration_scale_sum / max(calibration_scale_count, 1)).item())
    if a.density_noise_space == "log1p":
        result["log_density_diagnostics"] = {
            "mean_abs_raw_physical_shift": float((log_raw_shift_sum / T).item()),
            "negative_fraction_before_clip": float((log_negative_sum / T).item()),
        }
    if calibration_grid is not None:
        result["density_calibration_grid"] = calibration_grid
    if raw_scores is not None:
        result["raw_scores_before_report_calibration"] = raw_scores
        result["report_calibration_only"] = True
    if a.calibration_export:
        # Keep this optional: normal evaluation files remain small.  Observation
        # age is causal and is computed over the complete prefix, rather than
        # being reset at the scoring warmup boundary.
        age = np.zeros_like(omega, dtype=np.uint16)
        for t in range(1, T):
            age[t] = np.where(omega[t], 0, np.minimum(age[t - 1] + 1, 65535))
        export_dir = os.path.dirname(os.path.abspath(a.calibration_export))
        os.makedirs(export_dir, exist_ok=True)
        np.savez_compressed(
            a.calibration_export,
            mean=est[a.warmup:].astype(np.float32),
            spread=spread[a.warmup:].astype(np.float32),
            truth=x_true[a.warmup:].astype(np.float32),
            observed=omega[a.warmup:].astype(bool),
            observation_age=age[a.warmup:],
            walkable=navigation.build_valid_mask_from_config().astype(bool),
            day=np.asarray(a.day),
        )
        result["calibration_export"] = os.path.abspath(a.calibration_export)
    if a.forecast_calibration_export:
        export_dir = os.path.dirname(os.path.abspath(a.forecast_calibration_export))
        os.makedirs(export_dir, exist_ok=True)
        np.savez_compressed(
            a.forecast_calibration_export,
            mean=forecast_est[a.warmup:].astype(np.float32),
            spread=forecast_spread[a.warmup:].astype(np.float32),
            truth=x_true[a.warmup:].astype(np.float32),
            observed=omega[a.warmup:].astype(bool),
            observation_age=observation_age[a.warmup:],
            walkable=navigation.build_valid_mask_from_config().astype(bool),
            day=np.asarray(a.day),
        )
        result["forecast_calibration_export"] = os.path.abspath(
            a.forecast_calibration_export)
    if a.analysis_snapshot_out:
        snapshot_indices = np.asarray(snapshot_indices, dtype=np.int32)
        export_dir = os.path.dirname(os.path.abspath(a.analysis_snapshot_out))
        os.makedirs(export_dir, exist_ok=True)
        np.savez_compressed(
            a.analysis_snapshot_out,
            members=np.stack(snapshot_members),
            truth=x_true[snapshot_indices].astype(np.float32),
            observations=y[snapshot_indices].astype(np.float32),
            observed=omega[snapshot_indices].astype(bool),
            observation_age=observation_age[snapshot_indices],
            obs_std=obs_std.astype(np.float32),
            walkable=navigation.build_valid_mask_from_config().astype(bool),
            frame=snapshot_indices,
            day=np.asarray(a.day),
        )
        result["analysis_snapshot_out"] = os.path.abspath(a.analysis_snapshot_out)
    tag = f"gpu_{a.day}_{a.noise_kind}_s{a.scale:g}_rtps{a.rtps:g}"
    out = a.out or os.path.join(OPT, "experiments", "outputs", f"{tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    al, db = scores["all"], scores["defined_blind"]
    print(f"[all] RMSE={al['rmse']:.5f} CRPS={al['crps']:.5f} "
          f"spread/skill={al['spread_skill']:.3f} cov90={al['coverage'][90]:.3f}")
    print(f"[defined_blind] RMSE={db['rmse']:.5f} CRPS={db['crps']:.5f} "
          f"spread/skill={db['spread_skill']:.3f} cov90={db['coverage'][90]:.3f}")
    for name in CH:
        p = scores["per_channel"][name]
        rho = "None" if p["spearman"] is None else f"{p['spearman']:+.3f}"
        ause = "None" if p["ause"] is None else f"{p['ause']:.3f}"
        print(f"[{name}] spearman={rho} AUSE={ause}")
    print(f"[runtime] {result['seconds']}s on {result['gpu']}; failures={failures}")
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
