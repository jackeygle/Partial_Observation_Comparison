"""
model.py — the vendored PedPred3, imported from enkf_lab unchanged, plus the read-outs the
surrogate needs

The architecture is not copied: it is imported from enkf_lab (read-only), so a mean-arm
network is by construction the same model the EnKF already runs, and its checkpoint loads with
enkf_lab's own utils.load_model().

enkf_lab is not a subpackage (it holds a `pedpred -> .` self-symlink and is loaded by path, see
methods/enkf/__init__.py), hence the sys.path insert.
"""
from __future__ import annotations

import sys

import torch

from crowdcore import paths

#: State channel order, as in grid_cache and the EnKF state.
CH = ("density", "vx", "vy", "var")


def _pedpred():
    root = paths.enkf_vendor("enkf_lab")
    if root not in sys.path:
        sys.path.insert(0, root)
    from pedpred import metrics, models
    return models, metrics


def load_pedpred3(ckpt: str | None = None, device="cpu") -> torch.nn.Module:
    """A fresh PedPred3 (initialised from the current torch seed), or one loaded from a
    checkpoint holding {'model': state_dict} -- the format of apt-ibex_train_model_28D.pth."""
    models, _ = _pedpred()
    net = models.PedPred3()
    if ckpt:
        net.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    return net.to(device)


def mean_forecast(net, x, horizon: int = 1):
    """x (B,Tin,4,H,W) -> mu (B,horizon,4,H,W): PedPred3's output interpretation, done on
    x's device -- density = exp(ch0), velocity = ch1:3, variance = exp(ch3).

    Not net(x) itself: PedPred3.forward builds its output with GridData(logdensity=...,
    vel_mean=..., vel_logvar=...), whose constructor allocates with torch.full on the DEFAULT
    device, so on a GPU the forecast silently lands on the CPU. The EnKF only ever runs on the
    CPU, which is why it never showed. check_vendor_forward() asserts the two are bit-identical."""
    raw = raw_forecast(net, x, horizon=horizon)
    return torch.cat([raw[:, :, 0:1].exp(), raw[:, :, 1:3], raw[:, :, 3:4].exp()], dim=2)


def check_vendor_forward(net, x, horizon: int = 1):
    """Exit unless mean_forecast() equals PedPred3.forward exactly, on a CPU copy of `net`."""
    import copy
    cpu = copy.deepcopy(net).cpu().eval()
    xc = x[:8].detach().cpu()
    with torch.no_grad():
        ours = mean_forecast(cpu, xc, horizon=horizon)
        vendor = cpu(xc, horizon=horizon).as_subclass(torch.Tensor)
    if not torch.equal(ours, vendor):
        raise SystemExit(f"mean_forecast differs from PedPred3.forward: "
                         f"max |diff| {float((ours - vendor).abs().max()):.3g}")


def raw_forecast(net, x, horizon: int = 1):
    """The same network's four output channels BEFORE PedPred3.forward interprets them (no exp).
    The sigma arm reads them as log sigma^2 of the four state channels."""
    models, _ = _pedpred()
    out = super(models.PedPred3, net).forward(x.transpose(0, 1), None, horizon=horizon)
    return out.transpose(0, 1)


#: LocalizedEnsembleKalmanFilter._clip_bounds: density [0,5], vx/vy [-5,5], variance [0,2].
CLIP_LO = (0.0, -5.0, -5.0, 0.0)
CLIP_HI = (5.0, 5.0, 5.0, 2.0)
#: Where the forecast density is below this, velocity and variance are set to 0. The grid defines
#: them as 0 on empty cells, but the original loss weights them by the true density and never
#: trains them there, so the raw forecast drifts (valid split, walkable cells: var RMSE 0.51-0.73
#: raw, 0.11 zeroed). Chosen on the validation split: 0.01 and 0.02 tie, 0.01 zeroes fewer
#: occupied cells.
EMPTY_DENSITY = 0.01


def surrogate_mean(net, x, horizon: int = 1):
    """The mean the EnKF propagates: mean_forecast() with vx/vy/var zeroed where the forecast
    density < EMPTY_DENSITY, clipped to the EnKF's bounds."""
    mu = mean_forecast(net, x, horizon=horizon)
    empty = mu[:, :, 0:1] < EMPTY_DENSITY
    mu = torch.cat([mu[:, :, 0:1], torch.where(empty, torch.zeros_like(mu[:, :, 1:]), mu[:, :, 1:])], dim=2)
    lo = torch.tensor(CLIP_LO, device=mu.device, dtype=mu.dtype).view(1, 1, 4, 1, 1)
    hi = torch.tensor(CLIP_HI, device=mu.device, dtype=mu.dtype).view(1, 1, 4, 1, 1)
    return torch.maximum(torch.minimum(mu, hi), lo)


def original_loss(pred, target):
    """The loss the vendored model was trained with (Partial_observation/config.py default)."""
    _, metrics = _pedpred()
    return metrics.Metrics(pred, target)["mean total weighted NLLL"]
