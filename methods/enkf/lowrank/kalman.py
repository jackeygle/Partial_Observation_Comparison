"""Exact differentiable Kalman algebra for ``B = U U^T + diag(d)``.

The observation operator in this project selects state entries.  Exploiting that
structure and Woodbury's identity reduces every solve from the number of observed
state entries (usually about 1,000) to the learned rank (8--32).  No dense
``(state_dim, state_dim)`` covariance is formed.
"""
from __future__ import annotations

import math

import torch


def _flat_inputs(mean, factor, diagonal, observation, mask, obs_var):
    """Validate and flatten grids to mean/y/d/mask ``(B,n)`` and U ``(B,n,r)``."""
    if mean.ndim != 4:
        raise ValueError(f"mean must be (B,C,H,W), got {tuple(mean.shape)}")
    if factor.ndim != 5:
        raise ValueError(f"factor must be (B,R,C,H,W), got {tuple(factor.shape)}")
    if factor.shape[0] != mean.shape[0] or factor.shape[2:] != mean.shape[1:]:
        raise ValueError("factor and mean shapes do not agree")
    if diagonal.shape != mean.shape or observation.shape != mean.shape:
        raise ValueError("diagonal, observation and mean must have the same shape")
    if mask.ndim == 3:
        mask = mask[:, None].expand_as(mean)
    if mask.shape != mean.shape:
        raise ValueError("mask must be (B,H,W) or (B,C,H,W)")

    b, c, h, w = mean.shape
    rank = factor.shape[1]
    u = factor.permute(0, 2, 3, 4, 1).reshape(b, c * h * w, rank)
    d = diagonal.reshape(b, -1)
    mu = mean.reshape(b, -1)
    y = observation.reshape(b, -1)
    observed = mask.reshape(b, -1).to(mean.dtype)
    r = torch.as_tensor(obs_var, dtype=mean.dtype, device=mean.device)
    if r.ndim == 1:
        if r.numel() != c:
            raise ValueError(f"obs_var has {r.numel()} channels, expected {c}")
        r = r.view(1, c, 1, 1).expand(b, c, h, w)
    elif r.shape == (b, c):
        r = r[:, :, None, None].expand(b, c, h, w)
    elif r.shape != mean.shape:
        raise ValueError("obs_var must be (C,), (B,C), or (B,C,H,W)")
    return mu, u, d, y, observed, r.reshape(b, -1)


def _latent_system(u, weight, innovation, jitter):
    """Return Cholesky and posterior latent mean for the Woodbury system."""
    rank = u.shape[-1]
    eye = torch.eye(rank, dtype=u.dtype, device=u.device).expand(u.shape[0], rank, rank)
    system = eye + torch.einsum("bnr,bn,bns->brs", u, weight, u)
    chol = torch.linalg.cholesky(system + jitter * eye)
    rhs = torch.einsum("bnr,bn->br", u, weight * innovation)
    latent = torch.cholesky_solve(rhs.unsqueeze(-1), chol).squeeze(-1)
    return chol, latent


def analysis_update(mean, factor, diagonal, observation, mask, obs_var, jitter=1e-6):
    """Return the posterior mean for direct, independently noisy observations.

    All operations are differentiable. ``diagonal`` and ``obs_var`` are variances,
    not standard deviations, and must be strictly positive at observed entries.
    """
    mu, u, d, y, observed, r = _flat_inputs(
        mean, factor, diagonal, observation, mask, obs_var)
    if torch.any(d <= 0) or torch.any(r <= 0):
        raise ValueError("diagonal and observation variances must be positive")
    innovation = observed * (y - mu)
    weight = observed / (d + r)
    _, latent = _latent_system(u, weight, innovation, jitter)
    latent_fit = torch.einsum("bnr,br->bn", u, latent)
    residual_precision = weight * (innovation - latent_fit)
    increment = latent_fit + d * residual_precision
    return (mu + increment).reshape_as(mean)


def posterior_diag(mean, factor, diagonal, observation, mask, obs_var, jitter=1e-6):
    """Return the exact diagonal of the posterior covariance in grid shape."""
    mu, u, d, y, observed, r = _flat_inputs(
        mean, factor, diagonal, observation, mask, obs_var)
    innovation = observed * (y - mu)
    weight = observed / (d + r)
    chol, _ = _latent_system(u, weight, innovation, jitter)
    # diag(U C U^T), C = latent posterior covariance = system^-1.
    uc = torch.cholesky_solve(u.transpose(1, 2), chol).transpose(1, 2)
    latent_var = (u * uc).sum(-1)
    attenuation = torch.where(observed.bool(), r / (d + r), torch.ones_like(d))
    independent_var = torch.where(observed.bool(), d * r / (d + r), d)
    return (independent_var + attenuation.square() * latent_var).reshape_as(mean)


def gaussian_nll(target, mean, factor, diagonal, jitter=1e-6):
    """Mean per-state Gaussian NLL under ``N(mean, U U^T + diag(d))``."""
    if target.shape != mean.shape or diagonal.shape != mean.shape:
        raise ValueError("target, mean and diagonal must have the same shape")
    b = mean.shape[0]
    output_dtype = mean.dtype
    # I + U^T D^-1 U is positive definite analytically, but becomes sufficiently
    # ill-conditioned in trained rank-32 models for float32 Cholesky to report a
    # false non-PD failure.  Only this small latent system and its reductions need
    # float64; network activations and Kalman analysis remain float32.
    work_dtype = torch.float64 if mean.dtype in (torch.float16, torch.bfloat16,
                                                  torch.float32) else mean.dtype
    residual = (target - mean).reshape(b, -1).to(work_dtype)
    d = diagonal.reshape(b, -1).to(work_dtype)
    if torch.any(d <= 0):
        raise ValueError("diagonal variances must be positive")
    u = factor.permute(0, 2, 3, 4, 1).reshape(
        b, residual.shape[1], factor.shape[1]).to(work_dtype)
    precision = d.reciprocal()
    chol, latent = _latent_system(u, precision, residual, jitter)
    mahal = (residual.square() * precision).sum(-1)
    rhs = torch.einsum("bnr,bn->br", u, precision * residual)
    mahal = mahal - (rhs * latent).sum(-1)
    logdet = d.log().sum(-1) + 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
    n = residual.shape[1]
    nll = (0.5 * (mahal + logdet + n * math.log(2.0 * math.pi)) / n).mean()
    return nll.to(output_dtype)


def masked_gaussian_nll(target, mean, factor, diagonal, valid, jitter=1e-6):
    """Gaussian NLL of each sample's dynamically selected state subvector.

    ``valid`` has the same shape as ``mean``. Entries outside it are marginalized
    out exactly by assigning zero precision in the Woodbury system. Each sample
    is normalized by its own number of defined entries before the batch mean.
    """
    if target.shape != mean.shape or diagonal.shape != mean.shape or valid.shape != mean.shape:
        raise ValueError("target, mean, diagonal, and valid must have the same shape")
    b = mean.shape[0]
    output_dtype = mean.dtype
    work_dtype = torch.float64 if mean.dtype in (torch.float16, torch.bfloat16,
                                                  torch.float32) else mean.dtype
    residual = (target - mean).reshape(b, -1).to(work_dtype)
    d = diagonal.reshape(b, -1).to(work_dtype)
    selected = valid.reshape(b, -1).to(work_dtype)
    if torch.any(d <= 0):
        raise ValueError("diagonal variances must be positive")
    counts = selected.sum(-1)
    if torch.any(counts <= 0):
        raise ValueError("every sample must contain at least one valid state entry")
    u = factor.permute(0, 2, 3, 4, 1).reshape(
        b, residual.shape[1], factor.shape[1]).to(work_dtype)
    precision = selected / d
    chol, latent = _latent_system(u, precision, residual, jitter)
    mahal = (residual.square() * precision).sum(-1)
    rhs = torch.einsum("bnr,bn->br", u, precision * residual)
    mahal = (mahal - (rhs * latent).sum(-1)).clamp_min(0)
    logdet = (selected * d.log()).sum(-1) + 2.0 * torch.log(
        torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
    nll = 0.5 * (mahal + logdet + counts * math.log(2.0 * math.pi)) / counts
    return nll.mean().to(output_dtype)


def student_t_nll(target, mean, factor, diagonal, degrees_of_freedom=3.0, jitter=1e-6):
    """Mean per-state multivariate Student-t NLL with low-rank scale matrix.

    The scale matrix is ``U U^T + diag(d)``.  ``degrees_of_freedom`` controls
    tail weight and must be positive; values around 3--5 tolerate occasional
    large forecast errors much better than a Gaussian.
    """
    if target.shape != mean.shape or diagonal.shape != mean.shape:
        raise ValueError("target, mean and diagonal must have the same shape")
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    b = mean.shape[0]
    output_dtype = mean.dtype
    work_dtype = torch.float64 if mean.dtype in (torch.float16, torch.bfloat16,
                                                  torch.float32) else mean.dtype
    residual = (target - mean).reshape(b, -1).to(work_dtype)
    d = diagonal.reshape(b, -1).to(work_dtype)
    if torch.any(d <= 0):
        raise ValueError("diagonal variances must be positive")
    u = factor.permute(0, 2, 3, 4, 1).reshape(
        b, residual.shape[1], factor.shape[1]).to(work_dtype)
    precision = d.reciprocal()
    chol, latent = _latent_system(u, precision, residual, jitter)
    mahal = (residual.square() * precision).sum(-1)
    rhs = torch.einsum("bnr,bn->br", u, precision * residual)
    mahal = (mahal - (rhs * latent).sum(-1)).clamp_min(0)
    logdet = d.log().sum(-1) + 2.0 * torch.log(
        torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
    n = residual.shape[1]
    nu = torch.as_tensor(degrees_of_freedom, dtype=work_dtype, device=mean.device)
    nll = (torch.lgamma(0.5 * nu) - torch.lgamma(0.5 * (nu + n))
           + 0.5 * (n * torch.log(nu * math.pi) + logdet)
           + 0.5 * (nu + n) * torch.log1p(mahal / nu))
    return (nll / n).mean().to(output_dtype)
