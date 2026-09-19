"""
variational_solver.py  —  Baseline Component 3: variational cost + iterative solver
====================================================================================

Ties the whole reconstruction together. Given partial observations, it minimises
the 4DVarNet-style variational cost

    J(x) = α_obs² · ‖(x − y) ⊙ Ω‖²        (observation term)
         + α_reg² · ‖x − Φ(x)‖²            (dynamical prior term, Φ = GENN)

over the full state x, starting from the initial estimate X0. Following the paper
(and CIA-Oceanix/4dvarnet-core), the minimisation is a LEARNED gradient descent:

    for k in range(n_iter):                # n_iter = 5..20 (paper)
        g_k   = ∂J/∂x          (automatic differentiation — no hand-derived gradient)
        u_k   = LSTM(g_k)      (a learned update from the gradient — "learn to solve")
        x     = x − u_k

The LSTM update rule and the prior Φ are trained jointly, end-to-end, so the solver
learns how to descend this particular cost. This file owns a clean re-implementation
of that structure (cf. solver.py in 4dvarnet-core), adapted to our state convention.

State / I-O convention
----------------------
    x0, y, mask : (B, C, T, H, W)    C=4 physical channels, T=time window
    mask Ω      : per-(channel,cell,time) observation mask (1=observed)   [our Omega_c]
    solve(...)  -> x_rec (B, C, T, H, W)   the reconstructed full state

The gradient-update LSTM operates on the state viewed as (B, C·T, H, W) — time and
channels folded into the conv-channel axis, 2-D conv-LSTM over space — exactly the
"time-as-channel" trick from 4dvarnet. Φ keeps the (B,C,T,H,W) view internally.

Verification (shapes, end-to-end differentiability, a can-it-learn smoke test) is in
`check_variational_solver.py`.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Observation operator  (= 4dvarnet Model_H; for us literally (x - y) ⊙ Ω)
# --------------------------------------------------------------------------- #
class ObsOperator(nn.Module):
    """Observation misfit on observed cells only: (x − y) ⊙ Ω.

    For the ATC baseline the robots measure the field directly, so H = identity on
    observed cells. (Kept as a module so a non-trivial H can replace it later.)
    """

    def forward(self, x, y, mask):
        return (x - y) * mask


# --------------------------------------------------------------------------- #
# Variational cost  J = α_obs²·‖dy‖² + α_reg²·‖dx‖²   (learnable weights)
# --------------------------------------------------------------------------- #
class VarCost(nn.Module):
    """Weighted variational cost, matching 4dvarnet-core's Model_Var_Cost:

        J = alpha_reg^2 * ||dx||^2_{W_reg}  +  alpha_obs^2 * ||dy||^2_{W_obs}

    where ||v||^2_W is a PER-CHANNEL weighted L2: sum of squares over (B,T,H,W)
    weighted by W_c^2 per physical channel c, normalised by the number of points
    per channel. alpha_obs/alpha_reg (scalar) and W_obs/W_reg (per-channel) are all
    learnable — so the 4 channels (density, vx, vy, var), which live on very
    different scales, can be weighted individually as in the reference.
    """

    def __init__(self, n_channels=4, obs_nll=False):
        super().__init__()
        self.obs_nll = bool(obs_nll)
        self.alpha_obs = nn.Parameter(torch.tensor(1.0))          # scalar obs weight (= alphaObs)
        self.alpha_reg = nn.Parameter(torch.tensor(1.0))          # scalar prior weight (= alphaReg)
        self.w_obs = nn.Parameter(torch.ones(n_channels))         # per-channel obs weight (= WObs)
        self.w_reg = nn.Parameter(torch.ones(n_channels))         # per-channel prior weight (= WReg)

    @staticmethod
    def _weighted_l2(v, w):
        # v: (B, C, T, H, W). Per-channel sum of squares, weighted by w^2, then
        # normalised by points-per-channel (cf. Model_WeightedL2Norm).
        per_channel_sq = (v ** 2).sum(dim=(0, 2, 3, 4))           # (C,)
        n_per_channel = v.numel() / v.shape[1]
        return (per_channel_sq * w ** 2).sum() / n_per_channel

    def forward(self, dx, dy, s_logvar=None, mask=None):
        """s_logvar = log sigma^2, an AUGMENTED STATE channel set, or None for the plain cost.

        When given, the prior term becomes a Gaussian log-likelihood of the prior residual
        instead of a plain squared norm:

            plain       ||x - Phi(x)||^2_{W_reg}
            augmented   mean[ (x - Phi(x))^2 e^{-s} + s ]  (per channel, W_reg-weighted)

        Setting d/ds to zero gives sigma^2 = (x - Phi(x))^2, so descending J now drives the
        variance toward the squared prior residual -- which is the point: sigma^2 acquires a
        gradient from the variational cost itself and can be iterated alongside x, rather than
        being a read-out that only hears from the outer training loss through 20 steps of
        backprop.

        Why the prior residual and not the observation residual: the observation term is
        identically zero on unobserved cells, so a variance placed there would have no gradient
        exactly where it is needed. The prior term is active everywhere. Measured, the
        same-channel prior residual predicts the TRUE error with R^2 0.38 (vx), 0.35 (vy), 0.11
        (density) versus 0.013/0.001/0.028 for distance-to-observation
        (check_outputs/eval/sigma_drivers.json, TRUE_ERROR block) -- so it is a real signal, and
        those figures are also the CEILING of what this route can reach.

        obs_nll=True puts the SAME s into the observation term, on observed cells only:
        mean[ ((x - y)^2 e^{-s} + s) * Omega ]. The +s is masked too, otherwise unobserved cells
        would pay the log penalty twice. With Omega = 0 everywhere this equals the prior-only cost.
        """
        if s_logvar is None:
            obs = self.alpha_obs ** 2 * self._weighted_l2(dy, self.w_obs)
            return obs + self.alpha_reg ** 2 * self._weighted_l2(dx, self.w_reg)
        if self.obs_nll:
            # clamp at sigma^2 = 1e-6: x can match y almost exactly on observed cells, and
            # e^{-s} there would otherwise grow without bound inside the unrolled solve
            s_obs = s_logvar.clamp(min=-13.8)
            obs_pt = (dy ** 2 * torch.exp(-s_obs) + s_obs) * mask
            obs = self.alpha_obs ** 2 * (obs_pt.sum(dim=(0, 2, 3, 4)) * self.w_obs ** 2).sum() \
                / (obs_pt.numel() / obs_pt.shape[1])
        else:
            obs = self.alpha_obs ** 2 * self._weighted_l2(dy, self.w_obs)
        # per-channel, weighted, normalised the same way _weighted_l2 does
        nll = dx ** 2 * torch.exp(-s_logvar) + s_logvar          # (B, C, T, H, W)
        per_channel = nll.sum(dim=(0, 2, 3, 4))
        n_per_channel = nll.numel() / nll.shape[1]
        prior = (per_channel * self.w_reg ** 2).sum() / n_per_channel
        return obs + self.alpha_reg ** 2 * prior


# --------------------------------------------------------------------------- #
# Learned gradient update: a 2-D conv-LSTM over space (cf. ConvLSTM2d in solver.py)
# --------------------------------------------------------------------------- #
class ConvLSTM2d(nn.Module):
    """Standard convolutional LSTM cell on (B, ch, H, W)."""

    def __init__(self, in_ch, hidden_ch, kernel=3):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.gates = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel,
                               padding=kernel // 2)

    def forward(self, x, state):
        B, _, H, W = x.shape
        if state is None:
            h = x.new_zeros(B, self.hidden_ch, H, W)
            c = x.new_zeros(B, self.hidden_ch, H, W)
        else:
            h, c = state
        i, f, o, g = self.gates(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        i, f, o, g = i.sigmoid(), f.sigmoid(), o.sigmoid(), g.tanh()
        c = f * c + i * g
        h = o * c.tanh()
        return h, (h, c)


class GradUpdateLSTM(nn.Module):
    """Map the cost gradient g (B,C,T,H,W) -> an update u (B,C,T,H,W) via a conv-LSTM.

    Time and channels are folded into the conv-channel axis (B, C·T, H, W); a final
    1×1 conv projects the LSTM hidden state back to C·T channels. The initial small
    output scale (k≈0.1) keeps the first solver steps gentle, as in 4dvarnet.

    Variance is not read out here. An earlier design added a second read-out on the
    same hidden state (Lakshminarayanan et al. Sec. 2.2.1); it was measured against
    the augmented-state design, lost, and its checkpoints and code were dropped --
    see the root README's uncertainty table. The surviving design (augmented_var,
    below) iterates log sigma^2 as part of the state instead.

    sigma^2 is NOT added to the iterated state. It has no term in J -- the observation term has
    no sigma^2 component and Phi maps state channels to state channels -- so an augmented state
    would leave it unconstrained by the variational cost. Constraining it properly needs a prior
    over the covariance itself (Beauchamp et al., AIES 2025, use an SPDE prior for exactly this).
    Here it is a read-out taken once, after the last iteration.
    """

    def __init__(self, n_state_ch, hidden_ch=64, dropout=0.0):
        super().__init__()
        self.lstm = ConvLSTM2d(n_state_ch, hidden_ch)
        self.out = nn.Conv2d(hidden_ch, n_state_ch, 1, bias=False)
        nn.init.constant_(self.out.weight, 0.1)            # small initial updates for stability
        self.dropout = nn.Dropout(dropout)

    def forward(self, grad_2d, state):
        h, state = self.lstm(self.dropout(grad_2d), state)
        return self.out(self.dropout(h)), state, h


# --------------------------------------------------------------------------- #
# The solver
# --------------------------------------------------------------------------- #
class GradSolver(nn.Module):
    """Learned-gradient-descent solver for the variational cost (4DVarNet-style).

    Args:
        phi     : the dynamical prior Φ (GENN), maps (B,C,T,H,W)->(B,C,T,H,W)
        n_iter  : number of solver iterations (paper: 5..20)
        n_channels, dT : to size the gradient-update LSTM (C·T input channels)
    """

    def __init__(self, phi, n_channels=4, dT=7, n_iter=15, hidden_ch=64, dropout=0.0,
                 var_eps=1e-6,
                 augmented_var=False, obs_nll=False):
        super().__init__()
        if obs_nll and not augmented_var:
            raise ValueError("obs_nll needs augmented_var: the observation NLL uses the iterated "
                             "log sigma^2, which only the augmented solver has")
        self.phi = phi
        self.obs_op = ObsOperator()
        self.var_cost = VarCost(n_channels, obs_nll=obs_nll)
        # augmented_var doubles the state the optimiser sees: [x, log sigma^2], so the LSTM's
        # channel axis goes from C*dT to 2*C*dT and there is no separate variance read-out.
        self.augmented_var = bool(augmented_var)
        n_in = n_channels * dT * (2 if self.augmented_var else 1)
        self.grad_net = GradUpdateLSTM(n_in, hidden_ch, dropout,
)
        self.n_iter = n_iter
        self.C, self.T = n_channels, dT
        # the minimum variance from Lakshminarayanan et al. Sec. 2.2.1, footnote 2: "pass the
        # second output through the softplus function log(1+exp(.)), and add a minimum variance
        # (e.g. 1e-6) for numerical stability". Their value, not ours. It matters because the
        # NLL is unbounded below as sigma^2 -> 0 and 62% of these cells are empty and can be
        # fitted almost exactly. Not to be confused with the retired 0.25^2 x MAD floor, which
        # WAS ours, was three orders of magnitude larger, and shaped what the head learnt.
        self.var_eps = float(var_eps)

    def _cost_and_grad(self, x, y, mask, s_logvar=None):
        """Variational cost J and its gradient (automatic differentiation — no hand-derived
        gradient of Phi).

        With s_logvar the state is AUGMENTED to [x, log sigma^2] and the gradient is taken with
        respect to both, so the solver descends the variance alongside the field. Both come out
        of one autograd call, which is the whole reason this is cheap to do: nothing about the
        variance's gradient had to be derived by hand either.
        """
        dy = self.obs_op(x, y, mask)
        dx = x - self.phi(x)
        if s_logvar is None:
            J = self.var_cost(dx, dy)
            return J, torch.autograd.grad(J, x, create_graph=self.training)[0], None
        J = self.var_cost(dx, dy, s_logvar, mask)
        gx, gs = torch.autograd.grad(J, (x, s_logvar), create_graph=self.training)
        return J, gx, gs

    def solve(self, x0, y, mask, return_var=False):
        """Iterate n_iter times from x0 and return the reconstruction.

        return_var=True additionally returns the predictive variance, read off the LAST
        hidden state through the second output layer. Default False so that every caller that
        only wants the reconstruction keeps working unchanged.
        """
        B, C, T, H, W = x0.shape
        x = x0.requires_grad_(True)          # x0 is a leaf tensor; enabling grad once suffices
        state = None
        normg = None                         # gradient RMS: computed ONCE (first step) and reused,

        if self.augmented_var:
            # ---- augmented state z = [x, s], s = log sigma^2 -----------------------
            # Both halves are iterated by the SAME learned optimiser on the SAME cost. That is
            # the difference from a read-out: sigma^2 is refined n_iter times against a gradient
            # of J, rather than produced once from a hidden state and supervised only through
            # the outer loss across 20 steps of backprop.
            # s starts at log(softplus(-3)) so the initial sigma matches the read-out design's
            # 0.049 (sigma ~ 0.22, about the model's own RMSE).
            s_lv = torch.full_like(x0, float(np.log(np.log1p(np.exp(-3.0))))).requires_grad_(True)
            for _ in range(self.n_iter):
                J, gx, gs = self._cost_and_grad(x, y, mask, s_lv)
                g = torch.cat([gx, gs], dim=1)                   # (B, 2C, T, H, W)
                if normg is None:
                    normg = torch.sqrt((g ** 2).mean() + 1e-12)
                upd_2d, state, h_last = self.grad_net(
                    (g / normg).reshape(B, 2 * C * T, H, W), state)
                upd = upd_2d.reshape(B, 2 * C, T, H, W) / self.n_iter
                x = x - upd[:, :C]
                s_lv = s_lv - upd[:, C:]
            if not return_var:
                return x
            # clamp only so exp() stays finite; -13.8 is sigma^2 = 1e-6, the paper's own floor
            return x, torch.exp(s_lv.clamp(-13.8, 6.0)) + self.var_eps

        for _ in range(self.n_iter):         #   matching 4dvarnet-core (normgrad_ carried across steps)
            # Subsequent x is non-leaf (x - upd), already in the graph and requires
            # grad automatically — no need to toggle again
            J, grad, _ = self._cost_and_grad(x, y, mask)
            if normg is None:                # reference: normgrad_ = sqrt(mean(grad**2)) at step 0
                normg = torch.sqrt((grad ** 2).mean() + 1e-12)
            grad_2d = (grad / normg).reshape(B, C * T, H, W)
            upd_2d, state, h_last = self.grad_net(grad_2d, state)
            upd = upd_2d.reshape(B, C, T, H, W) / self.n_iter   # reference: grad *= 1/n_grad
            x = x - upd
        if return_var:
            raise ValueError("only augmented_var solvers carry a variance; this one was "
                             "built without it")
        return x

    def forward(self, x0, y, mask, return_var=False):
        return self.solve(x0, y, mask, return_var=return_var)
