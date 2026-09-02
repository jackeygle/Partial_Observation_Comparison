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

    def __init__(self, n_channels=4):
        super().__init__()
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

    def forward(self, dx, dy):
        return (self.alpha_obs ** 2 * self._weighted_l2(dy, self.w_obs)
                + self.alpha_reg ** 2 * self._weighted_l2(dx, self.w_reg))


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

    With predict_var=True a SECOND read-out is added on the same hidden state, giving the
    predictive variance. This is Lakshminarayanan et al. Sec. 2.2.1 as written -- "a network
    that outputs two values in the final layer, corresponding to the predicted mean and
    variance" -- rather than a separate module bolted on after the solve, which is what this
    file used to do.

    The variance read-out sees h AND the current state x_hat (var_sees_state, default on).
    An earlier version of this class read h alone, on the argument that h has seen all n_iter
    gradients of J and carries spatial context while x_hat is only one cell. That argument was
    right about h and wrong to exclude x_hat: measured, sigma^2 came out spatially FLAT, varying
    0.96-1.21x between empty and occupied blind cells where the true error varies 19-30x
    (checks/diag_beta_gradient_share.py). A 24-bin 1-D lookup on |x_hat|, fitted on one day and
    applied unchanged to another, beat it on every channel -- NLL -2.31 vs +9.32 on density,
    -0.90 vs +38.14 on vx (checks/diag_sigma_headroom.py). The information was available and
    the read-out could not reach it, because x_hat is a 20-step accumulator (x = x - upd) and
    is not a function of h. Feeding both keeps h's context and adds what it was missing.

    x_hat enters DETACHED. In Lakshminarayanan et al. Sec. 2.2.1 mu and sigma^2 are siblings off
    one shared representation; neither is an input to the other, and detaching keeps that -- the
    variance term cannot lower its own loss by moving the very x_hat it conditions on.

    It does NOT isolate the mean, and an earlier version of this docstring wrongly claimed it
    did. Measured by backpropagating the variance branch alone with out_var.weight off zero:
    grad_net.out.weight 7.2e1, lstm.gates.weight 5.7e3, phi 2.1e1 -- the variance term reshapes
    the trunk, the mean read-out and the prior. It has to: h_last depends on the earlier x
    values, and those are accumulated through out(), so only the final x_hat -> out_var edge is
    cut. That is also what Sec. 2.2.1 implies, where the NLL's gradient reaches the shared
    representation through both outputs. (An init-time test reads zero everywhere here, because
    out_var.weight starts at zero and kills d(sigma^2)/d(h); do not test at initialisation.)

    sigma^2 is NOT added to the iterated state. It has no term in J -- the observation term has
    no sigma^2 component and Phi maps state channels to state channels -- so an augmented state
    would leave it unconstrained by the variational cost. Constraining it properly needs a prior
    over the covariance itself (Beauchamp et al., AIES 2025, use an SPDE prior for exactly this).
    Here it is a read-out taken once, after the last iteration.
    """

    def __init__(self, n_state_ch, hidden_ch=64, dropout=0.0, predict_var=False,
                 var_sees_state=True):
        super().__init__()
        self.lstm = ConvLSTM2d(n_state_ch, hidden_ch)
        self.out = nn.Conv2d(hidden_ch, n_state_ch, 1, bias=False)
        nn.init.constant_(self.out.weight, 0.1)            # small initial updates for stability
        self.dropout = nn.Dropout(dropout)
        self.out_var = None
        if predict_var:
            # bias -3 so softplus(-3) = 0.049, i.e. sigma ~ 0.22 at initialisation, which is
            # about the model's own RMSE. Starting a little over-dispersed is the safe side:
            # the NLL's 1/(2 sigma^2) term is benign when sigma^2 is too big and explosive when
            # it is too small.
            # hidden_ch + n_state_ch = 864 with the standard settings, which is exactly the
            # input width of the LSTM's own gates conv -- the same concat this architecture
            # already uses everywhere, not a new pattern.
            var_in = hidden_ch + (n_state_ch if var_sees_state else 0)
            self.out_var = nn.Conv2d(var_in, n_state_ch, 1, bias=True)
            nn.init.zeros_(self.out_var.weight)
            nn.init.constant_(self.out_var.bias, -3.0)
        self.var_sees_state = bool(predict_var and var_sees_state)

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
                 predict_var=False, var_eps=1e-6, var_sees_state=True):
        super().__init__()
        self.phi = phi
        self.obs_op = ObsOperator()
        self.var_cost = VarCost(n_channels)
        self.grad_net = GradUpdateLSTM(n_channels * dT, hidden_ch, dropout,
                                       predict_var=predict_var,
                                       var_sees_state=var_sees_state)
        self.n_iter = n_iter
        self.C, self.T = n_channels, dT
        # the minimum variance from Lakshminarayanan et al. Sec. 2.2.1, footnote 2: "pass the
        # second output through the softplus function log(1+exp(.)), and add a minimum variance
        # (e.g. 1e-6) for numerical stability". Their value, not ours. It matters because the
        # NLL is unbounded below as sigma^2 -> 0 and 62% of these cells are empty and can be
        # fitted almost exactly. Not to be confused with the retired 0.25^2 x MAD floor, which
        # WAS ours, was three orders of magnitude larger, and shaped what the head learnt.
        self.var_eps = float(var_eps)

    def _cost_and_grad(self, x, y, mask):
        """Variational cost J and its gradient w.r.t. x (automatic differentiation —
        no hand-derived gradient of Φ)."""
        dy = self.obs_op(x, y, mask)
        dx = x - self.phi(x)
        J = self.var_cost(dx, dy)
        grad = torch.autograd.grad(J, x, create_graph=self.training)[0]
        return J, grad

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
        for _ in range(self.n_iter):         #   matching 4dvarnet-core (normgrad_ carried across steps)
            # Subsequent x is non-leaf (x - upd), already in the graph and requires
            # grad automatically — no need to toggle again
            J, grad = self._cost_and_grad(x, y, mask)
            if normg is None:                # reference: normgrad_ = sqrt(mean(grad**2)) at step 0
                normg = torch.sqrt((grad ** 2).mean() + 1e-12)
            grad_2d = (grad / normg).reshape(B, C * T, H, W)
            upd_2d, state, h_last = self.grad_net(grad_2d, state)
            upd = upd_2d.reshape(B, C, T, H, W) / self.n_iter   # reference: grad *= 1/n_grad
            x = x - upd
        if not return_var:
            return x
        if self.grad_net.out_var is None:
            raise ValueError("this solver was built with predict_var=False; there is no "
                             "variance read-out to return")
        hv = h_last
        if self.grad_net.var_sees_state:
            # x_hat detached: mu and sigma^2 are siblings off a shared representation in
            # Lakshminarayanan et al. Sec. 2.2.1, never one feeding the other.
            hv = torch.cat([h_last, x.reshape(B, C * T, H, W).detach()], dim=1)
        var = F.softplus(self.grad_net.out_var(hv)).reshape(B, C, T, H, W)
        return x, var + self.var_eps

    def forward(self, x0, y, mask, return_var=False):
        return self.solve(x0, y, mask, return_var=return_var)
