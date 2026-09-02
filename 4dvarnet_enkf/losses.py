"""
losses.py  —  training losses for the variational reconstruction
================================================================

The two losses from the paper, and one dispatcher so train_varnet.py holds no loss logic:

    supervised    (Eq.14)  ‖x_rec − X‖²                          — uses ground truth
    unsupervised  (Eq.13)  ‖(x_rec−y)⊙Ω‖² + ‖x_rec−Φ(x_rec)‖²    — the variational cost

State channels are [density, vx, vy, var].

Note: several alternative losses were implemented and tested against `supervised`
(teacher-style probabilistic NLL and four variants of it, per-channel variance
normalisation, the teacher's density-weighted square error, a density-weighted MSE, and a
multi-scale/pyramid density term). All of them trained markedly worse — the plain
unweighted MSE of Eq.14 was the only one that optimised well for this unrolled solver.
They have been removed to keep this file to what is actually used; the experiments and the
mechanisms behind the failures are written up in the project memory.
"""
from __future__ import annotations


def supervised_loss(x_rec, X):
    """Paper Eq.14: plain MSE to the ground truth  ‖x_rec − X‖² ."""
    return ((x_rec - X) ** 2).mean()


def unsupervised_loss(x_rec, y, mask, phi):
    """Paper Eq.13: the variational cost itself (no ground truth) —
    observation term ‖(x_rec − y)⊙Ω‖²  +  prior term ‖x_rec − Φ(x_rec)‖² ."""
    return ((x_rec - y) ** 2 * mask).mean() + ((x_rec - phi(x_rec)) ** 2).mean()


def nll_loss(x_rec, X, var, beta=1.0):
    """Gaussian NLL — Lakshminarayanan et al. 2017 (arXiv:1612.01474) Eq.1:

        -log p(y|x) = log(sigma^2)/2 + (y - mu)^2 / (2 sigma^2) + const

    `var` is sigma^2 from the solver's second output layer, already positive, so
    no epsilon is needed here and the term cannot diverge on the ~62% of cells that are empty.

    Why this replaces Eq.14 rather than supplementing it: NLL is a PROPER SCORING RULE (their
    Sec. 2.2), so optimising it rewards a calibrated sigma^2 instead of merely a small error.
    MSE cannot do that — it has no sigma^2 to grade. The cost is accuracy: across their Table 1
    regression sets the RMSE penalty against MC-dropout has a median of 8.0% and a mean of 10.9%
    (recomputed from the table; Naval excluded, its 0.00-vs-0.01 is a rounding artefact). Their
    own explanation, Sec. 3.3: "our method optimizes for NLL (which captures predictive
    uncertainty) instead of MSE". Note the spread is wide -- -10% on Kin8nm to +42% on Yacht --
    so this is not a constant tax to quote as one number.

    Note this is the TRAINING loss only. The variational cost J(x) that the solver descends is
    untouched, so 4DVarNet Sec. 3.4 still holds: the gradient handed to the solver comes from
    the cost, computed from observations alone, never from the ground truth.

    beta -- a per-point weight, NOT part of this paper
    -------------------------------------------------
    Eq.1 above is the whole criterion the paper specifies. beta multiplies each point's NLL by
    its own detached sigma^2, which is a technique from a DIFFERENT paper and is therefore not
    used: beta defaults to 0, at which the factor is 1 and this reduces to Eq.1 exactly. The
    knob is kept only because beta=0 has failed here once before and having the fallback
    measurable is worth four lines of code.

    What that earlier failure was, since it will be asked: at beta=0 sigma^2 ran away toward
    zero and the loss followed it down; killed at epoch 42, vx +203%, full-state RMSE +52%
    (runs/FAILED_ml5_beta0.json). But that run had TWO things this file no longer has -- a
    separate pointwise head reading x_hat, and a per-channel sigma^2 floor of 0.25^2 times the
    field's MAD spread, which the collapse landed on and which turned the loss into an MSE
    weighted by the inverse floors [154, 5, 67, 11905]. Both are gone. sigma^2 now comes from
    the solver's second output layer and the only floor is the paper's own 1e-6 (their Sec.
    2.2.1, footnote 2), so beta=0 is being retried rather than assumed broken.

    Multiplying each point's NLL by a DETACHED sigma^(2*beta) rescales that gradient without
    changing what sigma^2 itself is fitted to. At beta=1 the mean's gradient becomes
    -(X - mu)/2, i.e. MSE's gradient up to a constant factor, identically for every point and
    every channel -- verified numerically: with beta=0 the per-point gradient ratio to MSE
    spanned 17x across a 50x range of sigma^2, and with beta=1 it was 0.50 everywhere. The log
    term still trains sigma^2, so calibration is still learned.

    beta=0 reproduces the paper's Eq.1 exactly and is kept reachable for that reason.
    """
    nll = var.log() / 2 + (X - x_rec) ** 2 / (2 * var)
    if beta:
        nll = var.detach() ** beta * nll
    return nll.mean()


def compute_loss(kind, x_rec, X, y, mask, phi, var=None, nll_beta=1.0):
    """Single entry point used by train_varnet.py — dispatch on --loss."""
    if kind == "supervised":
        return supervised_loss(x_rec, X)
    if kind == "unsupervised":
        return unsupervised_loss(x_rec, y, mask, phi)
    if kind == "nll":
        if var is None:
            raise ValueError("--loss nll needs the solver built with predict_var=True; "
                             "got var=None")
        return nll_loss(x_rec, X, var, beta=nll_beta)
    raise ValueError(f"unknown loss kind: {kind!r}")
