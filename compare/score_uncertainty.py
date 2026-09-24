"""
score_uncertainty.py — scoring for predictive uncertainty, the single
implementation shared by all four methods
================================================================

Why this exists: `crps_gaussian` used to have two verbatim copies
(`methods/varnet/checks/eval_uncertainty.py` and
`methods/enkf/checks/eval_uncertainty_enkf.py`), the latter with a comment saying
"same as the 4DVarNet side so the two numbers are directly comparable" -- a promise
that only a human can keep in sync. Adding DINCAE to table B would have made a
third copy. So it was pulled out here; all three import the same function.

Four metrics, each answering a different question (missing any one means
misreading the result):

  RMSE            is mu accurate? Independent of sigma.
  CRPS            the **verdict metric**. Scored pointwise, penalising both "wrong
                  centre" and "wrong width" at once, minimised at the true sigma
                  (a proper scoring rule). Same units as the data; reduces to MAE
                  as sigma -> 0.
  spread/skill    mean(sigma) / rmse. Looks only at the **average scale**, entirely
                  blind to which cells sigma actually lands on -- a constant-sigma
                  null model always scores a perfect 1.00 on this.
  coverage        how much truth the nominal z% interval actually contains. The
                  most intuitive one, and what a downstream consumer (a robot's
                  planner) actually needs.

Why NLL is not the verdict metric: it has an (x-mu)^2/(2 sigma^2) term, unbounded
as sigma -> 0. The EnKF's ensemble collapses to sigma=0.0025 while actually being
wrong by 0.2388, blowing its NLL up to 1.67e18 -- that cannot rank methods against
each other. It is still computed and written to the json, just not used in the
main table.

The constant-sigma baseline (`const_sigma_baseline`)
-----------------------------------------------------
Replace every cell's sigma-hat with a single number, that slice's own RMSE. It is
a **null model**: it knows how big its average error is and nothing about
**where**. The only way to judge whether a sigma-hat carries real information is
to check whether its CRPS beats this baseline -- because the null model's
spread/skill is always exactly 1.00, so that "perfect calibration" score is free.

Everything in physical units
----------------------------
All methods are scored in physical units (`supervisor_evaluation/evaluate.py
uncertainty`). 4DVarNet and EnKF predict Gaussians there directly. DINCAE predicts
a Gaussian in its standardised space; density/vx/vy are only standardised
(linear), so they map back to physical Gaussians with sigma * std, while `var`
also goes through log1p, so its physical predictive is a shifted log-normal --
scored with `Accumulator.add_lognormal1p` (closed-form CRPS, no sampling).

An earlier version scored DINCAE in its standardised space and argued that the
ratios cancel the transform. That holds for coverage (a monotonic transform keeps
"inside the interval"), but not for CRPS skill or spread/skill: the log1p space
compresses the large errors, and a constant sigma there is already
heteroscedastic in physical units -- a different null model.

Usage
-----
    from compare import score_uncertainty as su

    # one-shot (data fits in memory entirely; 4DVarNet takes this path)
    r = su.score(mu, sigma, x)                    # -> dict
    b = su.score(mu, su.const_sigma(mu, x), x)    # the null-model baseline

    # streaming (4.8e8 points do not fit in memory; EnKF/DINCAE take this path)
    a = su.Accumulator()
    for chunk in ...:  a.add(mu_c, sigma_c, x_c)
    r = a.result()
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

#: Which nominal confidence levels to report. Only looking at all nine together
#: gives a reliability curve -- looking only at 90% would miss a wrong
#: distribution shape (measured: the constant-sigma baseline is a tolerable 95.4%
#: at 90%, but only 21.3% at 10%, because the real error distribution is
#: heavy-tailed).
ZS = tuple(range(10, 100, 10))

_SQRT_PI = np.sqrt(np.pi)


def _f64(a):
    """Coerce to a float64 numpy array. Accepts a torch tensor, numpy array, or scalar.

    Accumulating 4.8e8 numbers in float32 loses significant digits; the cost here
    is just a one-time conversion.
    """
    if hasattr(a, "detach"):                      # torch tensor
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=np.float64)


def crps_gaussian(mu, sigma, x):
    """Closed-form CRPS of N(mu, sigma^2) against the observed x (Gneiting & Raftery 2007).

        CRPS = sigma * [ z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ],  z = (x - mu)/sigma

    Lower is better, and it is in the units of x, so it can be read next to the RMSE.
    """
    mu, sigma, x = _f64(mu), _f64(sigma), _f64(x)
    z = (x - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / _SQRT_PI)


def nll_gaussian(mu, sigma, x):
    """Gaussian negative log-likelihood. Unbounded as sigma -> 0 -- see the module
    docstring for why this is not the verdict metric."""
    mu, sigma, x = _f64(mu), _f64(sigma), _f64(x)
    return 0.5 * np.log(2 * np.pi * sigma ** 2) + (x - mu) ** 2 / (2 * sigma ** 2)


#: exp() argument cap for the log-normal helpers: exp(700) is still finite in float64.
_EXP_CAP = 700.0


def crps_lognormal1p(m, s, v):
    """Closed-form CRPS when log1p(V) ~ N(m, s^2), scored against the observed v >= 0.

    V + 1 is log-normal, and CRPS is shift-invariant, so this is the log-normal
    CRPS of Baran & Lerch (2015) evaluated at y = v + 1:

        CRPS = y (2 Phi(z) - 1) - 2 exp(m + s^2/2) [Phi(z - s) + Phi(s/sqrt 2) - 1],
        z = (ln y - m) / s

    In the units of v, so it sits next to the Gaussian CRPS of the other methods.
    """
    m, s, v = _f64(m), _f64(s), _f64(v)
    y = np.maximum(v, 0.0) + 1.0
    z = (np.log(y) - m) / s
    e = np.exp(np.minimum(m + 0.5 * s ** 2, _EXP_CAP))
    return y * (2 * norm.cdf(z) - 1) - 2 * e * (norm.cdf(z - s) + norm.cdf(s / np.sqrt(2)) - 1)


def const_sigma(mu, x):
    """The null model's sigma: this slice's own RMSE (a scalar).

    It must use **this slice's** RMSE. Using the full-field RMSE as the baseline
    for a blind-cell slice would put a metric and its own baseline on two
    different cell sets -- exactly the mistake that's easiest to make when the two
    tables' conventions diverge.
    """
    mu, x = _f64(mu), _f64(x)
    return float(np.sqrt(((x - mu) ** 2).mean()))


class Accumulator:
    """Streaming accumulation: a single channel over the full test set is already
    ~6e7 points, four channels over the full field ~4.8e8 -- the residuals cannot
    be kept around.

    Only scalar sums are kept, so memory is O(1). `add` accepts any shape; it is
    flattened internally.
    """

    def __init__(self):
        self.crps = self.nll = self.se = self.sig = 0.0
        self.n = 0
        self.cov = {z: 0 for z in ZS}

    def _add(self, crps, nll, err, sigma, r):
        """Pointwise CRPS, NLL, point error, predictive SD and |standardised residual|."""
        self.crps += crps.sum()
        self.nll += nll.sum()
        self.se += (err ** 2).sum()
        self.sig += sigma.sum()
        self.n += err.size
        for z in ZS:
            self.cov[z] += int((r <= norm.ppf(0.5 + z / 200)).sum())

    def add(self, mu, sigma, x):
        """Gaussian predictive N(mu, sigma^2)."""
        mu, sigma, x = _f64(mu).ravel(), _f64(sigma).ravel(), _f64(x).ravel()
        if x.size == 0:
            return
        self._add(crps_gaussian(mu, sigma, x), nll_gaussian(mu, sigma, x),
                  x - mu, sigma, np.abs(x - mu) / sigma)

    def add_lognormal1p(self, m, s, v):
        """Predictive log1p(V) ~ N(m, s^2), everything scored in the units of v.

        Point estimate = the median expm1(m) (the same point the accuracy table
        uses for such a channel); sigma = the log-normal predictive SD. The
        central z% interval is expm1(m +/- q s), so coverage is decided in log1p
        space -- identical to transforming the interval back.
        """
        m, s, v = _f64(m).ravel(), _f64(s).ravel(), _f64(v).ravel()
        if v.size == 0:
            return
        y = np.log1p(np.maximum(v, 0.0))
        point = np.expm1(np.minimum(m, _EXP_CAP))
        sd = (np.sqrt(np.expm1(np.minimum(s ** 2, _EXP_CAP)))
              * np.exp(np.minimum(m + 0.5 * s ** 2, _EXP_CAP)))
        # density of V at v = N(y; m, s) / (1 + v)  ->  NLL gains + log1p(v)
        self._add(crps_lognormal1p(m, s, v), nll_gaussian(m, s, y) + y,
                  v - point, sd, np.abs(y - m) / s)

    def result(self):
        if not self.n:
            return None
        rmse = float(np.sqrt(self.se / self.n))
        sig = float(self.sig / self.n)
        return {"rmse": rmse,
                "crps": float(self.crps / self.n),
                "nll": float(self.nll / self.n),
                "sigma_mean": sig,
                "spread_skill": sig / rmse if rmse > 0 else float("nan"),
                "coverage": {z: self.cov[z] / self.n for z in ZS},
                "n": int(self.n)}


def score(mu, sigma, x):
    """One-shot scoring (when the data fits in memory entirely). Returns a dict
    with the same shape as `Accumulator.result()`."""
    a = Accumulator()
    a.add(mu, sigma, x)
    return a.result()


def score_with_baseline(mu, sigma, x):
    """Score, and attach this slice's own constant-sigma null model.

    Returns (metrics, baseline_metrics). Verdict = metrics['crps'] / baseline['crps'];
    < 1 means the learned sigma-hat genuinely carries spatial information.
    """
    m = score(mu, sigma, x)
    c = const_sigma(mu, x)
    b = score(mu, np.full_like(_f64(x), c), x)
    return m, b
