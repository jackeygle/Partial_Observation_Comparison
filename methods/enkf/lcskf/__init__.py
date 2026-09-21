"""LCSKF — the learned-covariance sequential Kalman filter.

The EnKF's ensemble spread is contractive (see ../README.md), so this method
replaces the ensemble entirely: a covariance U-Net predicts ``B = U U^T + diag(d)``
for a neural dynamics forecast, and a differentiable Kalman analysis turns the
pair into a posterior mean and an exact posterior diagonal.

  dynamics/    the PedPred3 forecast model and its grid_cache pair loader
  covariance.py  the two-head U-Net that predicts U and d
  kalman.py    Woodbury algebra: analysis_update, posterior_diag, gaussian_nll
  train.py     trains covariance.py (and optionally dynamics/) through kalman.py
  filter.py    runs a trained pair as a sequential filter over a whole day
  checks/      diagnostics, ablations and the calibration/evaluation drivers

There is no covariance ground truth, so nothing here is trained against B
directly: the objective is the analysis error after the Kalman update, which is
why the Kalman layer has to be differentiable.
"""

from .covariance import CovarianceUNet
from .kalman import analysis_update, gaussian_nll, posterior_diag

__all__ = ["CovarianceUNet", "analysis_update", "gaussian_nll", "posterior_diag"]
