# lcskf

What remains here is used by the final EnKF:

- `dynamics/` — trains the PedPred3 forecast model (5 frames in, 5 out) that the final
  EnKF propagates its ensemble with (`sbatch/submit_dynamics_5to5.sbatch`; the exact
  commands are in the project overview).
- `covariance.py`, `kalman.py` — imported by the package; not used by the final filter.

The learned-covariance sequential Kalman filter (LCSKF) that this directory was built
for — a U-Net-predicted background covariance with a differentiable Kalman analysis — is
**not in the final comparison**. Its training, evaluation, ablations and README are in
the git tag `archive-full-2026-09-24`.
