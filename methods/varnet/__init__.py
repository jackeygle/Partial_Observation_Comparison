"""4DVarNet: minimise the variational cost J with a learned optimiser.

  prior_model         the GENN prior Phi
  variational_solver  GradSolver — unrolled gradient descent + ConvLSTM
  losses              Eq.14's plain MSE and the Gaussian NLL
  train               training entry point
  checks/             this method's own diagnostics (cross-method ones live in top-level compare/)
"""
