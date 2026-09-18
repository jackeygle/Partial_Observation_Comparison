"""PedPred3 surrogate as a deep ensemble — the EnKF's forecast model, retrained by us.

  model.py   the vendored PedPred3, imported from enkf_lab unchanged, plus the read-outs
  data.py    one-step (x_t, x_{t+1}) pairs from grid_cache
  train.py   --arm mean  (original loss) / --arm sigma (Gaussian NLL against a frozen mean net)

Pair s = (mean_s, sigma_s), five seeds. See train.py for the design.
"""
