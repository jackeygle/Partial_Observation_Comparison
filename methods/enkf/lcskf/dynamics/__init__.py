"""The forecast model LCSKF propagates the state with.

  model.py   the vendored PedPred3, imported from enkf_lab unchanged, plus the
             read-outs (density = exp(ch0), velocity, variance = exp(ch3)) and
             the EnKF's clipping/empty-cell projection
  data.py    causal (x_{t-k:t}, x_{t+1:t+n}) pairs from grid_cache
  train.py   pretrains PedPred3 with the vendored "mean total weighted NLLL"

PedPred3 is a ConvLSTM encoder-forecaster, so it consumes a history of any
length; ``--input-frames`` and ``--output-frames`` choose the training regime
and the filter then calls it at ``horizon=1``.
"""
