"""DINCAE: convolutional autoencoder inpainting, outputting a per-channel mean and
sigma^2 (information form).

  state / encoding    channel definitions, transforms, per-cell statistics, input encoding
  dataset / model     data and network
  losses / train      loss and training entry point

Note it only trains on cells where **that channel is defined** (empty cells have no
velocity) — this decides which scoring convention it must use, see the header of
the top-level compare/compare4.py.
"""
