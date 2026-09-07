"""methods — each subpackage is one method, independent of the others.

  varnet      4DVarNet variational assimilation (plain MSE arm and NLL/uncertainty-head arm)
  enkf        localised EnKF — a vendor copy of Partial_observation, see enkf/README
  dincae      DINCAE convolutional autoencoder inpainting
  senseiver   Senseiver sparse-sensor reconstruction

Methods **must not import each other**. Cross-method evaluation and plotting live
in the top-level compare/. The shared observation model / navigation / config live
in the top-level crowdcore/.
"""
