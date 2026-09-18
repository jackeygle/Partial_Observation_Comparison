"""Paper-style local covariance learning for the PedPred3 Kalman filter."""

from .targets import (extract_centered_patches, local_error_outer_products,
                      normalized_local_targets, sampled_normalized_targets)

__all__ = ("extract_centered_patches", "local_error_outer_products",
           "normalized_local_targets", "sampled_normalized_targets")
