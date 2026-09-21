"""LCSKF's own diagnostics and evaluation drivers.

The uncertainty scorer itself is shared with the EnKF baseline and stays there:
``methods.enkf.checks.eval_uncertainty_enkf`` scores any exported directory of
``est_<day>.npz``, whichever method wrote it.
"""
