"""compare — cross-method evaluation and plotting. The only place allowed to import
multiple methods at once.

Why this exists: before this, cross-method scripts were scattered in two places —
compare3/compare4 lived under senseiver_crowd/checks/ (yet had to read all four
methods), and eval_threeway_accuracy lived under 4dvarnet_enkf/checks/. The result
was two mutually contradicting scripts for the same quantity: 4DVarNet's blind RMSE
was reported as 0.1912 in one and 0.2198 in the other, a 15% gap never tracked down.

Each quantity should have exactly one implementation. New cross-method evaluation
goes here, not back into some method's checks/.
"""
