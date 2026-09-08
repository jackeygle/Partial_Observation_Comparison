"""compare — cross-method evaluation and plotting. The only place allowed to import
multiple methods at once.

Why this exists: before this, cross-method scripts were scattered in two places —
compare3/compare4 lived under senseiver_crowd/checks/ (yet had to read all four
methods), and eval_threeway_accuracy lived under 4dvarnet_enkf/checks/. The result
was two mutually contradicting scripts for the same quantity: 4DVarNet's blind RMSE
was reported as 0.1912 in one and 0.2198 in the other, a 15% gap never tracked down.

That duplication was removed on 2026-09-08: compare3.py, plot_compare3.py and
eval_threeway_accuracy.py are deleted, and `compare5.py` is the single
implementation. The 15% gap is therefore resolved by construction rather than by
diagnosis -- there is no longer a second number to disagree with. Ruled out before
giving up on the cause: scoring convention (the EnKF row agreed to 0.06% across
both scripts), physical clipping (accounts for 0.8% of the gap), the observation
realisation (bit-identical masks and values), and init_method. Whatever remained
lived in the 4DVarNet inference path only, and affected b0_k1 (15%) far more than
the NLL runs (3-5%). Recoverable from git history if it ever matters again.

Each quantity should have exactly one implementation. New cross-method evaluation
goes here, not back into some method's checks/.
"""
