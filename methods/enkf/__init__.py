"""EnKF baseline — not written by us, vendored in.

  enkf_lab/   a byte-identical, read-only copy of /scratch/work/zhangx29/Partial_observation.
              **Do not edit** — the files are deliberately chmod 444.
  enkf_opt/   a copy that may be modified; any change must pass the bit-identical
              comparison in checks/verify_enkf_opt.py (np.array_equal, not isclose).

Note that enkf_lab/pedpred/ and enkf_opt/pedpred/ are **deliberately not**
subpackages of this package: they contain a `pedpred -> .` self-symlink, relying on
that directory being placed on sys.path so that `from pedpred.X import Y` inside the
package resolves — this is the original project's layout, and changing it would mean
modifying the vendor copy. So neither directory has an __init__.py continuing down
from methods.enkf; they are loaded by path instead.
"""
