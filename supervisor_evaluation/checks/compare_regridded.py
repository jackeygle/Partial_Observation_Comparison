"""Compare a regenerated trajectory H5 and grid_cache file with the existing ones (route 2 check).

    python3 -m supervisor_evaluation.checks.compare_regridded --new-traj A --old-traj B --new-grid C --old-grid D
"""
import argparse, json
import h5py, numpy as np
ap = argparse.ArgumentParser()
for k in ("new_traj", "old_traj", "new_grid", "old_grid"):
    ap.add_argument("--" + k.replace("_", "-"), required=True)
a = ap.parse_args()
out = {}
with h5py.File(a.new_traj, "r") as n, h5py.File(a.old_traj, "r") as o:
    for k in ("position", "velocity"):
        x, y = n[k][:], o[k][:]
        out[k] = {"shape_new": x.shape, "shape_old": y.shape,
                  "max_abs_diff": float(np.abs(x - y).max()) if x.shape == y.shape else None}
with h5py.File(a.new_grid, "r") as n, h5py.File(a.old_grid, "r") as o:
    x, y = n["grid"][:], o["grid"][:]
    out["grid"] = {"shape_new": x.shape, "shape_old": y.shape,
                   "identical": bool(x.shape == y.shape and np.array_equal(x, y)),
                   "max_abs_diff": float(np.abs(x - y).max()) if x.shape == y.shape else None}
    out["time_identical"] = bool(np.array_equal(n["time"][:], o["time"][:]))
    out["attrs_equal"] = {k: bool(np.all(np.asarray(n.attrs[k]) == np.asarray(o.attrs[k]))) for k in o.attrs}
print(json.dumps(out, indent=1, default=str))
