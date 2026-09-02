# `enkf_lab/` — pristine copy of the EnKF baseline (DO NOT EDIT)

A byte-for-byte copy of the EnKF baseline from `/scratch/work/zhangx29/Partial_observation`,
vendored here so experiments never touch that project. **This copy is the reference: it must
stay identical to the original.** The `.py`/`.yaml` files are chmod'd read-only on purpose.

Modify `../enkf_opt/` instead. `enkf_lab` exists so any change made there can be checked
against an unmodified baseline — same inputs must give bit-identical outputs.

## Contents

Dependency closure of `ENKF.py`, nothing more:

    ENKF.py                       the filter (LocalizedEnsembleKalmanFilter)
      └── utils.py                load_model
            └── config.py, dataset.py, grid.py, models.py (PedPred3)
                  └── tools/      gauss, gpytorch, mpl, torch
    metrics.py                    the project's own scoring helpers
    Parameters.yaml               read at import time by ENKF.py:8 — required
    pedpred -> .                  self-link, so in-package `from pedpred.X import Y` resolves
                                  (mirrors the original project's own layout)
    apt-ibex_train_model_28D.pth  symlink to the original (14 MB, not duplicated)

## Provenance

Copied 2026-08-02 from `Partial_observation` (source files dated 2026-05-14 / 05-20).
The original project has not been modified — verified by mtime.

## Verifying a copy against the original

    obs_cells from a real exported Omega, then
    LocalizedEnsembleKalmanFilter._localization_matrix(obs_cells)
    must satisfy np.array_equal(original, copy)

Checked at vendoring time: (252, 1728), sum = 94010.379333517776, bit-identical.
