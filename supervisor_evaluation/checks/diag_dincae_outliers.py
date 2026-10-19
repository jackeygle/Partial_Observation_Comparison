"""Where do DINCAE's unclipped predictions leave the physical range? (route 2 test days)

    DATA_VARIANT=all92 EVAL_VARIANT=six_allwalk_all92 \
        python3 -m supervisor_evaluation.checks.diag_dincae_outliers [DAY ...]

The uncertainty protocol scores each method's own distribution without clipping. On all
walkable cells DINCAE's velocity-variance RMSE is far above its clipped value, from a few cells.
For each day (default: every test day), on the frames and cells the uncertainty protocol scores
(after the EnKF warm-up, complete 4DVarNet windows, gap blocks excluded, walkable cells), this
counts the predicted means outside the bounds (density [0,5], velocity [-5,5], variance [0,2]),
per channel and split into observed / blind cells, with their largest value, their frames, the
predicted sigma there, and their share of the squared error of that channel and scope.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate as ev  # noqa: E402

from crowdcore import frame_reliability as fr, navigation, observation_model as om  # noqa: E402
from methods.dincae.checks import evaluate as dincae_eval  # noqa: E402

LO = np.array([0, -5, -5, 0], np.float32); HI = np.array([5, 5, 5, 2], np.float32)
CH = ("density", "vx", "vy", "variance")


def main() -> None:
    want = sys.argv[1:]
    dev = torch.device("cuda")
    models, _, _ = dincae_eval.load_models(str(ev.MODELS), str(ev.MODELS / ev.FINAL_CONFIG["dincae"]["file"]), dev)
    stats = dincae_eval.StateStats()
    std = stats.std.reshape(1, -1, 1, 1)
    walk = navigation.build_valid_mask_from_config().astype(bool)
    warmup = int(ev.FINAL_CONFIG["enkf"]["warmup"])
    report = []
    for fp in om.split_files("test"):
        day = Path(fp).name.split("_")[0]
        if want and not any(w in day for w in want):
            continue
        Xt, rec, _, sd_n, M = dincae_eval.predict_day(models, stats, str(fp), dev, 0, 256)
        n = len(Xt) + 2                                    # rows are frames 1..n-2
        lo, hi = max(1, warmup), min(n - 1, (n // 200) * 200)
        keep = fr.scorable_frames(fp, n)[lo:hi]
        X, R, S, O = Xt[lo - 1:hi - 1], rec[lo - 1:hi - 1], sd_n[lo - 1:hi - 1] * std, M[lo - 1:hi - 1, 0]
        frames = np.arange(lo, hi)
        cell = walk[None] & keep[:, None, None]            # (T,H,W) scored walkable cells
        row = {"day": day, "frames_scored": int(keep.sum())}
        for c, name in enumerate(CH):
            out = (R[:, c] < LO[c]) | (R[:, c] > HI[c])
            se = (R[:, c] - X[:, c]).astype(np.float64) ** 2
            for scope, sel in (("observed", cell & O), ("blind", cell & ~O)):
                bad = out & sel
                width = HI[c] - LO[c]                      # "far": more than one range width outside
                far = ((R[:, c] < LO[c] - width) | (R[:, c] > HI[c] + width)) & sel
                t_out = ((X[:, c] < LO[c]) | (X[:, c] > HI[c])) & sel
                ent = {"cells": int(bad.sum()), "share_of_cells": float(bad.sum() / max(sel.sum(), 1)),
                       "share_of_squared_error": float(se[bad].sum() / max(se[sel].sum(), 1e-300)),
                       "scored_cells": int(sel.sum()),
                       "far_cells": int(far.sum()),
                       "far_share_of_squared_error": float(se[far].sum() / max(se[sel].sum(), 1e-300)),
                       "far_cells_truth_out_of_range": int((far & t_out).sum()),
                       "truth_out_of_range_cells": int(t_out.sum()),
                       "truth_max": float(X[:, c][sel].max()) if sel.any() else None}
                if bad.any():
                    t, i, j = np.nonzero(bad)
                    k = int(np.argmax(np.abs(R[:, c][bad])))
                    ent.update(max_abs_value=float(np.abs(R[:, c][bad]).max()),
                               value_at_max=float(R[:, c][bad][k]), truth_at_max=float(X[:, c][bad][k]),
                               sigma_at_max=float(S[:, c][bad][k]),
                               median_sigma_out=float(np.median(S[:, c][bad])),
                               frames=sorted(set(int(x) for x in frames[t]))[:20],
                               n_frames=len(set(t.tolist())),
                               cells_rc=sorted(set((int(a), int(b)) for a, b in zip(i, j)))[:10])
                row[f"{name}_{scope}"] = ent
        report.append(row)
        print(json.dumps(row), flush=True)
    out = ev.HERE / "outputs/full_6m_all92/dincae_outliers_v2.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"[out] {out}", flush=True)


if __name__ == "__main__":
    main()
