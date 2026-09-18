"""
summarize_grid.py — verdict for the variant G (grid latent) experiment
=======================================================================

Reads the nine compare5 evaluations in `check_outputs/grid/eval/<config>_s<seed>.json`
(or `eval_dbg/`, where the same evaluation was re-submitted on gpu-debug)
(produced with `compare.compare5 --senseiver <ckpt> --arms '' --no-with-ensemble`,
whose Senseiver row reproduces the published numbers exactly) and applies the
decision rule fixed in the README before any result existed:

  primary metric: blind-walkable pooled RMSE (`walkable`, the reported scope)
  G beats A40 only if all three hold:
    1. mean over the 3 seeds is lower
    2. G's worst seed beats A40's best seed
    3. no channel's 3-seed mean blind-walkable MSE degrades by more than 5%

Also printed, not deciding: paired per-seed differences (same seed = same training
observations), all-walkable RMSE, Senseiver's own seed spread (never measured
before), A40_s123 vs the published 100-epoch variant A (same seed, so the
difference is the epoch budget alone), and the `defined` cell set as a diagnostic.

Pure JSON, no torch:  python3 -m methods.senseiver.checks.summarize_grid
"""
from __future__ import annotations

import json
import os
import statistics as st

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(HERE))
EVAL = os.path.join(HERE, "check_outputs", "grid", "eval")
EVAL_DBG = EVAL + "_dbg"                   # same evaluation, submitted on gpu-debug
PUBLISHED = os.path.join(ROOT, "compare", "results", "compare5_final.json")
SEEDS = (123, 124, 125)
CONFIGS = (("A40", "A40"), ("G-dec", "Gdec"), ("G-direct", "Gdirect"))
CH = ("density", "vx", "vy", "var")


def load(dirname, seed):
    for d in (EVAL, EVAL_DBG):
        path = os.path.join(d, f"{dirname}_s{seed}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)["results"]["Senseiver"]
    raise FileNotFoundError(f"{dirname}_s{seed}.json in neither {EVAL} nor {EVAL_DBG}")


def main():
    R = {name: {s: load(d, s) for s in SEEDS} for name, d in CONFIGS}
    with open(PUBLISHED) as f:
        pub = json.load(f)["results"]
    out = {"rule": "blind-walkable pooled RMSE; mean lower AND worst G seed < best A40 seed "
                   "AND no channel mean degrades > 5%", "configs": {}, "verdicts": {}}

    print(f"{'':<10}{'seed':>6}{'blind RMSE':>12}{'all RMSE':>10}" + "".join(f"{c:>9}" for c in CH)
          + f"{'vx(defined)':>13}")
    for name, _ in CONFIGS:
        rows = {}
        for s in SEEDS:
            w, wf, df = R[name][s]["walkable"], R[name][s]["walkable_full"], R[name][s]["defined"]
            rows[s] = dict(blind=w["rmse_pooled"], full=wf["rmse_pooled"],
                           ch={c: w["per_channel"][c] for c in CH}, vx_def=df["per_channel"]["vx"])
            print(f"{name:<10}{s:>6}{rows[s]['blind']:>12.4f}{rows[s]['full']:>10.4f}"
                  + "".join(f"{rows[s]['ch'][c]:>9.4f}" for c in CH) + f"{rows[s]['vx_def']:>13.4f}")
        b = [rows[s]["blind"] for s in SEEDS]
        agg = dict(blind_mean=st.mean(b), blind_min=min(b), blind_max=max(b),
                   blind_spread_pct=(max(b) - min(b)) / min(b) * 100,
                   full_mean=st.mean(rows[s]["full"] for s in SEEDS),
                   ch_mean={c: st.mean(rows[s]["ch"][c] for s in SEEDS) for c in CH},
                   vx_defined_mean=st.mean(rows[s]["vx_def"] for s in SEEDS),
                   per_seed=rows)
        out["configs"][name] = agg
        print(f"{name:<10}{'mean':>6}{agg['blind_mean']:>12.4f}{agg['full_mean']:>10.4f}"
              + "".join(f"{agg['ch_mean'][c]:>9.4f}" for c in CH) + f"{agg['vx_defined_mean']:>13.4f}"
              + f"   seed spread {agg['blind_spread_pct']:.2f}%")
        print()

    A = out["configs"]["A40"]
    print("=" * 96)
    for g in ("G-dec", "G-direct"):
        G = out["configs"][g]
        c1 = G["blind_mean"] < A["blind_mean"]
        c2 = G["blind_max"] < A["blind_min"]
        deg = {c: (G["ch_mean"][c] - A["ch_mean"][c]) / A["ch_mean"][c] * 100 for c in CH}
        c3 = max(deg.values()) <= 5.0
        paired = {s: G["per_seed"][s]["blind"] - A["per_seed"][s]["blind"] for s in SEEDS}
        verdict = c1 and c2 and c3
        out["verdicts"][g] = dict(mean_lower=c1, worst_beats_best=c2, no_channel_degrades=c3,
                                  channel_change_pct=deg, paired_blind_rmse_diff=paired,
                                  blind_mean_change_pct=(G["blind_mean"] - A["blind_mean"]) / A["blind_mean"] * 100,
                                  beats_A40=verdict)
        print(f"{g} vs A40: blind RMSE mean {G['blind_mean']:.4f} vs {A['blind_mean']:.4f} "
              f"({out['verdicts'][g]['blind_mean_change_pct']:+.2f}%)")
        print(f"  1. mean lower                     {'PASS' if c1 else 'fail'}")
        print(f"  2. worst G seed {G['blind_max']:.4f} < best A40 seed {A['blind_min']:.4f}   {'PASS' if c2 else 'fail'}")
        print(f"  3. no channel degrades > 5%        {'PASS' if c3 else 'fail'}   "
              + "  ".join(f"{c} {v:+.1f}%" for c, v in deg.items()))
        print(f"  paired per seed (G - A40)          "
              + "  ".join(f"s{s} {d:+.4f}" for s, d in paired.items()))
        print(f"  => {'G BEATS A40' if verdict else 'G does NOT beat A40 under the rule'}")
        print()

    off = pub["Senseiver"]["walkable"]["rmse_pooled"]
    a123 = A["per_seed"][123]["blind"]
    print(f"Epoch budget (same seed 123): A40 {a123:.4f} vs published 100-epoch A {off:.4f} "
          f"({(a123 - off) / off * 100:+.2f}%)")
    print("Published context (blind walkable RMSE, one model per method): "
          + ", ".join(f"{k} {pub[k]['walkable']['rmse_pooled']:.4f}"
                      for k in ("Senseiver", "DINCAE", "4DVarNet MSE s3", "EnKF k1") if k in pub))
    out["epoch_budget"] = dict(A40_s123=a123, published_A_100ep=off)

    dst = os.path.join(EVAL, "summary.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[out] {dst}")


if __name__ == "__main__":
    main()
