"""
summarize_temporal.py — verdict for the temporal extension (time window k on G-direct)
======================================================================================

Reads compare5 evaluations of the temporal runs, `check_outputs/temporal/eval/k<k>_s<seed>.json`
(`compare.compare5 --senseiver runs/temporal/k<k>_s<seed>/best.pt --arms '' --no-with-ensemble`),
and of the k = 1 baseline -- the existing G-direct runs, `check_outputs/grid/eval[_dbg]/Gdirect_s<seed>.json`
-- and applies the decision rule written in the README before any temporal model was trained:

  primary metric: blind-walkable pooled RMSE (`walkable`, the reported scope)
  a window k beats k = 1 only if all three hold:
    1. mean over the 3 seeds is lower
    2. its worst seed beats G-direct's best seed
    3. no channel's 3-seed mean blind-walkable MSE degrades by more than 5%
  best k = lowest blind-walkable mean among the k that pass; none passing means a window of
  past observations does not help this model

Also printed, not deciding: paired per-seed differences (same seed = same targets and
current-frame observations), all-walkable RMSE, per-channel changes (vx is the channel to
watch), the `defined` cell set as a diagnostic, and the gap to 4DVarNet's reported single
model on vx. A k with a missing seed is listed and left out of the verdict.

Pure JSON, no torch:  python3 -m methods.senseiver.checks.summarize_temporal
"""
from __future__ import annotations

import json
import os
import statistics as st

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(HERE))
TEMP_EVAL = os.path.join(HERE, "check_outputs", "temporal", "eval")
GRID_EVAL = [os.path.join(HERE, "check_outputs", "grid", d) for d in ("eval", "eval_dbg")]
PUBLISHED = os.path.join(ROOT, "compare", "results", "compare5_final.json")
SEEDS = (123, 124, 125)
KS = (2, 4, 8, 16)                       # k = 16 was added after the k = 2-8 results; same rule
CH = ("density", "vx", "vy", "var")


def find(candidates):
    return next((p for p in candidates if os.path.exists(p)), None)


def load(path):
    with open(path) as f:
        return json.load(f)["results"]["Senseiver"]


def summarise(rows):
    b = [rows[s]["blind"] for s in SEEDS]
    return dict(blind_mean=st.mean(b), blind_min=min(b), blind_max=max(b),
                blind_spread_pct=(max(b) - min(b)) / min(b) * 100,
                full_mean=st.mean(rows[s]["full"] for s in SEEDS),
                ch_mean={c: st.mean(rows[s]["ch"][c] for s in SEEDS) for c in CH},
                vx_defined_mean=st.mean(rows[s]["vx_def"] for s in SEEDS), per_seed=rows)


def row(r):
    return dict(blind=r["walkable"]["rmse_pooled"], full=r["walkable_full"]["rmse_pooled"],
                ch={c: r["walkable"]["per_channel"][c] for c in CH}, vx_def=r["defined"]["per_channel"]["vx"])


def main():
    configs, missing = {}, []
    base_paths = {s: find([os.path.join(d, f"Gdirect_s{s}.json") for d in GRID_EVAL]) for s in SEEDS}
    if any(p is None for p in base_paths.values()):
        raise SystemExit(f"k=1 baseline (G-direct) evaluations missing: {base_paths}")
    configs[1] = summarise({s: row(load(p)) for s, p in base_paths.items()})
    for k in KS:
        paths = {s: os.path.join(TEMP_EVAL, f"k{k}_s{s}.json") for s in SEEDS}
        absent = [s for s, p in paths.items() if not os.path.exists(p)]
        if absent:
            missing.append((k, absent))
            continue
        configs[k] = summarise({s: row(load(p)) for s, p in paths.items()})

    print(f"{'k':>3}{'seed':>6}{'blind RMSE':>12}{'all RMSE':>10}" + "".join(f"{c:>9}" for c in CH) + f"{'vx(defined)':>13}")
    for k, agg in configs.items():
        for s in SEEDS:
            r = agg["per_seed"][s]
            print(f"{k:>3}{s:>6}{r['blind']:>12.4f}{r['full']:>10.4f}" + "".join(f"{r['ch'][c]:>9.4f}" for c in CH)
                  + f"{r['vx_def']:>13.4f}")
        print(f"{k:>3}{'mean':>6}{agg['blind_mean']:>12.4f}{agg['full_mean']:>10.4f}"
              + "".join(f"{agg['ch_mean'][c]:>9.4f}" for c in CH) + f"{agg['vx_defined_mean']:>13.4f}"
              + f"   seed spread {agg['blind_spread_pct']:.2f}%\n")
    for k, absent in missing:
        print(f"  k={k}: missing seeds {absent} -- left out of the verdict")

    base = configs[1]
    out = {"rule": "blind-walkable pooled RMSE vs G-direct (k=1): mean lower AND worst seed < "
                   "G-direct best seed AND no channel mean degrades > 5%",
           "configs": {str(k): v for k, v in configs.items()}, "verdicts": {}, "missing": missing}
    print("=" * 100)
    passing = []
    for k in [k for k in KS if k in configs]:
        a = configs[k]
        c1 = a["blind_mean"] < base["blind_mean"]
        c2 = a["blind_max"] < base["blind_min"]
        deg = {c: (a["ch_mean"][c] - base["ch_mean"][c]) / base["ch_mean"][c] * 100 for c in CH}
        c3 = max(deg.values()) <= 5.0
        paired = {s: a["per_seed"][s]["blind"] - base["per_seed"][s]["blind"] for s in SEEDS}
        ok = c1 and c2 and c3
        if ok:
            passing.append(k)
        chg = (a["blind_mean"] - base["blind_mean"]) / base["blind_mean"] * 100
        out["verdicts"][str(k)] = dict(mean_lower=c1, worst_beats_best=c2, no_channel_degrades=c3,
                                       channel_change_pct=deg, paired_blind_rmse_diff=paired,
                                       blind_mean_change_pct=chg, beats_k1=ok)
        print(f"k={k} vs k=1 (G-direct): blind RMSE mean {a['blind_mean']:.4f} vs {base['blind_mean']:.4f} ({chg:+.2f}%)")
        print(f"  1. mean lower                              {'PASS' if c1 else 'fail'}")
        print(f"  2. worst seed {a['blind_max']:.4f} < G-direct best {base['blind_min']:.4f}   {'PASS' if c2 else 'fail'}")
        print(f"  3. no channel degrades > 5%                 {'PASS' if c3 else 'fail'}   "
              + "  ".join(f"{c} {v:+.1f}%" for c, v in deg.items()))
        print(f"  paired per seed (k - k1)                   " + "  ".join(f"s{s} {d:+.4f}" for s, d in paired.items()))
        print(f"  => {'BEATS k=1' if ok else 'does NOT beat k=1 under the rule'}\n")

    done = [k for k in (1,) + KS if k in configs]
    print("From one window to the next (3-seed blind-walkable mean; paired per seed):")
    steps = {}
    for a, b in zip(done, done[1:]):
        ma, mb = configs[a]["blind_mean"], configs[b]["blind_mean"]
        pd = {s: configs[b]["per_seed"][s]["blind"] - configs[a]["per_seed"][s]["blind"] for s in SEEDS}
        steps[f"{a}->{b}"] = dict(change_pct=(mb - ma) / ma * 100, paired=pd)
        print(f"  k={a:>2} -> k={b:<2}  {ma:.4f} -> {mb:.4f}  ({(mb - ma) / ma * 100:+.2f}%)   "
              + "  ".join(f"s{s} {d:+.4f}" for s, d in pd.items()))
    out["steps"] = steps
    print()

    best = min(passing, key=lambda k: configs[k]["blind_mean"]) if passing else None
    out["best_k"] = best
    print(f"Best k: {best if best is not None else 'none -- a window of past observations does not help this model under the rule'}")

    pub = json.load(open(PUBLISHED))["results"]
    ref_vx = pub["4DVarNet MSE s3"]["walkable"]["per_channel"]["vx"]
    for k, agg in configs.items():
        v = agg["ch_mean"]["vx"]
        print(f"  vx k={k}: {v:.4f}  vs 4DVarNet MSE s3 {ref_vx:.4f}  ({(v - ref_vx) / ref_vx * 100:+.1f}%)")
    print("Published context (blind walkable RMSE, one model per method): "
          + ", ".join(f"{m} {pub[m]['walkable']['rmse_pooled']:.4f}"
                      for m in ("Senseiver", "DINCAE", "4DVarNet MSE s3", "EnKF k1") if m in pub))

    os.makedirs(TEMP_EVAL, exist_ok=True)
    dst = os.path.join(TEMP_EVAL, "summary.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[out] {dst}")


if __name__ == "__main__":
    main()
