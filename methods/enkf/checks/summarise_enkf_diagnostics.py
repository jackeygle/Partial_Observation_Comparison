"""
summarise_enkf_diagnostics.py — one table from every EnKF diagnostic in check_outputs/eval.

The investigation produced five result files per variant and they only mean something read
together, so this collects them instead of leaving the reader to open five jsons:

    proc_scale_<s>[_FIXED].json          injection multiplier sweep (calibration)
    correction_paths_<arm>[_FIXED].json  gain vs the EMA bias path (who assimilates)
    gain_census[_FIXED].json             why the gain is the size it is
    observed_vs_blind.json               is "observed scores worse" a composition effect

"as published" is the shipped configuration, bug included -- the run of record that every
reported EnKF number came from. "FIXED" restores the feature-block division dropped by the
localisation fallback, so all four fields are assimilated rather than density alone.

    python3 -m methods.enkf.checks.summarise_enkf_diagnostics
"""
from __future__ import annotations
import argparse
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from crowdcore import paths                                              # noqa: E402

CH = ("density", "vx", "vy", "var")
ARM_ORDER = ("A_forecast_only", "B_bias_only", "C_gain_only", "D_shipped")
ARM_LABEL = {"A_forecast_only": "A 纯预报", "B_bias_only": "B 只有偏差修正",
             "C_gain_only": "C 只有卡尔曼增益", "D_shipped": "D 出厂(两者)"}


def load(d, name):
    p = os.path.join(d, name)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=paths.eval_out(paths.ENKF))
    a = ap.parse_args()
    d = a.dir

    # ---- 1. correction paths: who actually assimilates -------------------------------
    print("=" * 78)
    print("1. 两条修正通路的分解  (RMSE,frames 500-2000,atc-20130811)")
    print("=" * 78)
    print(f"{'臂':20s}{'as published':>15s}{'localisation FIXED':>22s}")
    base = {}
    for tag in ("", "_FIXED"):
        for arm in ARM_ORDER:
            r = load(d, f"correction_paths_{arm}{tag}.json")
            if r:
                base[(arm, tag)] = r["arms"][arm]["all"]["rmse"]
    for arm in ARM_ORDER:
        pub, fix = base.get((arm, "")), base.get((arm, "_FIXED"))
        print(f"{ARM_LABEL[arm]:20s}"
              f"{(f'{pub:.4f}' if pub else '-'):>15s}"
              f"{(f'{fix:.4f}' if fix else '-'):>22s}")
    for tag, nm in (("", "as published"), ("_FIXED", "loc FIXED"),
                    ("_FIXED_Q1", "FIXED + 完整Q")):
        A, C, D = base.get(("A_forecast_only", tag)), base.get(("C_gain_only", tag)), \
                  base.get(("D_shipped", tag))
        if A and C and D:
            print(f"  [{nm:14s}] 增益单独 A->C: {1 - C / A:+7.2%}    "
                  f"全部 A->D: {1 - D / A:+7.2%}")

    # 分层：空格子占 ~82%，会主导 "all"。这两个切分才回答"增益在答案非平凡的地方有没有用"
    for split, label in (("defined", "有定义格子 (channel_valid ∩ walkable)"),
                         ("occupied", "有人格子 (density > 0)")):
        rows = {}
        for tag in ("", "_FIXED", "_FIXED_Q1"):
            for arm in ARM_ORDER:
                r = load(d, f"correction_paths_{arm}{tag}.json")
                if r and split in r["arms"][arm]:
                    rows[(arm, tag)] = r["arms"][arm][split]["rmse"]
        if not rows:
            continue
        print()
        print(f"  -- {label} --")
        print(f"  {'臂':20s}{'as published':>15s}{'loc FIXED':>13s}{'FIXED + 完整Q':>17s}")
        for arm in ARM_ORDER:
            cells = [rows.get((arm, t)) for t in ("", "_FIXED", "_FIXED_Q1")]
            print(f"  {ARM_LABEL[arm]:20s}" + "".join(
                (f"{v:15.4f}" if i == 0 else f"{v:13.4f}" if i == 1 else f"{v:17.4f}")
                if v is not None else "-".rjust([15, 13, 17][i])
                for i, v in enumerate(cells)))
        for tag, nm in (("", "as published"), ("_FIXED", "loc FIXED"),
                        ("_FIXED_Q1", "FIXED + 完整Q")):
            A, C = rows.get(("A_forecast_only", tag)), rows.get(("C_gain_only", tag))
            if A and C:
                print(f"    [{nm:14s}] 增益单独 A->C: {1 - C / A:+7.2%}")

    # ---- 2. calibration ---------------------------------------------------------------
    print()
    print("=" * 78)
    print("2. 注入噪声与标定")
    print("=" * 78)
    print(f"{'配置':28s}{'RMSE':>9s}{'sigma':>11s}{'spread/skill':>14s}{'cov@90':>9s}")
    for f in sorted(glob.glob(os.path.join(d, "proc_scale_*.json"))):
        r = json.load(open(f))
        s = list(r["arms"])[0]
        al = r["arms"][s]["all"]
        lab = f"scale={s}" + ("  FIXED" if r.get("fix_localization") else "  as published")
        print(f"{lab:28s}{al['rmse']:9.4f}{al['sigma_mean']:11.5f}"
              f"{al['spread_skill']:14.4f}{al['coverage']['90']:9.3f}")

    # ---- 3. ranking power per channel -------------------------------------------------
    print()
    print("=" * 78)
    print("3. sigma 的排序能力  (Spearman,越大越好;AUSE,越小越好)")
    print("=" * 78)
    print(f"{'配置':28s}" + "".join(f"{c:>16s}" for c in CH))
    for f in sorted(glob.glob(os.path.join(d, "proc_scale_*.json"))):
        r = json.load(open(f))
        s = list(r["arms"])[0]
        pc = r["arms"][s]["per_channel"]
        lab = f"scale={s}" + ("  FIXED" if r.get("fix_localization") else "  as published")
        print(f"{lab:28s}" + "".join(
            f"{pc[c]['spearman']:+.3f}/{pc[c]['ause']:.2f}".rjust(16) for c in CH))

    # ---- 4. gain census ---------------------------------------------------------------
    print()
    print("=" * 78)
    print("4. 自增益 K[格子, 该格子自己的观测]")
    print("=" * 78)
    print(f"{'配置':28s}" + "".join(f"{c:>13s}" for c in CH))
    for tag, nm in (("", "as published"), ("_FIXED", "FIXED")):
        r = load(d, f"gain_census{tag}.json")
        if not r:
            continue
        print(f"{nm:28s}" + "".join(
            f"{r['per_channel'][c]['self_gain_mean']:13.3e}" for c in CH))
    for tag, nm in (("", "as published"), ("_FIXED", "FIXED")):
        r = load(d, f"gain_census{tag}.json")
        if r:
            print(f"  [{nm}] ||K@innov|| {r['incr_norm_mean']:.3e}   "
                  f"偏差修正/analysis 增量 = {r['bias_over_incr']:.1f}x")

    # ---- 5. observed vs blind ---------------------------------------------------------
    r = load(d, "observed_vs_blind.json")
    if r:
        print()
        print("=" * 78)
        print("5. observed / blind  的 RMSE 比值,按真实密度分层  (<1 = 同化有帮助)")
        print("=" * 78)
        print(f"{'分层':16s}" + "".join(f"{c:>11s}" for c in CH))
        for nm, row in r["by_stratum"].items():
            print(f"{nm:16s}" + "".join(
                f"{row['obs_over_blind'][c]:11.3f}"
                if row["obs_over_blind"][c] is not None else f"{'-':>11s}" for c in CH))


if __name__ == "__main__":
    main()
