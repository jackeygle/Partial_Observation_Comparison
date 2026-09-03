"""
plot_compare5.py — 主结果图：五方 × 四通道 × 两套口径

读 `compare/results/compare5.json`，画一张图讲完全部主结果，外加"名次会翻转"这件事。

版面：**两行 × 五列的小多组**

    行 = 口径      上：有定义格子 ∩ walkable ∩ 盲区（五方可比）
                   下：所有格子 ∩ 盲区（旧口径，供对照）
    列 = 通道      density / vx / vy / var / 合计

为什么是小多组而不是分组柱状图：四个通道量级差一个数量级（vx 0.50、var 0.04）。
放同一根 y 轴上，density/vy/var 三个面板会被压成看不见的细条，读者只能看出 vx 的
差异 —— 而那恰好是唯一 4DVarNet 领先的通道，图会给出与数据相反的印象。
每通道一个面板、各自 y 轴，比的是"该通道内谁赢"，这正是要传达的信息。
面板之间不可比，写在副标题里。

为什么两行必须放在同一张图：**名次会随口径翻转**（上行 DINCAE 第一，下行它垫底
且差 12 倍）。拆成两张图放在不同章节，读者就得靠正文里一句"名次翻转了"去相信；
并排放着，翻转是看出来的。

y 轴一律线性、逐面板自适应，不做对数也不截断。下行 vy 那一格里 DINCAE 是 1.65 而
其余是 0.02，于是其余四根柱子几乎贴地 —— 那正是实情，而且每根柱子头上都标了数值，
信息没有丢。为了让小柱子好看而改成对数，等于把"差 80 倍"这个结论柔化掉。

用法（登录节点即可，纯 matplotlib）:
    source sbatch/_env.sh
    python3 -m compare.plot_compare5
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from compare import plotstyle as ps
from crowdcore import paths

#: 主表只画这五个方法。a4_k1、各单成员属于消融，另有图。
#: 值是 compare5.json 里的行名；None 表示按前缀找（成员数可能不是 5）。
ROWS = [
    ("DINCAE",                 "DINCAE"),
    ("4DVarNet MSE ens",       "4DVarNet (MSE loss)"),
    ("4DVarNet NLL ens",       "4DVarNet + uncertainty head"),
    ("Senseiver",              "Senseiver"),
    ("EnKF k1",                "EnKF"),
]

PANELS = [
    ("defined",  "channel-defined cells\n(blind, walkable)"),
    ("allcells", "all cells\n(blind, old convention)"),
]


def pick(results: dict, prefix: str):
    """按前缀取行，容忍 ens5/ens2 这种成员数差异。"""
    for k in results:
        if k.startswith(prefix):
            return k, results[k]
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(paths.COMPARE, "results", "compare5.json"))
    ap.add_argument("--out", default=os.path.join(paths.COMPARE, "results", "compare5.png"))
    args = ap.parse_args()

    with open(args.json) as f:
        doc = json.load(f)
    results, chans = doc["results"], doc["channels"]
    cols = chans + ["total"]

    rows = []
    for prefix, label in ROWS:
        key, entry = pick(results, prefix)
        if entry is None:
            print(f"  [warn] {args.json} 里没有 {prefix}*，跳过这一行")
            continue
        rows.append((label, ps.method_color(key), entry))

    ps.use()
    # wspace 给得大：每个面板有自己的 y 轴，刻度标签需要横向空间，否则会压到
    # 左邻面板的柱子上。第一版没给，vy 的 "0.150" 直接盖在 vx 的柱子上。
    fig, axes = ps.figure(ncols=len(cols), nrows=len(PANELS),
                          width=ps.TEXT_W * 1.75, rows_h=2.9,
                          gridspec_kw={"wspace": 0.52, "hspace": 0.22})
    axes = np.atleast_2d(axes)

    for r, (conv, conv_title) in enumerate(PANELS):
        for c, ch in enumerate(cols):
            ax = axes[r, c]
            vals, colors, labels = [], [], []
            for label, color, entry in rows:
                q = entry.get(conv)
                if q is None:
                    continue
                vals.append(q["overall"] if ch == "total" else q["per_channel"][ch])
                colors.append(color)
                labels.append(label)

            bars = ax.bar(range(len(vals)), vals, color=colors, width=0.66, zorder=3)
            hi = max(vals) if vals else 1.0
            # 数值竖排：5 根柱子 × 4 位小数，横排一定会撞在一起（第一版就撞成
            # ".0242.0246" 这样）。竖排后每个标签只占一根柱子的宽度。
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + hi * 0.04, f"{v:.4f}".lstrip("0"),
                        ha="center", va="bottom", fontsize=6.2, color=ps.INK_MUTED,
                        rotation=90)
            ax.set_ylim(0, hi * 1.42)          # 留够竖排标签的高度
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels([], )
            ax.tick_params(axis="x", length=0)
            ax.set_title(ch if r == 0 else "", fontsize=ps.FS_LABEL, fontweight="bold",
                         color=ps.INK, pad=8)
            if c == 0:
                ax.set_ylabel(f"{conv_title}\nblind MSE", fontsize=7.4, color=ps.INK)

    # 图例放最下方一行，五个方法各一个色块 —— 面板里不再重复标签
    handles = [__import__("matplotlib").patches.Patch(facecolor=col, label=lab)
               for lab, col, _ in rows]
    fig.legend(handles=handles, loc="lower center", ncol=len(rows), frameon=False,
               bbox_to_anchor=(0.5, -0.035), fontsize=ps.FS_TICK)

    n_days = doc.get("protocol", {}).get("n_days", "?")
    fig.suptitle(
        f"Per-channel reconstruction error, five methods  —  {n_days} held-out days, "
        "full day, obs_every_k=1, identical clipping\n"
        "Each panel has its own y axis: the four channels differ by an order of magnitude, "
        "so a shared axis would flatten density/vy/var under vx. Panels are not comparable "
        "to each other.",
        fontsize=ps.FS_TICK, color=ps.INK_MUTED, y=1.04)

    ps.save(fig, args.out)


if __name__ == "__main__":
    main()
