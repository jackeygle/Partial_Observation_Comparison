"""
plotstyle.py — 所有图共用的样式。只固定三件事，别的一概不管。

存在的理由（都是实测出来的，不是洁癖）：

  1. **方法 -> 颜色的映射**。重构前配色是逐脚本定的，同一个方法在不同图里颜色不同。
     这是投入产出比最高的一条：读者一旦记住"蓝色是 DINCAE"，后面每张图都省一次查图例。

  2. **语义 -> colormap 一对一**。重构前 `Blues` 出现 4 次、`gray` 3 次、`hot` 2 次、
     `Reds` 2 次，其中 `Reds` 同时被用来表示"集合离散度"和"误差"，密度在有的图里是
     Blues 有的图里是 hot。同一个 colormap 表示两种量，读者就没法跨图比较。

  3. **图宽固定为论文正文宽**。重构前字号散落在 23 个不同值（8.2/8.4/8.6/8.8/9.0/9.2/
     9.4/9.5/9.8/10/10.2/10.5/11/11.5/12/12.2/12.5/13/14/16...），根因不是手抖 ——
     是每张图被缩放到不同宽度后再手调字号补偿。宽度固定后这个问题自动消失。

用法:

    from compare import plotstyle as ps
    ps.use()                                  # 调 rcParams，进程里调一次
    fig, axes = ps.figure(ncols=4, rows_h=2.2)
    ax.bar(..., color=ps.METHOD_COLORS["DINCAE"])
    im = ax.imshow(density, cmap=ps.CMAP["density"], vmin=0, vmax=vmax)

**图上的文字一律用英文。**Triton 的 matplotlib 只有 24 个字体，一个 CJK 的都没有，
中文会被渲染成豆腐块并抛一串 "Glyph ... missing from font" 警告。注释和文档写中文，
图上写英文 —— 现有的图（plot_compare3 等）本来就是这个约定。

不做的事：不替你选图型、不替你算 vmin/vmax、不画图例。那些和具体图有关，
塞进公共模块只会变成一堆开关。
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# 1. 方法 -> 颜色
# --------------------------------------------------------------------------- #
#: 分类色板。取自 dataviz 规范的分类槽位，固定顺序、不循环。
#: 五个方法各占一槽，**跨全部图、全部子项目一致**。
#: 消融行（a4_k1、单成员）用同色系的浅色，见 `shade()`。
METHOD_COLORS = {
    "DINCAE":    "#4269d0",   # 蓝
    "4DVarNet":  "#efb118",   # 金
    "Senseiver": "#ff725c",   # 橙红
    "EnKF":      "#6cc5b0",   # 青
    "4DVarNet+var": "#a463f2",  # 紫 —— 不确定性头单独一槽，它是第五个方法
}

#: 中性色：轴、文字、网格。三档，不要再加。
INK = "#1b1b1b"          # 正文与轴标签
INK_MUTED = "#6b6b6b"    # 次要标注、脚注
RULE = "#d9d9d9"         # 网格线与分隔线


def method_color(name: str) -> str:
    """按行名取颜色，认得 compare5 的行名（"4DVarNet NLL ens5"、"EnKF k1" …）。

    找不到就返回中性灰而不是抛异常 —— 图里多一条灰线比整个脚本崩掉好，
    而且灰色本身就在提示"这一行没有登记颜色"。
    """
    n = name.strip()
    if n.startswith("4DVarNet"):
        return METHOD_COLORS["4DVarNet+var"] if ("NLL" in n or "nll" in n) \
            else METHOD_COLORS["4DVarNet"]
    for k, v in METHOD_COLORS.items():
        if n.startswith(k):
            return v
    return INK_MUTED


def shade(hex_color: str, amount: float = 0.55) -> str:
    """把颜色往白色混，用于同一方法的消融/单成员行。amount=0 原色，1 全白。"""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    mix = lambda c: int(round(c + (255 - c) * amount))
    return "#%02x%02x%02x" % (mix(r), mix(g), mix(b))


# --------------------------------------------------------------------------- #
# 2. 语义 -> colormap
# --------------------------------------------------------------------------- #
#: 一个 colormap 只表示一种量。加新语义就在这里加一行，不要在画图脚本里现选。
CMAP = {
    "density":   "Blues",      # 人群密度，顺序、零值为白
    "abs_error": "Purples",    # |误差|，顺序 —— 刻意不用 Reds，那个留给 spread
    "spread":    "Reds",       # 集合离散度 / σ̂，顺序
    "signed":    "RdBu_r",     # 有符号量（vx、vy、残差），零值居中
    "mask":      "gray",       # walkable / 观测掩膜这类二值图
}

#: 无定义格子的填充色。速度通道有 88.4% 的盲区格子落在这里，
#: 画出来才能让"口径为什么决定胜负"看得见。
UNDEFINED = "#ececec"


# --------------------------------------------------------------------------- #
# 3. 尺寸与 rcParams
# --------------------------------------------------------------------------- #
#: 论文正文宽（英寸）。A4 + 2.5cm 页边距约 6.3in。图一律按这个宽度出，
#: 插进文档时**不要缩放** —— 缩放就是 23 种字号的来源。
TEXT_W = 6.3
#: 幻灯片宽（16:9 正文区）
SLIDE_W = 9.5

#: 字号只留三档。够用了；不够用说明该拆图，不是该加字号。
FS_TITLE, FS_LABEL, FS_TICK = 10.0, 9.0, 8.0


def use(dpi: int = 200):
    """调 rcParams。一个进程里调一次就行。"""
    plt.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "savefig.bbox": "tight",
        "font.size": FS_LABEL,
        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_TICK,
        "axes.edgecolor": RULE,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,           # 网格在数据下面，不然柱子被网格切开
        "grid.color": RULE,
        "grid.linewidth": 0.6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def figure(ncols: int = 1, nrows: int = 1, width: float = TEXT_W,
           rows_h: float = 2.4, **kw):
    """按固定宽度建图。高度按行数给，宽度不给调用方选 —— 那是重点。"""
    return plt.subplots(nrows, ncols, figsize=(width, rows_h * nrows), **kw)


def save(fig, path: str):
    """存图并打印路径（各脚本原来都自己 print 一行 [figure] ...，统一到这里）。"""
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"[figure] {path}", flush=True)
