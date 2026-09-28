"""Thesis figure: DINCAE on the crowd grid.

    python3 -m thesis_figures.dincae            # -> thesis_figures/out/dincae.{pdf,png}

Input (information form, 2.0 sec. 2.3): for each of the frames t-1, t, t+1 and each of the
four variables, the scaled residual y/sigma^2 and the precision 1/sigma^2 (0 where not
observed), plus the two cell coordinates and the cos/sin of the daily and weekly cycle:
30 channels. A U-Net with three mean-pooling levels (36x12 -> 18x6 -> 9x3 -> 5x2, 32/64/96
filters) and additive skips; a second, identical U-Net refines from [input, first output].
The output per variable is (m/sigma^2, log 1/sigma^2), i.e. a mean and a sigma-hat,
trained with a Gaussian negative log-likelihood. Thumbnails: DINCAE on the example frame.
"""
from __future__ import annotations

import os

import numpy as np
from matplotlib.patches import Rectangle

from compare import plotstyle as ps
from thesis_figures.common import (FS, FS_SMALL, GREY, LINE, arrow, block, canvas, field_rgb,
                                   method_color, save, text, thumbnail)

HERE = os.path.dirname(os.path.abspath(__file__))
C = method_color("DINCAE")


def unet(ax, x0, yc, scale=1.0, labels=True):
    """Encoder-decoder drawn as feature maps whose height follows the spatial size."""
    levels = [(36, 12, 32), (18, 6, 64), (9, 3, 96), (5, 2, 96)]
    hs = [2.3 * scale * h / 36 for h, _, _ in levels]
    w, step = 0.22 * scale, 0.62 * scale
    enc_x = [x0 + i * step for i in range(len(levels))]
    dec_x = [x0 + (2 * len(levels) - 2 - i) * step for i in range(len(levels))]
    for i, ((h, wd, f), hh) in enumerate(zip(levels, hs)):
        for xx in {enc_x[i], dec_x[i]}:
            ax.add_patch(Rectangle((xx, yc - hh / 2), w, hh, fc="white", ec=C, lw=0.7, zorder=2))
        if labels:
            text(ax, enc_x[i] + w / 2, yc - hh / 2 - 0.08, rf"${h}\times{wd}$" "\n" f"{f}",
                 FS_SMALL - 1.5, color=GREY, ha="center", va="top", linespacing=1.05)
        if i + 1 < len(levels):                        # pool down / upsample back
            arrow(ax, (enc_x[i] + w + 0.03, yc - hh / 2 + 0.05),
                  (enc_x[i + 1] - 0.03, yc - hs[i + 1] / 2 + 0.05), color=GREY, lw=0.5)
            arrow(ax, (dec_x[i + 1] + w + 0.03, yc - hs[i + 1] / 2 + 0.05),
                  (dec_x[i] - 0.03, yc - hh / 2 + 0.05), color=GREY, lw=0.5)
        if labels and 1 <= i < len(levels) - 1:        # additive skips (levels 2 and 3)
            y = yc + hh / 2 + 0.12
            ax.plot([enc_x[i] + w / 2, enc_x[i] + w / 2, dec_x[i] + w / 2, dec_x[i] + w / 2],
                    [yc + hh / 2, y, y, yc + hh / 2], color=C, lw=0.5, ls=(0, (2, 1.2)))
            text(ax, (enc_x[i] + dec_x[i]) / 2 + w / 2, y + 0.02, "+", FS_SMALL, color=C,
                 ha="center", va="bottom")
    return x0, dec_x[0] + w, hs[0]


def main() -> None:
    d = np.load(os.path.join(HERE, "data", "example_frame.npz"))
    walk, seen = d["walkable"], d["observed"]
    vmax = float(np.percentile(d["truth"][0][walk], 99))
    fig, ax = canvas(16.0, 4.6)
    yc = 2.3

    # ---- input: three frames of information-form observations + clock and coordinates ---
    cell = 0.05
    obs = field_rgb(d["observation"][0], walk, ps.CMAP["density"], vmax, hide=~seen)
    for k in (2, 1):                                   # t+1 and t-1 as cards behind frame t
        x, y = 0.3 + k * 0.16, yc - 0.9 + k * 0.16
        ax.add_patch(Rectangle((x, y), 12 * cell, 36 * cell, fc="white", ec=LINE, lw=0.5, zorder=1))
    thumbnail(ax, obs, 0.3, yc - 0.9, cell)
    text(ax, 0.9, yc + 1.3, r"$y/\sigma^2,\ 1/\sigma^2$" "\n" r"at $t-1,\,t,\,t+1$", FS_SMALL,
         ha="center", va="bottom", linespacing=1.1)
    text(ax, 0.9, yc - 1.05, "+ coordinates,\ntime of day, weekday", FS_SMALL - 0.5, color=GREY,
         ha="center", va="top", linespacing=1.1)
    text(ax, 0.9, yc - 1.6, "30 channels", FS_SMALL - 0.5, color=GREY, ha="center", va="top")

    # ---- U-Net, then the refinement U-Net ----------------------------------------------------
    u0, u1, uh = unet(ax, 2.55, yc)
    arrow(ax, (1.55, yc), (u0 - 0.1, yc))
    text(ax, (u0 + u1) / 2, yc + uh / 2 + 0.3, "U-Net", FS, ha="center", va="bottom")
    text(ax, (u0 + u1) / 2, yc + uh / 2 + 0.08, "mean pooling, additive skips", FS_SMALL - 0.5,
         color=GREY, ha="center", va="bottom")
    r0 = u1 + 0.9
    r0_, r1, rh = unet(ax, r0, yc, scale=0.45, labels=False)
    arrow(ax, (u1 + 0.1, yc), (r0 - 0.1, yc))
    text(ax, (r0 + r1) / 2, yc + rh / 2 + 0.2, "refinement\n(same U-Net)", FS_SMALL, ha="center",
         va="bottom", linespacing=1.1)
    text(ax, (r0 + r1) / 2, yc - rh / 2 - 0.15, "input: [$x$, first output]", FS_SMALL - 0.5,
         color=GREY, ha="center", va="top")
    ly = 0.2                                           # the input also feeds the refinement
    ax.plot([u0 - 0.45, u0 - 0.45, (r0 + r1) / 2], [yc, ly, ly], color=GREY, lw=0.5,
            ls=(0, (2, 1.2)))
    arrow(ax, ((r0 + r1) / 2, ly), ((r0 + r1) / 2, yc - rh / 2 - 0.5), color=GREY, lw=0.5)

    # ---- output: mean and sigma-hat ------------------------------------------------------------
    ox = r1 + 1.0
    arrow(ax, (r1 + 0.1, yc), (ox - 0.1, yc))
    e1 = thumbnail(ax, field_rgb(d["dincae_mean"][0], walk, ps.CMAP["density"], vmax), ox, yc - 0.9, cell)
    smax = float(np.percentile(d["dincae_density_sigma"][walk], 99))
    e2 = thumbnail(ax, field_rgb(d["dincae_density_sigma"], walk, ps.CMAP["spread"], smax),
                   e1[1] + 0.2, yc - 0.9, cell)
    text(ax, (e1[0] + e1[1]) / 2, e1[2] - 0.08, r"$\hat{x}_t$", 9, ha="center")
    text(ax, (e2[0] + e2[1]) / 2, e2[2] - 0.08, r"$\hat{\sigma}_t$", 9, ha="center")
    text(ax, (e1[0] + e2[1]) / 2, e1[3] + 0.15, "mean and\nuncertainty", FS_SMALL, ha="center",
         va="bottom", linespacing=1.1)
    block(ax, e2[1] + 0.45, yc - 0.45, 2.3, 0.9, "Gaussian NLL", "against the truth")
    arrow(ax, (e2[1] + 0.08, yc), (e2[1] + 0.4, yc), color=GREY)
    text(ax, e2[1] + 1.6, yc - 0.6, "training loss", FS_SMALL - 0.5, color=GREY, ha="center",
         va="top")

    save(fig, os.path.join(HERE, "out", "dincae"))


if __name__ == "__main__":
    main()
