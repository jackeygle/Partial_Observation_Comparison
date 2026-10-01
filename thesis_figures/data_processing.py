"""Thesis figure: from the ATC tracking records to the gridded state.

    python3 -m thesis_figures.data_processing   # -> thesis_figures/out/data_processing.{pdf,png}

The steps of crowdcore/data/csv_to_h5.py and h5_to_grid.py: a record (time, person, x, y in
mm, speed in mm/s, direction of motion) -> metres and a velocity vector -> shifted and
rotated into the 36 x 12 m corridor rectangle -> all records of one second spread over the
1 m cells with a tent kernel -> one 4 x 36 x 12 state per second. The person identifier,
the height and the facing direction are not used.
"""
from __future__ import annotations

import os

from thesis_figures.common import FS_SMALL, GREY, arrow, block, canvas, save, text

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    fig, ax = canvas(16.0, 2.75)
    bw, bh, gap, yc = 2.8, 1.3, 0.35, 1.8
    steps = [("tracking records", "one person at one instant:\nposition, speed, direction"),
             ("units and velocity", "millimetres to metres; speed\nand direction to a vector"),
             ("corridor frame", "shift and rotate into the\n" r"$36 \times 12$ m rectangle"),
             ("one frame per second", "all records of the second,\ntent kernel on 1 m cells"),
             (r"state $x_t$", r"$4 \times 36 \times 12$:" "\n" r"$\rho,\ v_x,\ v_y,\ \nu$")]
    x = 0.15
    for i, (name, sub) in enumerate(steps):
        b = block(ax, x, yc - bh / 2, bw, bh, name, sub, size=FS_SMALL + 0.5)
        if i:
            arrow(ax, (x - gap + 0.04, yc), (x - 0.04, yc))
        x = b[2] + gap
    text(ax, 8.0, 0.75, "46 Sundays, split by date: 32 training, 7 validation, 7 test days."
         "   Walkable mask: 290 of the 432 cells,\nfrom the obstacle map and the training days;"
         " the same for every day and every method.", FS_SMALL, color=GREY, ha="center",
         va="top", linespacing=1.25)
    save(fig, os.path.join(HERE, "out", "data_processing"))


if __name__ == "__main__":
    main()
