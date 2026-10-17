#!/usr/bin/env python3
#-------- run_hysteresis.py --------------------------------------------------------
#-----------------------------------------------------------------------------
# Does the landscape tip, and does it come back? Lower the environmental quality b
# (think: rainfall) slowly from wet to dry, then raise it again, letting the lattice
# settle at each step. With strong facilitation the vegetation should collapse abruptly
# and need a much wetter climate to return (hysteresis); without it, it should not.
#
#   python run_hysteresis.py
#
# Output: results/hysteresis.csv, figures/fig1_hysteresis.png

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import figstyle                                  # noqa: E402
from sahel_ca import Landscape, percolates       # noqa: E402

B_GRID = np.round(np.arange(1.0, 0.0, -0.025), 3)
SETTLE = 150                                     # years per step (the slowest rate, r + f q, needs time)
FS = [0.2, 0.5, 0.9]
COLOURS = {0.2: "0.45", 0.5: "#009E73", 0.9: "#882255"}


def sweep(f, n=80, seed=0):
    land = Landscape(n=n, b=B_GRID[0], cover=0.6, seed=seed, f=f)
    rows = []
    for direction, bs in [("drying", B_GRID), ("wetting", B_GRID[::-1])]:
        for b in bs:
            land.b = b
            if direction == "wetting" and land.cover() == 0:
                # a bare landscape has no seeds left and cannot regrow however wet it gets
                # (the desert is an absorbing state): scatter a few plants on fertile or
                # degraded ground, as wind and animals would bring them
                land.grid[land.rng.random(land.grid.shape) < 0.01] = 1
            land.step(SETTLE)
            rows.append(dict(f=f, direction=direction, b=b, cover=land.cover(),
                             degraded=land.degraded(), spans=percolates(land.grid)))
    return rows


def main():
    rows = []
    for f in FS:
        rows += sweep(f)
        print(f"f = {f} done")
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    df.to_csv(os.path.join(HERE, "results", "hysteresis.csv"), index=False)
    plot(df)


def plot(df):
    import matplotlib.pyplot as plt
    figstyle.apply()
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for f in FS:
        for direction, ls in [("drying", "-"), ("wetting", "--")]:
            s = df[(df.f == f) & (df.direction == direction)].sort_values("b")
            ax.plot(s.b, s.cover, ls, color=COLOURS[f], lw=1.8)
        ax.plot([], [], "-", color=COLOURS[f], lw=4, label=f"facilitation f = {f}")
    ax.axhline(0.5927, color="0.6", lw=0.8, ls=":")
    ax.annotate("site-percolation threshold, 0.593", (0.02, 0.5927), xytext=(0, 4), textcoords="offset points",
                fontsize=8, color="0.4")
    ax.plot([], [], "-", color="0.3", label="drying (b falling)")
    ax.plot([], [], "--", color="0.3", label="wetting (b rising)")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.33))
    ax.set_xlabel("environmental quality b (rainfall)")
    ax.set_ylabel("vegetation cover")
    ax.set_title("Plants that help plants tip, and do not come back on the same path", loc="left")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.02, 0.85)
    figstyle.save(fig, os.path.join(HERE, "figures", "fig1_hysteresis"))


if __name__ == "__main__":
    main()
