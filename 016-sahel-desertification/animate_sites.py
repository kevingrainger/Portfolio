#!/usr/bin/env python3
#-------- animate_sites.py ----------------------------------------------------------
#-----------------------------------------------------------------------------
# The lattice played year by year beside what satellites saw, at both Sahel sites.
# Each site is a strip of landscape (60 x 240 cells), the shape of a transect across a
# real site rather than a square, driven by its own CHIRPS rainy-season rain (ERA5 scaled
# before 1981, as in run_sites.py). Beside each strip: the Sentinel-2 vegetation pattern
# of the real site, and below, the model's cover against MODIS peak greenness each year.
#
#   python animate_sites.py
#
# Output: figures/fig7_sites_animation.gif, figures/fig7_sites_final.png

import io
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from numba import njit
from PIL import Image

from run_sites import b_series, ndvi_annual
from sahel_ca import DEFAULTS

HERE = os.path.dirname(os.path.abspath(__file__))
NY, NX = 60, 240
NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"
CMAP = ListedColormap(["#b08850", "#f2e6cf", "#2e7d32"])        # degraded, empty, vegetated


@njit(cache=True)
def sweep_rect(grid, b, c, m, d, r, f, delta, dt, order):
    """The Kefi update of sahel_ca._sweep on a rectangular, periodic lattice."""
    ny, nx = grid.shape
    N = ny * nx
    nveg = 0
    for x in range(N):
        if grid.flat[x] == 1:
            nveg += 1
    rho = nveg / N
    for idx in order:
        i, j = idx // nx, idx % nx
        q = 0.25 * ((grid[(i + 1) % ny, j] == 1) + (grid[(i - 1) % ny, j] == 1)
                    + (grid[i, (j + 1) % nx] == 1) + (grid[i, (j - 1) % nx] == 1))
        s = grid[i, j]
        u = np.random.random()
        if s == 0:
            grow = max((delta * rho + (1 - delta) * q) * (b - c * rho), 0.0) * dt
            if u < grow:
                grid[i, j] = 1
                nveg += 1
            elif u < grow + d * dt:
                grid[i, j] = -1
        elif s == 1:
            if u < m * dt:
                grid[i, j] = 0
                nveg -= 1
        else:
            if u < (r + f * q) * dt:
                grid[i, j] = 0
        rho = nveg / N
    return grid


@njit(cache=True)
def seed_numba(s):
    np.random.seed(s)


def simulate(site, d=0.2, seed=0, dt=0.2):
    b, _ = b_series(site, "chirps")
    p = {**DEFAULTS, "d": d, "f": 0.9}
    rng = np.random.default_rng(seed); seed_numba(seed)
    u = rng.random((NY, NX))
    grid = np.where(u < 0.6, 1, np.where(u < 0.8, 0, -1)).astype(np.int8)
    def years(bt, n):
        for _ in range(int(round(n / dt))):
            sweep_rect(grid, bt, p["c"], p["m"], p["d"], p["r"], p["f"], p["delta"], dt, rng.permutation(NY * NX))
    years(b.iloc[0], 100)
    frames, cover = {}, {}
    for year, bt in b.items():
        if (grid == 1).sum() == 0:
            grid[rng.random(grid.shape) < 0.01] = 1
        years(bt, 1)
        cover[year] = float((grid == 1).mean())
        if year >= 1975:
            frames[year] = grid.copy()
    return frames, pd.Series(cover), b


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    sims = {s: simulate(s, d=0.2, seed=1) for s in ["gourma", "fakara"]}
    nd = {s: ndvi_annual(s) for s in sims}
    s2 = {s: np.load(os.path.join(HERE, "data", f"s2_{s}.npz"))["ndvi"] for s in sims}
    years = sorted(sims["gourma"][0])
    images = []
    for y in years + [years[-1]] * 6:
        fig = plt.figure(figsize=(12, 7.2))
        gs = fig.add_gridspec(3, 2, width_ratios=[2.4, 1], height_ratios=[1, 1, 1.25], hspace=0.45, wspace=0.08)
        for row, site, name in [(0, "gourma", "Gourma, Mali"), (1, "fakara", "Fakara, Niger")]:
            ax = fig.add_subplot(gs[row, 0])
            ax.imshow(sims[site][0][y], cmap=CMAP, vmin=-1, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)
            ax.set_title(f"{name}: model, {y}", loc="left")
            ax2 = fig.add_subplot(gs[row, 1])
            ax2.imshow(s2[site], cmap="YlGn", vmin=0.08, vmax=0.32)
            ax2.set_xticks([]); ax2.set_yticks([]); ax2.spines[:].set_visible(False)
            ax2.set_title("Sentinel-2 NDVI", loc="left", fontsize=9)
        ax = fig.add_subplot(gs[2, :])
        ax2 = ax.twinx()
        for site, col in [("gourma", NAVY), ("fakara", RED)]:
            c = sims[site][1]
            ax.plot(c.index, c.values, color=col, lw=1.6, label=f"{site.capitalize()} model cover")
            ax2.plot(nd[site].index, nd[site].values, color=col, lw=1.2, ls="--", marker="o", ms=3,
                     label=f"{site.capitalize()} MODIS peak NDVI")
        ax.axvline(y, color=GREY, lw=1)
        ax.set_xlim(1975, 2025.5); ax.set_ylabel("model vegetation cover"); ax2.set_ylabel("MODIS peak NDVI")
        ax2.spines["right"].set_visible(True)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, ncol=2, loc="upper left")
        ax.set_title("Model cover and satellite greenness", loc="left")
        buf = io.BytesIO(); fig.savefig(buf, dpi=80, bbox_inches="tight"); plt.close(fig)
        images.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
        if y == years[-1] and len(images) == len(years):
            fig_last = buf
    images[0].save(os.path.join(HERE, "figures", "fig7_sites_animation.gif"), save_all=True,
                   append_images=images[1:], duration=350, loop=0, optimize=True)
    Image.open(fig_last).save(os.path.join(HERE, "figures", "fig7_sites_final.png"))
    for s in sims:
        c = sims[s][1]
        print(s, "cover 1980", round(c[1980], 2), "1990", round(c[1990], 2), "2025", round(c[2025], 2),
              "corr with MODIS", round(np.corrcoef(c.reindex(nd[s].index).dropna(), nd[s].reindex(c.index).dropna())[0, 1], 2))


if __name__ == "__main__":
    main()
