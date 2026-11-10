#!/usr/bin/env python3
#-------- make_extra_figures.py ------------------------------------------------------
# Three figures for the website timeline, each from a few model runs:
#   fig12_coupled_pair     one multilevel Monte Carlo pair: a coarse run (3,240 spittlebugs)
#                          and the fine run built from it (6,480), against a fine run that
#                          only shares the seed
#   fig13_season_occupancy share of spittlebugs on the trees through two years
#   fig14_spacing_groves   a grove at 4 m and at 8 m spacing after three years, spittlebugs
#                          per tree held fixed
#
#   python make_extra_figures.py        # a few minutes on one core

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import figstyle                                # noqa: E402
from olive_model import OliveGrove, _shade_seasons   # noqa: E402
from mlmc import CoupledGrove                  # noqa: E402

figstyle.apply()
OUT = os.path.join(HERE, "figures")
NAVY, ORANGE, GREY = "#0b2545", "#d55e00", "#9a9a9a"


def track(grove, days):
    """Run day by day, returning the share of trees ever infected each day."""
    grove.run(0, initial_infected=1)
    out = []
    for _ in range(days):
        grove.run(1)
        out.append(grove.ever_infected())
    return np.array(out)


def coupled_pair(seed=3, days=365):
    n_coarse = 3240
    coarse = CoupledGrove(seed=seed, sampling_rate=n_coarse / 810000)
    fine = CoupledGrove(seed=seed, sampling_rate=2 * n_coarse / 810000, n_shared=n_coarse)
    plain = OliveGrove(seed=seed, sampling_rate=2 * n_coarse / 810000)
    curves = {name: track(g, days) for name, g in [("coarse", coarse), ("fine", fine), ("plain", plain)]}

    fig = plt.figure(figsize=(13, 4.6))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.25)
    for k, (g, title) in enumerate([(coarse, "Coarse run: 3,240 spittlebugs"), (fine, "Fine run: the same 3,240 plus 3,240 more")]):
        ax = fig.add_subplot(grid[k])
        ax.imshow(g.grove_image(), origin="upper")
        shared = np.arange(g.n_vectors) < n_coarse
        ax.scatter(g.vector_x[shared], g.vector_y[shared], s=2, color=NAVY, label="shared spittlebugs")
        if k == 1:
            ax.scatter(g.vector_x[~shared], g.vector_y[~shared], s=2, color=ORANGE, label="extra spittlebugs")
        ax.set_title(f"{title}\nday {days}, {g.ever_infected():.0%} of trees infected", loc="left", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.legend(loc="lower left", fontsize=8, markerscale=4, framealpha=0.9)
    ax = fig.add_subplot(grid[2])
    d = np.arange(1, days + 1)
    ax.plot(d, curves["coarse"], color=NAVY, lw=2, label="coarse run")
    ax.plot(d, curves["fine"], color=ORANGE, lw=2, label="fine run built from it")
    ax.plot(d, curves["plain"], color=GREY, lw=1.6, ls="--", label="fine run sharing only the seed")
    _shade_seasons(ax, days)
    ax.set_xlabel("day"); ax.set_ylabel("share of trees ever infected")
    ax.set_title("Share of trees infected over a year", loc="left")
    ax.legend(loc="upper left", fontsize=8.5)
    figstyle.save(fig, os.path.join(OUT, "fig12_coupled_pair"))


def season_occupancy(seed=1, days=730):
    g = OliveGrove(seed=seed, sampling_rate=0.001)
    g.run(0, initial_infected=1)
    on = []
    for _ in range(days):
        g.run(1)
        on.append(np.mean(g.tree_of_cell[g.vector_y, g.vector_x] >= 0))
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.plot(np.arange(1, days + 1), on, color=NAVY, lw=1.8)
    _shade_seasons(ax, days)
    ax.set_ylim(0, 1)
    ax.set_xlabel("day (green: tender sprouts, days 90 to 185 of each year)")
    ax.set_ylabel("share on the trees")
    ax.set_title("Share of spittlebugs on the trees through two years", loc="left")
    figstyle.save(fig, os.path.join(OUT, "fig13_season_occupancy"))


def spacing_groves(seed=2, days=1095):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
    for ax, s in zip(axes, [4, 8]):
        size = 17 * s + 3
        g = OliveGrove(seed=seed, width=size, height=size, tree_spacing=s, vectors_per_hectare=1_000_000 * (88 / size) ** 2)
        g.run(days, initial_infected=1)
        ax.imshow(g.grove_image(), origin="upper")
        ax.set_title(f"Trees {s} m apart: {g.ever_infected():.0%} ever infected after three years", loc="left", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    figstyle.save(fig, os.path.join(OUT, "fig14_spacing_groves"))


if __name__ == "__main__":
    coupled_pair()
    season_occupancy()
    spacing_groves()
    print("wrote fig12, fig13, fig14")
