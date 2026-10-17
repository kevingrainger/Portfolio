#!/usr/bin/env python3
#-------- spacing_threshold.py -----------------------------------------------------
#-----------------------------------------------------------------------------
# How far apart must olive trees be planted for an outbreak to stay small? (Plan items
# X3-X5: the spacing threshold.)
#
# The grove is kept at 16 x 16 trees; only the gap between trees changes, from 4 cells
# (canopies 1 cell apart) to 10. Two versions:
#   per hectare   spittlebug density per hectare fixed, so a wider grove holds more of
#                 them per tree (they live in the ground cover between trees)
#   per tree      density scaled down as the grove spreads, so each tree has the same
#                 number of spittlebugs as at the default spacing of 5
# For each spacing, 20 seeds run three years from one infected tree. Recorded: the share of
# trees ever infected. If spread is a percolation process, the outbreak should fall from
# grove-wide to local over a narrow range of spacing: the critical planting distance.
#
#   python spacing_threshold.py          # about 15 minutes on 8 cores
#
# Output: results/spacing.csv, figures/fig_spacing_threshold

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from olive_model import run_ensemble          # noqa: E402

SPACINGS = [4, 5, 6, 7, 8, 9, 10]
SEEDS = range(20)
DAYS = 1095


def main():
    rows = []
    for version in ["per hectare", "per tree"]:
      for s in SPACINGS:
        size = 17 * s + 3                         # 16 x 16 trees at this spacing
        density = 1_000_000 if version == "per hectare" else 1_000_000 * (88 / size) ** 2
        res = run_ensemble(DAYS, SEEDS, width=size, height=size, tree_spacing=s, vectors_per_hectare=density)
        for r in res:
            rows.append(dict(version=version, spacing=s, seed=r["seed"], ever_infected=r["ever_infected"],
                             n_trees=len(r["tree_centres"])))
        f = np.array([r["ever_infected"] for r in res])
        print(f"{version}, spacing {s}: {len(res[0]['tree_centres'])} trees, ever infected mean {f.mean():.2f}, "
              f"P(> half the grove) {np.mean(f > 0.5):.2f}", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    df.to_csv(os.path.join(HERE, "results", "spacing.csv"), index=False)
    plot(df)


def plot(df):
    import matplotlib.pyplot as plt
    import figstyle
    figstyle.apply()
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    rng = np.random.default_rng(0)
    for version, c in [("per hectare", "#117733"), ("per tree", "#882255")]:
        d = df[df.version == version]
        ax.scatter(d.spacing + rng.uniform(-0.15, 0.15, len(d)), d.ever_infected, s=12, color=c, alpha=0.35, lw=0)
        g = d.groupby("spacing").ever_infected.mean()
        ax.plot(g.index, g.values, "o-", color=c, lw=2, label=f"spittlebugs fixed {version} (mean of 20 groves)")
    ax.set_xlabel("tree spacing, cells (canopy is 3 cells wide)")
    ax.set_ylabel("share of trees ever infected, 3 years")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="lower left")
    ax.set_title("Planting further apart: where the outbreak stops spreading", loc="left")
    figstyle.save(fig, os.path.join(HERE, "figures", "fig_spacing_threshold"))


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot(pd.read_csv(os.path.join(HERE, "results", "spacing.csv")))
    else:
        main()
