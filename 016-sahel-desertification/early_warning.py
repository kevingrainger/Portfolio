#!/usr/bin/env python3
#-------- early_warning.py ---------------------------------------------------------
#-----------------------------------------------------------------------------
# Can you see the collapse coming from above? Dry a large lattice (200 x 200, strong
# facilitation f = 0.9) slowly and record, at each step, what a satellite could measure:
#   - the patch-size distribution: Kéfi et al. (2007) found it follows a power law in
#     healthy drylands and loses its large patches near collapse
#   - the share of all vegetation held in the largest patch, and whether that patch
#     still spans the landscape (percolation)
#   - cover alone, the measure a coarse satellite gives
#
#   python early_warning.py
#
# Output: results/early_warning.csv, results/patches.npz (patch sizes and lattices at
# selected b)

import os

import numpy as np
import pandas as pd

from sahel_ca import Landscape, patch_sizes, percolates

HERE = os.path.dirname(os.path.abspath(__file__))
B_GRID = np.round(np.arange(0.80, 0.30, -0.02), 3)
KEEP = [0.80, 0.60, 0.50, 0.42, 0.36]
SETTLE, SAMPLES = 150, 5


def main():
    land = Landscape(n=200, b=B_GRID[0], cover=0.6, seed=3, f=0.9)
    rows, keep = [], {}
    for b in B_GRID:
        land.b = b
        land.step(SETTLE)
        sizes_all = []
        for _ in range(SAMPLES):                                   # a few snapshots, 10 years apart
            land.step(10)
            s = patch_sizes(land.grid)
            sizes_all.append(s)
            rows.append(dict(b=b, cover=land.cover(), largest_share=s.max() / s.sum() if len(s) else 0,
                             n_patches=len(s), spans=percolates(land.grid)))
        if np.isclose(b, KEEP).any():
            keep[f"sizes_{b:.2f}"] = np.concatenate(sizes_all)
            keep[f"grid_{b:.2f}"] = land.grid.copy()
        print(f"b = {b:.2f}: cover {land.cover():.2f}")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    pd.DataFrame(rows).groupby("b").mean().reset_index().to_csv(os.path.join(HERE, "results", "early_warning.csv"), index=False)
    np.savez_compressed(os.path.join(HERE, "results", "patches.npz"), **keep)


if __name__ == "__main__":
    main()
