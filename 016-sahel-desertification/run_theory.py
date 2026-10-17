#!/usr/bin/env python3
#-------- run_theory.py ------------------------------------------------------------
#-----------------------------------------------------------------------------
# The mean-field and pair approximations swept through the same drying and wetting as
# the lattice in run_hysteresis.py (strong facilitation, f = 0.9).
#
#   python run_theory.py
#
# Output: results/theory.csv

import os

import numpy as np
import pandas as pd

from approximations import sweep
from run_hysteresis import B_GRID

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    rows = sweep("mean field", B_GRID) + sweep("pair", B_GRID)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "results", "theory.csv"), index=False)
    for m, g in df.groupby("model"):
        dr, we = g[g.direction == "drying"], g[g.direction == "wetting"]
        print(f"{m}: collapses below b = {dr[dr.cover > 0.02].b.min()}, regrows from b = {we[we.cover > 0.02].b.min()}")


if __name__ == "__main__":
    main()
