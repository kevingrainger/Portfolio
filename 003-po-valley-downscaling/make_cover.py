#!/usr/bin/env python3
#-------- make_cover.py ----------------------------------------------------------------
# The 003 cover: the hottest July 2022 day, downscaled Tmax draped over the terrain.
# Kept apart from the notebook so re-running it never overwrites the cover.
#
#   python make_cover.py

import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import figstyle
from tmax1km import plots
from tmax1km.data import Dataset

if __name__ == "__main__":
    figstyle.apply()
    ds = Dataset(verbose=False)
    fig = plots.cover(ds, np.load(os.path.join(HERE, "results", "maps.npz")))
    if ds.placeholder:
        figstyle.watermark(fig)
    fig.savefig(os.path.join(HERE, "figures", "cover.png"), dpi=200, bbox_inches=fig.bbox_inches)   # exactly 2:1, no cropping
    print("wrote figures/cover.png")
