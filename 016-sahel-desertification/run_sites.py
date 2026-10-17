#!/usr/bin/env python3
#-------- run_sites.py -------------------------------------------------------------
#-----------------------------------------------------------------------------
# Drive the lattice with each site's real rainfall, 1950-2025, and compare with what
# MODIS saw from 2000.
#
# Rainfall to environmental quality. The model's b is "how good the year is for a
# seedling". It is taken proportional to the rainy-season (June-September) total,
# smoothed over three years because shrubs and soil water carry one year into the next:
#     b(t) = B_WET * P3(t) / P3_ref,     P3_ref = the 1950-1969 mean (the wet decades)
# so the wet 1950s and 60s sit at b = B_WET = 0.75, comfortably vegetated. That one
# number is the only calibration; it is set before looking at NDVI.
#
# Land use. The two sites had similar rainfall and opposite fates. The model's only
# other lever with a human meaning is d, the rate at which bare fertile soil degrades
# (crusting, erosion, trampling). Each site is run with the default d = 0.2 and with a
# heavier-use d = 0.3, and its MODIS record decides which fits.
#
# One model year = one calendar year. Each run is the average of 8 lattices of 100 x 100.
#
#   python run_sites.py
#
# Output: results/sites.csv (model cover by year, site and d), results/ndvi_annual.csv

import os

import numpy as np
import pandas as pd

from sahel_ca import Landscape

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
B_WET = 0.75
DS = [0.2, 0.3]
SEEDS = 8
SITES = ["gourma", "fakara"]


def rainy_season(site):
    r = pd.read_csv(os.path.join(DATA, f"rain_{site}.csv"), index_col=0, parse_dates=True).squeeze()
    jjas = r[r.index.month.isin([6, 7, 8, 9])]
    return jjas.groupby(jjas.index.year).sum()


def b_series(site):
    P = rainy_season(site)
    P3 = P.rolling(3, min_periods=1).mean()
    return B_WET * P3 / P3.loc[1950:1969].mean(), P


def ndvi_annual(site):
    """The growing-season peak: the mean of the three highest 16-day NDVI values each year."""
    n = pd.read_csv(os.path.join(DATA, f"ndvi_{site}.csv"), parse_dates=["date"]).dropna()
    n = n[n.date.dt.year < 2026]
    return n.groupby(n.date.dt.year).ndvi.apply(lambda s: s.nlargest(3).mean())


def run(b, d, seed):
    land = Landscape(n=100, b=b.iloc[0], cover=0.6, seed=seed, d=d, f=0.9)
    land.step(100)                                         # settle in the 1950 climate
    out = []
    for year, bt in b.items():
        land.b = bt
        if land.cover() == 0:
            land.grid[land.rng.random(land.grid.shape) < 0.01] = 1     # seed rain, as in run_hysteresis.py
        land.step(1)
        out.append(land.cover())
    return np.array(out)


def main():
    rows, nd = [], []
    for site in SITES:
        b, P = b_series(site)
        for d in DS:
            covers = np.mean([run(b, d, s) for s in range(SEEDS)], axis=0)
            for year, bt, p, c in zip(b.index, b.values, P.values, covers):
                rows.append(dict(site=site, d=d, year=year, rain_mm=p, b=bt, cover=c))
            print(f"{site}, d = {d}: cover 1955 {covers[5]:.2f}, 1985 {covers[35]:.2f}, 2020 {covers[70]:.2f}")
        a = ndvi_annual(site)
        nd += [dict(site=site, year=y, ndvi_peak=v) for y, v in a.items()]
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "results", "sites.csv"), index=False)
    pd.DataFrame(nd).to_csv(os.path.join(HERE, "results", "ndvi_annual.csv"), index=False)


if __name__ == "__main__":
    main()
