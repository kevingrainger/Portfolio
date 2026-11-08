#!/usr/bin/env python3
#-------- clumping.py ---------------------------------------------------------------
#-----------------------------------------------------------------------------
# How clumped is the vegetation, in the satellite pictures and in the lattice?
#
# Clumping is measured as Kefi et al. do: C = q(+|+) / rho, the chance that a neighbour of
# a vegetated cell is vegetated, over the share of cells vegetated. C = 1 is a random
# scatter; above 1, vegetation grows in patches.
#
#   satellite   each October-November picture (fetch_site_pictures.py) at 30 m (Sentinel-2
#               averaged over 3 x 3 pixels), with the greenest pixels taken as vegetated up
#               to the same share as the lattice's cover that year, so both are compared at
#               the same density
#   lattice     the animation's lattice (animate_landsat.py) at each site and year, for
#               several settings of the two things that scatter it: the share of seeds that
#               travel far (delta) and the seed rain from outside
#   fit         for each setting, the joint 1982-2021 test across the Sahel (joint_fit_1982.py
#               metrics) is rerun at the fitted rain scale and degradation rate, so a setting
#               that clumps right but fits worse is visible
#
#   python clumping.py
#
# Output: results/clumping.json, figures/fig16_clumping.png

import json
import os
from itertools import product
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"
DELTAS = [0.1, 0.03, 0.01]
SEEDRAINS = [0.03, 0.01, 0.003]
SITES = ["gourma", "fakara"]


def clump(mask):
    rho = mask.mean()
    if rho == 0 or rho == 1:
        return np.nan
    nb = (np.roll(mask, 1, 0).astype(int) + np.roll(mask, -1, 0) + np.roll(mask, 1, 1) + np.roll(mask, -1, 1))
    q = (nb[mask] / 4).mean()
    return float(q / rho)


def to_grid(a, n):
    """Resample a picture (pixels x pixels x bands) to the n x n lattice grid by averaging."""
    if a.shape[0] == n:
        return a
    k = a.shape[0] / n
    if float(k).is_integer():
        k = int(k)
        return a.reshape(n, k, n, k, a.shape[2]).mean(axis=(1, 3))
    from scipy.ndimage import zoom
    return zoom(np.nan_to_num(a, nan=np.nanmedian(a)), (n / a.shape[0], n / a.shape[1], 1), order=1)


def satellite_clumping(site, cover_by_year, n=200):
    d = np.load(os.path.join(HERE, "data", f"pictures_{site}.npz"))
    out = {}
    for y, sen in zip(d["years"], d["sensors"]):
        a = to_grid(d[f"rgb_{y}"].astype(float), n)             # to the lattice's 30 m cells
        nd = (a[..., 3] - a[..., 0]) / (a[..., 3] + a[..., 0] + 1e-6)
        rho = cover_by_year.get(int(y), cover_by_year[max(k for k in cover_by_year if k <= max(int(y), min(cover_by_year)))])
        thr = np.nanquantile(nd, 1 - rho)
        out[int(y)] = clump(np.nan_to_num(nd, nan=-1) > thr)
    return out


def lattice_run(args):
    site, delta, sr = args
    import animate_landsat as al
    from sahel_ca import Landscape, VEG, EMPTY
    p3 = al.site_rain(site)
    land = Landscape(n=200, b=float(min(1, al.B0 * p3.loc[1981:1983].mean() / 400)), cover=0.5, seed=7, d=al.D0, f=0.9, delta=delta)
    def seed():
        e = (land.grid == EMPTY) & (land.rng.random(land.grid.shape) < sr); land.grid[e] = VEG
    for _ in range(50):
        land.step(1); seed()
    cover, C = {}, {}
    for y, p in p3.items():
        land.b = float(min(1, al.B0 * p / 400)); land.step(1); seed()
        if y >= 1985:
            cover[int(y)] = land.cover(); C[int(y)] = clump(land.grid == VEG)
    return site, delta, sr, cover, C


def main():
    jobs = list(product(SITES, DELTAS, SEEDRAINS))
    with Pool(max(1, os.cpu_count() - 1)) as pool:
        runs = pool.map(lattice_run, jobs)
    rows = []
    sat = {}
    for site, delta, sr, cover, C in runs:
        s = satellite_clumping(site, cover)
        sat[(site, delta, sr)] = s
        yrs = [y for y in s if y in C]
        rows.append(dict(site=site, delta=delta, seed_rain=sr, lattice_C=float(np.mean([C[y] for y in yrs])),
                         satellite_C=float(np.nanmean([s[y] for y in yrs])), cover_mean=float(np.mean(list(cover.values()))),
                         cover_1985=cover[1985], cover_2024=cover[2024]))
    g = pd.DataFrame(rows)
    print(g.round(3).to_string(index=False))

    # the Sahel-wide fit for each setting, at the fitted rain scale and degradation rate
    import joint_fit_1982 as jf
    t = pd.read_csv(os.path.join(HERE, "results", "covariates_sites.csv"))
    r = pd.read_csv(os.path.join(HERE, "data", "sahel_chirps_jjas.csv")).pivot(index="year", columns="id", values="rain_jjas")
    obs = pd.read_csv(os.path.join(HERE, "data", "sahel_avhrr_dry.csv")).pivot(index="year", columns="id", values="avhrr_dry")
    t = t[t.id.isin(r.columns) & t.id.isin(obs.columns)].copy()
    p3 = r.rolling(3, min_periods=1).mean()
    band = ((t.lon + 16) // 4).astype(int)
    test = list(t.id[band % 2 == 1])
    fits = {}
    import sahel_ca
    for delta, sr in product(DELTAS, SEEDRAINS):
        sahel_ca.DEFAULTS["delta"] = delta                  # picked up by the workers through Landscape's defaults
        jobs = [(i, (0.45 * p3[i].loc[jf.YEARS] / 400).values, 0.3, sr, 1000 + int(i) * 10) for i in t.id]
        with Pool(max(1, os.cpu_count() - 1), initializer=_set_delta, initargs=(delta,)) as pool:
            cov = pd.DataFrame(dict(pool.map(jf.run_site, jobs)), index=jf.YEARS)
        fits[(delta, sr)] = jf.metrics(cov, obs, test)
        print(delta, sr, {k: round(v, 2) for k, v in fits[(delta, sr)].items()}, flush=True)
    g["test_trend"] = [fits[(r_.delta, r_.seed_rain)]["trend"] for r_ in g.itertuples()]
    g["test_year_to_year"] = [fits[(r_.delta, r_.seed_rain)]["year_to_year"] for r_ in g.itertuples()]
    g["test_level"] = [fits[(r_.delta, r_.seed_rain)]["level"] for r_ in g.itertuples()]
    json.dump(dict(table=g.to_dict(orient="records")), open(os.path.join(HERE, "results", "clumping.json"), "w"), indent=1)
    g.to_csv(os.path.join(HERE, "results", "clumping.csv"), index=False)
    print(g.round(3).to_string(index=False))


def _set_delta(delta):
    import sahel_ca
    sahel_ca.DEFAULTS["delta"] = delta


if __name__ == "__main__":
    main()
