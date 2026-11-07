#!/usr/bin/env python3
#-------- calibrate_sites.py --------------------------------------------------------
#-----------------------------------------------------------------------------
# Fitting the lattice to the satellite record.
#
# With Kefi's default rates and a rain-to-b scale set by hand, the lattice tracked MODIS
# poorly (long_term.py). Two reasons, and a fix for each:
#
#   1. What the satellite sees. Sahel greenness in August-September is mostly annual
#      grass, which follows that year's rain, on top of shrubs and trees, which change
#      slowly. The lattice is the slow part. So the observation model is
#          NDVI(t) = a + g * rain anomaly(t) + w * cover(t),
#      with a, g, w fitted by least squares for each candidate run (g: the grass, w: the
#      woody cover's weight).
#   2. The rates, and land use that changes. Fitted per site by grid search: the scale
#      that turns smoothed rainy-season rain into the lattice's b (B_WET, b in the wet
#      1950s-60s), and d, the rate at which bare soil degrades, which is where land use
#      enters. d is constant to 1985 and then changes linearly to a second value by 2025
#      (more cropping and grazing raise it, fallow and protection lower it). The cover
#      weight w must come out positive: more vegetation can't mean less green.
#
# Fitted on 2001-2015 and tested on 2016-2025, against rain alone (NDVI = a + g * rain),
# with CHIRPS rain. 6 lattices of 100 x 100 per candidate.
#
#   python calibrate_sites.py
#
# Output: results/calibration.json, results/calibration_grid.csv, figures/fig11_calibration.png

import json
import os
from itertools import product
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_sites import b_series, ndvi_annual, run, SITES

HERE = os.path.dirname(os.path.abspath(__file__))
NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"
B_WETS = [0.6, 0.7, 0.8, 0.9]
D0S = [0.1, 0.2, 0.3]                       # degradation rate up to 1985
D1S = [0.05, 0.1, 0.2, 0.3, 0.45]           # ... changing linearly to this by 2025
SEEDS = 6
FIT, TEST = (2001, 2015), (2016, 2025)


def run_tv(b, d_of_year, seed):
    from sahel_ca import Landscape
    land = Landscape(n=100, b=b.iloc[0], cover=0.6, seed=seed, d=d_of_year[b.index[0]], f=0.9)
    land.step(100)
    out = []
    for year, bt in b.items():
        land.b = bt; land.p["d"] = d_of_year[year]
        if land.cover() == 0:
            land.grid[land.rng.random(land.grid.shape) < 0.01] = 1
        land.step(1)
        out.append(land.cover())
    return np.array(out)


def cover_series(args):
    site, bw, d0, d1 = args
    b, P = b_series(site, "chirps")
    b = b * bw / 0.75
    yrs = b.index.values
    dy = dict(zip(yrs, np.where(yrs <= 1985, d0, d0 + (d1 - d0) * (yrs - 1985) / 40)))
    c = np.mean([run_tv(b, dy, s) for s in range(SEEDS)], axis=0)
    return site, bw, (d0, d1), pd.Series(c, index=b.index)


def fit_obs(y, X, fit_idx):
    A = np.column_stack([np.ones(len(X))] + [X[:, k] for k in range(X.shape[1])])
    w, *_ = np.linalg.lstsq(A[fit_idx], y[fit_idx], rcond=None)
    return A @ w, w


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    jobs = list(product(SITES, B_WETS, D0S, D1S))
    with Pool(max(1, os.cpu_count() - 1)) as pool:
        runs = pool.map(cover_series, jobs)
    out, rows, best_series = {}, [], {}
    for site in SITES:
        nd = ndvi_annual(site).loc[FIT[0]:TEST[1]]
        _, P = b_series(site, "chirps")
        rain = P.reindex(nd.index); rain = (rain - rain.loc[FIT[0]:FIT[1]].mean()) / rain.loc[FIT[0]:FIT[1]].std()
        y = nd.values; yrs = nd.index.values
        fi, te = (yrs >= FIT[0]) & (yrs <= FIT[1]), (yrs >= TEST[0]) & (yrs <= TEST[1])
        p_rain, _ = fit_obs(y, rain.values[:, None], fi)
        best = None
        for s, bw, d, c in runs:
            if s != site:
                continue
            cv = c.reindex(nd.index).values
            p, w = fit_obs(y, np.column_stack([rain.values, cv]), fi)
            rec = dict(site=site, b_wet=bw, d0=d[0], d1=d[1], r2_fit=r2(y[fi], p[fi]), r2_test=r2(y[te], p[te]), w_cover=float(w[2]),
                       cover_2001=float(c.loc[2001]), cover_2025=float(c.loc[2025]), cover_min=float(c.min()))
            rows.append(rec)
            if rec["w_cover"] > 0 and (best is None or rec["r2_fit"] > best["r2_fit"]):   # more cover can't mean less green
                best, best_p, best_c = rec, p, c
        flat = [r for r in rows if r["site"] == site and r["d0"] == 0.2 and r["d1"] == 0.2 and r["w_cover"] > 0]
        flat = max(flat, key=lambda r: r["r2_fit"]) if flat else None
        out[site] = dict(best=best, constant_d_0_2=flat, rain_only=dict(r2_fit=r2(y[fi], p_rain[fi]), r2_test=r2(y[te], p_rain[te])))
        best_series[site] = (nd, best_p, p_rain, best_c, fi, te)
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "results", "calibration_grid.csv"), index=False)
    json.dump(out, open(os.path.join(HERE, "results", "calibration.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8), gridspec_kw=dict(hspace=0.35, wspace=0.25, width_ratios=[1.6, 1]))
    g = pd.DataFrame(rows)
    for i, site in enumerate(SITES):
        nd, p, pr, c, fi, te = best_series[site]
        ax = axes[i, 0]
        ax.axvspan(TEST[0] - 0.5, TEST[1] + 0.5, color=GREY, alpha=0.1, lw=0)
        ax.plot(nd.index, nd.values, "o", color=NAVY, ms=5, label="MODIS peak greenness")
        ax.plot(nd.index, p, "-", color=RED, lw=1.8, label=f"lattice + this year's rain (test R² {out[site]['best']['r2_test']:.2f})")
        ax.plot(nd.index, pr, "--", color=BLUE, lw=1.4, label=f"rain alone (test R² {out[site]['rain_only']['r2_test']:.2f})")
        ax.set_ylabel("NDVI"); ax.set_title(f"{site.capitalize()}: fitted 2001-2015, tested 2016-2025 (grey)", loc="left")
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
        ax = axes[i, 1]
        gg = g[(g.site == site) & (g.d0 == out[site]["best"]["d0"])]
        piv = gg.pivot(index="d1", columns="b_wet", values="r2_fit")
        im = ax.imshow(piv.values, origin="lower", cmap="Blues", aspect="auto", vmin=max(0, piv.values.min()), vmax=piv.values.max())
        ax.set_xticks(range(len(piv.columns)), [f"{x:.2f}" for x in piv.columns], fontsize=8.5)
        ax.set_yticks(range(len(piv.index)), [f"{x:.2f}" for x in piv.index], fontsize=8.5)
        bi = list(piv.columns).index(out[site]["best"]["b_wet"]); di = list(piv.index).index(out[site]["best"]["d1"])
        ax.plot(bi, di, "*", color=RED, ms=14, mec="white")
        ax.set_xlabel("b in the wet 1950s-60s (rain scale)"); ax.set_ylabel("degradation rate reached by 2025")
        ax.set_title(f"Fit (R², 2001-2015), d = {out[site]['best']['d0']} until 1985", loc="left", fontsize=10.5)
        fig.colorbar(im, ax=ax, fraction=0.05)
    fig.savefig(os.path.join(HERE, "figures", "fig11_calibration.png"), dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
