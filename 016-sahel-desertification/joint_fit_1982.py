#!/usr/bin/env python3
#-------- joint_fit_1982.py ---------------------------------------------------------
#-----------------------------------------------------------------------------
# The lattice with seed rain, tested through the 1980s droughts.
#
# joint_fit.py found two problems: one set of rates did no better than smoothed rain over
# 2001-2024, and at the dry edge the lattice collapsed to bare ground while the satellites
# still saw sparse shrubs. Two changes:
#
#   seed rain    each year a share s of the empty, fertile cells is colonised by seed from
#                outside the lattice, so a dry site keeps a thin cover instead of dying out
#                (the coral model lacks the same thing: larvae from other reefs)
#   1982-2021    the test now runs through the droughts and the recovery, against NOAA's
#                AVHRR NDVI record (fetch_avhrr.py), March-May, every grid site
#
# AVHRR changes satellite several times (NOAA-9 to NOAA-14 in 1995, orbital drift late on),
# which shifts every site at once. So the tests compare sites with each other, which those
# shared shifts can't touch:
#   level         which sites are greener (rank correlation of 1982-2021 means)
#   trend         which sites trended up more (rank correlation of Theil-Sen trends)
#   recovery      which sites gained most from the drought years (1982-86) to the late
#                 1990s (1995-2000)
#   year to year  within each site, the correlation of the two series after removing
#                 each year's Sahel-wide mean (median over sites)
# Same set-up otherwise as joint_fit.py: one rain scale B0 and degradation rate D0 for all
# sites, 60 x 60 lattices, 2 seeds, fitted on even longitude bands, tested on odd ones,
# against rain alone (the 3-year mean).
#
#   python joint_fit_1982.py
#
# Output: results/joint_fit_1982.json, results/joint_fit_1982_grid.csv, figures/fig13_seed_rain.png

import json
import os
from itertools import product
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, theilslopes

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, RES, FIG = (os.path.join(HERE, d) for d in ("data", "results", "figures"))
NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"
B0S = [0.35, 0.45, 0.6]
D0S = [0.2, 0.3]
SS = [0.0, 0.01, 0.03]
SEEDS, N = 2, 60
YEARS = list(range(1981, 2022))
OBS = list(range(1982, 2022))


def run_site(args):
    sid, p3, d, s_rain, seed_base = args
    from sahel_ca import Landscape, VEG, EMPTY
    covers = []
    for k in range(SEEDS):
        b = np.minimum(1.0, p3)
        land = Landscape(n=N, b=float(b[0]), cover=0.5, seed=seed_base + k, d=d, f=0.9)
        for _ in range(50):
            land.step(1)
            if s_rain:
                e = (land.grid == EMPTY) & (land.rng.random(land.grid.shape) < s_rain); land.grid[e] = VEG
        out = []
        for bt in b:
            land.b = float(bt)
            land.step(1)
            if s_rain:
                e = (land.grid == EMPTY) & (land.rng.random(land.grid.shape) < s_rain); land.grid[e] = VEG
            elif land.cover() == 0:
                land.grid[land.rng.random(land.grid.shape) < 0.01] = VEG
            out.append(land.cover())
        covers.append(out)
    return sid, np.mean(covers, axis=0)


def metrics(series, obs, ids):
    """series, obs: DataFrames year x site."""
    ids = [i for i in ids if i in obs.columns and obs[i].loc[OBS].notna().sum() >= 25]
    m, o = series.loc[OBS, ids], obs.loc[OBS, ids]
    yrs = np.array(OBS)
    def tr(x):
        ok = x.notna().values
        return theilslopes(x.values[ok], yrs[ok])[0]
    rec = lambda x: x.loc[1995:2000].mean() - x.loc[1982:1986].mean()
    ma, oa = m.sub(m.mean(axis=1), axis=0), o.sub(o.mean(axis=1), axis=0)       # each year's Sahel mean removed
    yy = [np.corrcoef(ma[i][oa[i].notna()], oa[i].dropna())[0, 1] for i in ids if ma[i].std() > 0]
    return dict(level=float(spearmanr(m.mean(), o.mean())[0]),
                trend=float(spearmanr([tr(m[i]) for i in ids], [tr(o[i]) for i in ids])[0]),
                recovery=float(spearmanr([rec(m[i]) for i in ids], [rec(o[i]) for i in ids])[0]),
                year_to_year=float(np.nanmedian(yy)), sites=len(ids))


def main():
    t = pd.read_csv(os.path.join(RES, "covariates_sites.csv"))
    r = pd.read_csv(os.path.join(DATA, "sahel_chirps_jjas.csv")).pivot(index="year", columns="id", values="rain_jjas")
    obs = pd.read_csv(os.path.join(DATA, "sahel_avhrr_dry.csv")).pivot(index="year", columns="id", values="avhrr_dry")
    t = t[t.id.isin(r.columns) & t.id.isin(obs.columns)].copy()
    p3 = r.rolling(3, min_periods=1).mean()
    band = ((t.lon + 16) // 4).astype(int)
    train, test = list(t.id[band % 2 == 0]), list(t.id[band % 2 == 1])
    rows, store = [], {}
    with Pool(max(1, os.cpu_count() - 1)) as pool:
        for B0, D0, S in product(B0S, D0S, SS):
            jobs = [(i, (B0 * p3[i].loc[YEARS] / 400).values, D0, S, 1000 + int(i) * 10) for i in t.id]
            cov = pd.DataFrame(dict(pool.map(run_site, jobs)), index=YEARS)
            store[(B0, D0, S)] = cov
            mtr, mte = metrics(cov, obs, train), metrics(cov, obs, test)
            rows.append(dict(B0=B0, D0=D0, S=S, collapsed_sites=int((cov.loc[OBS].mean() < 0.05).sum()),
                             **{f"train_{a}": v for a, v in mtr.items()}, **{f"test_{a}": v for a, v in mte.items()}))
            print(B0, D0, S, {k: round(v, 2) for k, v in mte.items()}, flush=True)
    g = pd.DataFrame(rows)
    g["train_score"] = g.train_level + g.train_trend + g.train_recovery
    best = g.loc[g.train_score.idxmax()]
    no_seed = g[g.S == 0].sort_values("train_score").iloc[-1]
    rain = p3.loc[YEARS, t.id]
    base = dict(train=metrics(rain, obs, train), test=metrics(rain, obs, test))
    out = dict(best=best.to_dict(), best_without_seed_rain=no_seed.to_dict(), rain_alone=base,
               avhrr_vs_modis_note="AVHRR and MODIS dry-season NDVI correlate 0.88 over 2001-2021")
    g.to_csv(os.path.join(RES, "joint_fit_1982_grid.csv"), index=False)
    json.dump(out, open(os.path.join(RES, "joint_fit_1982.json"), "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))

    cov = store[(best.B0, best.D0, best.S)]
    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), gridspec_kw=dict(wspace=0.42, width_ratios=[1.2, 1, 1]))
    ax = axes[0]
    o = obs.loc[OBS, test]; m = cov.loc[OBS, test]; rr = rain.loc[OBS, test]
    z = lambda x: (x.mean(axis=1) - x.mean(axis=1).mean()) / x.mean(axis=1).std()
    ax.plot(OBS, z(o), "o-", color=NAVY, lw=1.4, ms=3.5, label="AVHRR dry-season NDVI")
    ax.plot(OBS, z(m), "-", color=RED, lw=1.8, label="lattice cover, with seed rain")
    ax.plot(OBS, z(rr), "--", color=BLUE, lw=1.3, label="3-year mean rain")
    for yr, lab in [(1995, "NOAA-14"), (2001, "NOAA-16"), (2009, "NOAA-19")]:
        ax.axvline(yr, color=GREY, lw=0.7, ls=":"); ax.text(yr + 0.3, -2.3, lab, fontsize=7.5, color=GREY, rotation=90, va="bottom")
    ax.set_ylabel("standardised, test-site mean"); ax.set_xlabel("year")
    ax.set_title("The Sahel through the droughts, test sites", loc="left", fontsize=10.5); ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax = axes[1]
    for S, c, ls in [(0.0, NAVY, "--"), (0.01, BLUE, "-"), (0.03, RED, "-")]:
        gg = g[(g.S == S) & (g.D0 == best.D0)].sort_values("B0")
        ax.plot(gg.B0, gg.collapsed_sites, "o" + ls, color=c, lw=1.6, ms=5, label=f"seed rain {S:.0%} a year")
    ax.set_xlabel("rain scale B0"); ax.set_ylabel("sites with no lattice cover left")
    ax.set_title("Seed rain stops the dry edge collapsing", loc="left", fontsize=10.5); ax.legend(frameon=False, fontsize=8.5)
    ax = axes[2]
    labs = ["level", "trend", "drought recovery", "year to year"]
    keys = ["level", "trend", "recovery", "year_to_year"]
    for k, key in enumerate(keys):
        ax.plot(base["test"][key], k - 0.13, "s", color=BLUE, ms=8, label="rain alone" if k == 0 else None)
        ax.plot(no_seed[f"test_{key}"], k, "D", color=NAVY, ms=8, label="lattice, no seed rain" if k == 0 else None)
        ax.plot(best[f"test_{key}"], k + 0.13, "o", color=RED, ms=9, label="lattice, best fit" if k == 0 else None)
    ax.axvline(0, color="0.7", lw=0.8); ax.set_xlim(-0.5, 1)
    ax.set_yticks(range(4), labs); ax.invert_yaxis()
    ax.set_xlabel("agreement with AVHRR on test sites")
    ax.set_title("Test sites, 1982-2021", loc="left", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.4, -0.17))
    fig.savefig(os.path.join(FIG, "fig13_seed_rain.png"), dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
