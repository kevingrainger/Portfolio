#!/usr/bin/env python3
#-------- joint_fit.py --------------------------------------------------------------
#-----------------------------------------------------------------------------
# One set of lattice rates for the whole Sahel, tested against dry-season greenness.
#
# Fitting two sites one at a time (calibrate_sites.py) couldn't tell the lattice's slow
# cover from the grass. Here the target is dry-season greenness (March-May MODIS NDVI,
# when the grass has died back: mostly shrubs and trees), and every vegetated grid site is
# run with the same rates, so they have to work everywhere:
#
#   rain -> b     b(t) = B0 x (3-year mean June-September CHIRPS rain) / 400 mm, capped
#                 at 1: one scale for all sites, so drier sites get a lower b
#   land use      d = D0 + D1 x the site's land-use index (cropland, livestock and people
#                 combined, from covariates.py, in standard deviations), floored at 0.02
#   lattice       60 x 60, f = 0.9, settled 50 years in the 1981-83 climate, run 1981-2024,
#                 2 seeds per site
#
# Compared with dry-season NDVI, 2001-2024, on three things:
#   level         across sites, mean cover against mean dry NDVI (rank correlation)
#   trend         across sites, cover trend against dry-NDVI trend (Theil-Sen, rank corr.)
#   year to year  within each site, correlation of the two series (median over sites)
# Rates fitted (grid search, on level + trend) on the sites in even 4-degree longitude
# bands, tested on the odd ones. Benchmark: rain alone (3-year mean rain in place of cover).
#
#   python joint_fit.py
#
# Output: results/joint_fit.json, results/joint_fit_sites.csv, figures/fig12_joint_fit.png

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
D0S = [0.1, 0.2, 0.3]
D1S = [0.0, 0.05]
SEEDS, N = 2, 60
YEARS = list(range(1981, 2025))


def site_inputs():
    t = pd.read_csv(os.path.join(RES, "covariates_sites.csv"))
    r = pd.read_csv(os.path.join(DATA, "sahel_chirps_jjas.csv")).pivot(index="year", columns="id", values="rain_jjas")
    dn = pd.read_csv(os.path.join(DATA, "sahel_dry_ndvi.csv")).pivot(index="year", columns="id", values="dry_ndvi")
    t = t[t.id.isin(r.columns) & t.id.isin(dn.columns)].copy()
    p3 = r.rolling(3, min_periods=1).mean()
    return t, p3, dn


def run_site(args):
    sid, p3, d, seed_base = args
    from sahel_ca import Landscape
    covers = []
    for s in range(SEEDS):
        b = np.minimum(1.0, p3)
        land = Landscape(n=N, b=float(b[0]), cover=0.5, seed=seed_base + s, d=d, f=0.9)
        land.step(50)
        out = []
        for bt in b:
            land.b = float(bt)
            if land.cover() == 0:
                land.grid[land.rng.random(land.grid.shape) < 0.01] = 1
            land.step(1)
            out.append(land.cover())
        covers.append(out)
    return sid, np.mean(covers, axis=0)


def metrics(t, cov, dn, ids):
    yrs = [y for y in YEARS if 2001 <= y <= 2024]
    lv_m, lv_o, tr_m, tr_o, yy = [], [], [], [], []
    for i in ids:
        c = pd.Series(cov[i], index=YEARS).loc[yrs]; o = dn[i].loc[yrs]
        ok = o.notna() & c.notna()
        if ok.sum() < 15:
            continue
        lv_m.append(c[ok].mean()); lv_o.append(o[ok].mean())
        tr_m.append(theilslopes(c[ok].values, np.array(yrs)[ok.values])[0]); tr_o.append(theilslopes(o[ok].values, np.array(yrs)[ok.values])[0])
        yy.append(np.corrcoef(c[ok], o[ok])[0, 1] if c[ok].std() > 0 else np.nan)
    return dict(level=float(spearmanr(lv_m, lv_o)[0]), trend=float(spearmanr(tr_m, tr_o)[0]),
                year_to_year=float(np.nanmedian(yy)), sites=len(lv_m))


def main():
    t, p3, dn = site_inputs()
    band = ((t.lon + 16) // 4).astype(int)
    train = set(t.id[band % 2 == 0]); test = set(t.id[band % 2 == 1])
    lu = dict(zip(t.id, t.land_use))
    results = {}
    with Pool(max(1, os.cpu_count() - 1)) as pool:
        for B0, D0, D1 in product(B0S, D0S, D1S):
            jobs = [(i, (B0 * p3[i].loc[YEARS] / 400).values, max(0.02, D0 + D1 * lu[i]), 1000 + int(i) * 10) for i in t.id]
            cov = dict(pool.map(run_site, jobs))
            results[(B0, D0, D1)] = cov
            print(B0, D0, D1, "done", flush=True)
    rows = []
    for k, cov in results.items():
        mtr, mte = metrics(t, cov, dn, train), metrics(t, cov, dn, test)
        rows.append(dict(B0=k[0], D0=k[1], D1=k[2], **{f"train_{a}": v for a, v in mtr.items()}, **{f"test_{a}": v for a, v in mte.items()}))
    g = pd.DataFrame(rows)
    g["train_score"] = g.train_level + g.train_trend
    best = g.loc[g.train_score.idxmax()]
    rain_cov = {i: p3[i].loc[YEARS].values for i in t.id}
    base = dict(train=metrics(t, rain_cov, dn, train), test=metrics(t, rain_cov, dn, test))
    flat = g[(g.D1 == 0)].sort_values("train_score").iloc[-1]
    lu_best = g[(g.D1 > 0)].sort_values("train_score").iloc[-1]
    out = dict(best=best.to_dict(), best_same_d=flat.to_dict(), best_d_with_land_use=lu_best.to_dict(), rain_alone=base, sites=len(t))
    g.to_csv(os.path.join(RES, "joint_fit_grid.csv"), index=False)
    json.dump(out, open(os.path.join(RES, "joint_fit.json"), "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))

    cov = results[(best.B0, best.D0, best.D1)]
    yrs = list(range(2001, 2025))
    site_rows = []
    for i in t.id:
        c = pd.Series(cov[i], index=YEARS).loc[yrs]; o = dn[i].loc[yrs]
        site_rows.append(dict(id=i, lat=float(t.lat[t.id == i].iloc[0]), lon=float(t.lon[t.id == i].iloc[0]), test=i in test,
                              cover_mean=c.mean(), ndvi_mean=o.mean(), cover_trend=theilslopes(c.values, yrs)[0] * 10,
                              ndvi_trend=theilslopes(o.values, yrs)[0] * 10, rain_mean=float(p3[i].loc[yrs].mean())))
    sr = pd.DataFrame(site_rows); sr.to_csv(os.path.join(RES, "joint_fit_sites.csv"), index=False)

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), gridspec_kw=dict(wspace=0.32))
    te = sr[sr.test]
    ax = axes[0]
    ax.scatter(te.cover_mean, te.ndvi_mean, s=18, color=NAVY, alpha=0.6, lw=0)
    ax.set_xlabel("lattice cover, 2001-2024 mean"); ax.set_ylabel("dry-season NDVI, 2001-2024 mean")
    ax.set_title(f"Level across test sites (rank corr. {best.test_level:.2f})", loc="left", fontsize=10.5)
    ax = axes[1]
    ax.scatter(100 * te.cover_trend, 100 * te.ndvi_trend, s=18, color=RED, alpha=0.6, lw=0)
    ax.axhline(0, color="0.7", lw=0.8); ax.axvline(0, color="0.7", lw=0.8)
    ax.set_xlabel("lattice cover trend, points per decade"); ax.set_ylabel("dry-season NDVI trend, x100 per decade")
    ax.set_title(f"Trend across test sites (rank corr. {best.test_trend:.2f})", loc="left", fontsize=10.5)
    ax = axes[2]
    labels = ["level", "trend", "year to year"]
    keys = ["level", "trend", "year_to_year"]
    for k, (lab, key) in enumerate(zip(labels, keys)):
        ax.plot(base["test"][key], k - 0.12, "s", color=BLUE, ms=8, label="rain alone" if k == 0 else None)
        ax.plot(flat[f"test_{key}"], k, "D", color=NAVY, ms=8, label="lattice, same d everywhere" if k == 0 else None)
        ax.plot(lu_best[f"test_{key}"], k + 0.12, "o", color=RED, ms=9, label="lattice, d rising with land use" if k == 0 else None)
    ax.axvline(0, color="0.7", lw=0.8)
    ax.set_yticks(range(3), labels); ax.invert_yaxis(); ax.set_xlim(-0.5, 1)
    ax.set_xlabel("agreement with dry-season NDVI on test sites")
    ax.set_title("Test sites: lattice against rain alone", loc="left", fontsize=10.5); ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=1)
    fig.savefig(os.path.join(FIG, "fig12_joint_fit.png"), dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
