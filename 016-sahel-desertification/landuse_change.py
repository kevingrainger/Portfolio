#!/usr/bin/env python3
#-------- landuse_change.py ---------------------------------------------------------
#-----------------------------------------------------------------------------
# Land use as it changes, not as a snapshot. The covariate study (covariates.py) used
# cropland in 2021 and livestock in 2015. Here cropland is followed year by year, 2001-2023,
# from the MODIS land-cover product MCD12Q1 (500 m; IGBP classes 12, croplands, and 14,
# cropland/natural mosaic, the mosaic counted as half), for the same Sahel grid sites.
#
# Two tests:
#   change on change   across sites, the rain-adjusted greenness trend against the change
#                      in cropland share from 2001-03 to 2021-23, with the starting
#                      cropland, livestock, population, soil and rain as controls
#   panel              within each site, does a year with more cropland have greenness
#                      above or below what its rain predicts? Site and year effects are
#                      removed (each site's own level, and each year's Sahel-wide
#                      anomaly), so only a site's own changes over time count
#
# MODIS at 500 m misses many small Sahel fields, so its cropland share is low in level;
# its year-to-year changes are what is used here. Intervals: block bootstrap over 4-deg
# bands of longitude, as in covariates.py.
#
#   python landuse_change.py      (Earth Engine account; EE_PROJECT)
#
# Output: data/sahel_cropland_modis.csv, results/landuse_change.json, figures/fig10_landuse_change.png

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, RES, FIG = (os.path.join(HERE, d) for d in ("data", "results", "figures"))
NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"
YEARS = range(2001, 2024)
HALF = 0.1
rng = np.random.default_rng(0)


def fetch():
    path = os.path.join(DATA, "sahel_cropland_modis.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    import ee
    ee.Initialize(project=os.environ.get("EE_PROJECT", "zinc-interface-468219-a4"))
    g = pd.read_csv(os.path.join(DATA, "sahel_static.csv"))[["id", "lat", "lon"]]
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Rectangle([r.lon - HALF, r.lat - HALF, r.lon + HALF, r.lat + HALF]), {"id": int(r.id)})
                               for r in g.itertuples()])
    lc = ee.ImageCollection("MODIS/061/MCD12Q1").select("LC_Type1")
    bands = []
    for y in YEARS:
        im = lc.filterDate(f"{y}-01-01", f"{y}-12-31").first()
        bands.append(im.eq(12).add(im.eq(14).multiply(0.5)).rename(f"crop_{y}"))
    rows = ee.Image.cat(bands).reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=500, tileScale=4).getInfo()
    w = pd.DataFrame([f["properties"] for f in rows["features"]])
    long = pd.concat([w[["id", f"crop_{y}"]].rename(columns={f"crop_{y}": "crop_modis"}).assign(year=y) for y in YEARS])
    long.to_csv(path, index=False)
    return long


def ols(y, X):
    X = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(X, y, rcond=None)[0][1:]


def main():
    crop = fetch()
    t = pd.read_csv(os.path.join(RES, "covariates_sites.csv"))
    cw = crop.pivot(index="id", columns="year", values="crop_modis")
    t["crop_start"] = t.id.map(cw[[2001, 2002, 2003]].mean(axis=1))
    t["crop_change"] = t.id.map(cw[[2021, 2022, 2023]].mean(axis=1) - cw[[2001, 2002, 2003]].mean(axis=1))
    band = ((t.lon + 16) // 4).astype(int)

    # change on change, standardised
    cols = ["crop_change", "crop_start", "log_tlu", "log_pop", "sand", "soc", "rain_mean", "rain_trend"]
    def cc(tt):
        X = tt[cols].values; X = (X - X.mean(0)) / X.std(0)
        return ols(tt.resid_trend.values, X)
    b = cc(t)
    boot = np.array([cc(pd.concat([t[band == k] for k in rng.choice(band.unique(), band.nunique())])) for _ in range(2000)])

    # panel: residual greenness against cropland, site and year effects removed
    y = pd.read_csv(os.path.join(DATA, "sahel_grid.csv")).dropna()
    y = y[y.id.isin(t.id) & y.year.isin(YEARS)]
    resid = []
    for i, d in y.groupby("id"):
        bb, aa = np.polyfit(d.rain, d.ndvi, 1)
        resid.append(d.assign(res=(d.ndvi - (aa + bb * d.rain)) / d.ndvi.mean() * 100))
    p = pd.concat(resid).merge(crop, on=["id", "year"])
    p = p.merge(t[["id", "lon"]], on="id")
    def demean(df, c):
        v = df[c] - df.groupby("id")[c].transform("mean")
        return v - v.groupby(df.year).transform("mean")
    def panel(df):
        x, yy = demean(df, "crop_modis") * 100, demean(df, "res")     # x in percentage points of cropland
        return float((x * yy).sum() / (x * x).sum())
    pb = panel(p)
    pband = ((p.lon + 16) // 4).astype(int)
    groups = {k: p[pband == k] for k in pband.unique()}
    pboot = [panel(pd.concat([groups[k] for k in rng.choice(list(groups), len(groups))])) for _ in range(1000)]
    out = dict(sites=len(t), modis_crop_start_median=float(t.crop_start.median()),
               share_sites_cropland_grew=float((t.crop_change > 0.005).mean()),
               crop_change_median_pts=float(100 * t.crop_change.median()),
               worldcover_vs_modis_rank_corr=float(t[["crop", "crop_start"]].corr(method="spearman").iloc[0, 1]),
               change_on_change=dict(zip(cols, [dict(beta=float(b[i]), ci90=[float(np.quantile(boot[:, i], 0.05)), float(np.quantile(boot[:, i], 0.95))])
                                                for i in range(len(cols))])),
               panel_per_pt_cropland=dict(beta=pb, ci90=[float(np.quantile(pboot, 0.05)), float(np.quantile(pboot, 0.95))], site_years=len(p)))
    json.dump(out, open(os.path.join(RES, "landuse_change.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), gridspec_kw=dict(wspace=0.5, width_ratios=[1, 1.05, 0.9]))
    ax = axes[0]
    q = 100 * cw.loc[t.id]
    for lo, hi, a in [(0.1, 0.9, 0.18), (0.25, 0.75, 0.35)]:
        ax.fill_between(q.columns, q.quantile(lo), q.quantile(hi), color=BLUE, alpha=a, lw=0, label=f"sites, {int(lo*100)}-{int(hi*100)}%")
    ax.plot(q.columns, q.median(), color=NAVY, lw=1.8, label="median site")
    ax.plot(q.columns, q.mean(), color=RED, lw=1.4, ls="--", label="mean")
    ax.set_xlabel("year"); ax.set_ylabel("cropland share of the box (%)")
    ax.set_title("Cropland share by year (MODIS MCD12Q1)", loc="left"); ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax = axes[1]
    sc = ax.scatter(100 * t.crop_change, t.resid_trend, c=100 * t.crop_start, cmap="Blues", s=22, edgecolor="0.4", lw=0.3, vmin=0, vmax=np.quantile(100 * t.crop_start, 0.95))
    xx = np.linspace(100 * t.crop_change.min(), 100 * t.crop_change.max(), 50)
    sl = np.polyfit(100 * t.crop_change, t.resid_trend, 1)
    ax.plot(xx, np.polyval(sl, xx), color=RED, lw=1.6, label="least-squares line")
    ax.axhline(0, color="0.6", lw=0.8); ax.axvline(0, color="0.6", lw=0.8)
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, location="top"); cb.set_label("cropland 2001-03 (%)", fontsize=9); cb.ax.tick_params(labelsize=8)
    ax.set_xlabel("change in cropland share, 2001-03 to 2021-23 (points)"); ax.set_ylabel("greenness trend not explained by rain (%/decade)")
    ax.text(0, -0.2, "Rain-adjusted trend against cropland change", transform=ax.transAxes, fontsize=10.5); ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax = axes[2]
    lab = {"crop_change": "cropland change", "crop_start": "cropland 2001-03", "log_tlu": "livestock (log)", "log_pop": "population (log)",
           "sand": "topsoil sand", "soc": "topsoil organic carbon", "rain_mean": "mean rain", "rain_trend": "rain trend"}
    order = np.argsort(b)
    for k, i in enumerate(order):
        lo, hi = out["change_on_change"][cols[i]]["ci90"]
        col = GREY if lo < 0 < hi else (BLUE if b[i] > 0 else RED)
        ax.plot([lo, hi], [k, k], color=col, lw=2.2, solid_capstyle="round")
        ax.plot(b[i], k, "o", color=col, ms=8, mec="white", mew=1.1)
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_yticks(range(len(cols)), [lab[cols[i]] for i in order], fontsize=9)
    ax.set_xlabel("change in trend per standard deviation (%/decade)")
    ax.set_title("Effect of each covariate, 90% interval", loc="left")
    fig.savefig(os.path.join(FIG, "fig10_landuse_change.png"), dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
