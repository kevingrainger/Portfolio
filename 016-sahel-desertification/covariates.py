#!/usr/bin/env python3
#-------- covariates.py -------------------------------------------------------------
#-----------------------------------------------------------------------------
# What, besides rain, goes with land loss and recovery across the Sahel?
#
# Rain alone predicted re-greening poorly at Gourma and Fakara. Here the question goes to
# the whole Sahel grid from fetch_covariates.py (sites every 0.5 deg x 1 deg, 13-17 N).
#
#   1. Remove the rain. At each site, greenness (August-September NDVI) is regressed on
#      that year's June-October rain, 2001-2024. The residual is the greenness rain doesn't
#      explain, and its trend (Theil-Sen, as % of the site's mean NDVI per decade) is the
#      site's land gain or loss beyond the weather. This is RESTREND (Evans and Geerken
#      2004), the usual way to separate human from climate effects in the Sahel.
#   2. Relate it to the pressures. The residual trend is regressed on cropland share,
#      livestock density, population density and its growth, topsoil sand and organic
#      carbon, with mean rain and the rain trend as controls. Every covariate is scaled to
#      one standard deviation, so the coefficients compare directly.
#   3. Uncertainty. Neighbouring sites share weather and land use, so plain standard
#      errors are too narrow. Intervals come from a block bootstrap that resamples 4-deg
#      bands of longitude, and the out-of-sample fit from leaving one band out at a time.
#
#   python covariates.py
#
# Output: results/covariates.json, results/covariates_sites.csv, figures/fig9_covariates.png

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import theilslopes, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, RES, FIG = (os.path.join(HERE, d) for d in ("data", "results", "figures"))
NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"
BASEMAP = os.path.join(HERE, "..", "008-spanish-solar-portfolio", "basemap")
SITES = {"Gourma": (15.33, -1.55), "Fakara": (13.50, 2.70)}
rng = np.random.default_rng(0)

COVS = [("crop", "cropland share"), ("log_tlu", "livestock density (log)"), ("log_pop", "population density 2000 (log)"),
        ("pop_growth", "population growth 2000-2020"), ("sand", "topsoil sand"), ("soc", "topsoil organic carbon"),
        ("rain_mean", "mean rain (control)"), ("rain_trend", "rain trend (control)")]


def site_trends():
    y = pd.read_csv(os.path.join(DATA, "sahel_grid.csv")).dropna()
    st = pd.read_csv(os.path.join(DATA, "sahel_static.csv"))
    rows = []
    for i, d in y.groupby("id"):
        if len(d) < 20:
            continue
        b, a = np.polyfit(d.rain, d.ndvi, 1)
        res = d.ndvi - (a + b * d.rain)
        m = d.ndvi.mean()
        rows.append(dict(id=i, ndvi_mean=m, rain_mean=d.rain.mean(), rain_slope=b, rain_r=np.corrcoef(d.rain, d.ndvi)[0, 1],
                         ndvi_trend=theilslopes(d.ndvi, d.year)[0] * 10 / m * 100,
                         resid_trend=theilslopes(res, d.year)[0] * 10 / m * 100,
                         rain_trend=theilslopes(d.rain, d.year)[0] * 10 / d.rain.mean() * 100))
    t = st.merge(pd.DataFrame(rows), on="id")
    # vegetated land only: drop bare desert, where NDVI carries little signal, and open water
    keep = (t.ndvi_mean > 0.12) & (t.ndvi_mean < 0.6) & (t.rain_mean > 100) & (t.soc > 0)
    t = t[keep].copy()
    t["log_tlu"] = np.log1p(t.tlu_km2)
    t["log_pop"] = np.log1p(t.pop2000)
    t["pop_growth"] = np.log((t.pop2020 + 1) / (t.pop2000 + 1))
    return t


def fit(t, cols):
    X = t[cols].values
    X = (X - X.mean(0)) / X.std(0)
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, t.resid_trend.values, rcond=None)
    return beta[1:]


def main():
    t = site_trends()
    cols = [c for c, _ in COVS]
    beta = fit(t, cols)
    band = ((t.lon + 16) // 4).astype(int)
    boot = []
    for _ in range(2000):
        pick = rng.choice(band.unique(), band.nunique())
        tb = pd.concat([t[band == b] for b in pick])
        boot.append(fit(tb, cols))
    boot = np.array(boot)
    # leave one longitude band out
    pred = np.empty(len(t))
    for b in band.unique():
        tr, te = t[band != b], t[band == b]
        Xtr = tr[cols].values; mu, sd = Xtr.mean(0), Xtr.std(0)
        A = np.column_stack([np.ones(len(tr)), (Xtr - mu) / sd])
        w, *_ = np.linalg.lstsq(A, tr.resid_trend.values, rcond=None)
        pred[(band == b).values] = np.column_stack([np.ones(len(te)), (te[cols].values - mu) / sd]) @ w
    y = t.resid_trend.values
    r2_cv = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    A = np.column_stack([np.ones(len(t)), (t[cols].values - t[cols].values.mean(0)) / t[cols].values.std(0)])
    r2_in = 1 - ((y - A @ np.r_[y.mean(), beta]) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    near = {}
    for nm, (la, lo) in SITES.items():
        k = ((t.lat - la) ** 2 + (t.lon - lo) ** 2).idxmin()
        near[nm] = dict(lat=float(t.lat[k]), lon=float(t.lon[k]), resid_trend=float(t.resid_trend[k]),
                        ndvi_trend=float(t.ndvi_trend[k]), crop=float(t.crop[k]), tlu_km2=float(t.tlu_km2[k]))
    out = dict(sites=len(t), median_rain_r=float(t.rain_r.median()),
               share_greening=float((t.ndvi_trend > 0).mean()), share_resid_positive=float((t.resid_trend > 0).mean()),
               median_ndvi_trend=float(t.ndvi_trend.median()), median_resid_trend=float(t.resid_trend.median()),
               r2_in_sample=float(r2_in), r2_leave_band_out=float(r2_cv),
               coefficients={c: dict(beta=float(beta[i]), ci90=[float(np.quantile(boot[:, i], 0.05)), float(np.quantile(boot[:, i], 0.95))],
                                     spearman=float(spearmanr(t[c], t.resid_trend)[0])) for i, c in enumerate(cols)},
               nearest_to_field_sites=near)
    # cropping, livestock and people rise together, so also fit them as one index: the first
    # principal component of the three (scaled), with soil and rain as before
    Z = t[["crop", "log_tlu", "log_pop"]].values
    Z = (Z - Z.mean(0)) / Z.std(0)
    vec = np.linalg.eigh(np.cov(Z.T))[1][:, -1]
    t["land_use"] = Z @ (vec * np.sign(vec.sum()))
    c2 = ["land_use", "pop_growth", "sand", "soc", "rain_mean", "rain_trend"]
    b2 = fit(t, c2)
    bb = np.array([fit(pd.concat([t[band == b] for b in rng.choice(band.unique(), band.nunique())]), c2) for _ in range(2000)])
    out["combined_land_use"] = dict(loadings=dict(zip(["crop", "log_tlu", "log_pop"], map(float, vec * np.sign(vec.sum())))),
                                    variance_share=float(np.linalg.eigvalsh(np.cov(Z.T))[-1] / 3),
                                    beta=float(b2[0]), ci90=[float(np.quantile(bb[:, 0], 0.05)), float(np.quantile(bb[:, 0], 0.95))])
    json.dump(out, open(os.path.join(RES, "covariates.json"), "w"), indent=1)
    t.to_csv(os.path.join(RES, "covariates_sites.csv"), index=False)
    print(json.dumps(out, indent=1))

    # ---------------------------------------------------------------- figure
    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(14, 8.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.8, 1], hspace=0.3, wspace=0.32)
    ax = fig.add_subplot(gs[0, :])
    for nm in ["ne_50m_coastline", "ne_50m_admin_0_boundary_lines_land"]:
        for feat in json.load(open(os.path.join(BASEMAP, f"{nm}.geojson")))["features"]:
            g = feat["geometry"]
            for part in (g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]):
                part = np.asarray(part)
                if ((part[:, 0] > -18) & (part[:, 0] < 18) & (part[:, 1] > 11) & (part[:, 1] < 19)).any():
                    ax.plot(part[:, 0], part[:, 1], color="0.55", lw=0.6, zorder=1)
    lim = np.quantile(abs(t.resid_trend), 0.95)
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    pc = PatchCollection([Rectangle((lo - 0.48, la - 0.24), 0.96, 0.48) for la, lo in zip(t.lat, t.lon)],
                         cmap="RdBu", edgecolor="white", lw=0.4, zorder=2)
    pc.set_array(t.resid_trend.values); pc.set_clim(-lim, lim)
    sc = ax.add_collection(pc)
    for nm, (la, lo) in SITES.items():
        ax.plot(lo, la, "*", color="k", ms=13, zorder=3)
        ax.annotate(nm, (lo, la), xytext=(6, 6), textcoords="offset points", fontsize=10, weight="bold")
    ax.set_xlim(-17, 16.8); ax.set_ylim(12.6, 17.4); ax.set_aspect(1 / np.cos(np.radians(15)))
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    cb = fig.colorbar(sc, ax=ax, fraction=0.02, pad=0.01); cb.set_label("% of mean NDVI per decade")
    ax.set_title("Greenness trend not explained by rain, 2001-2024 (MODIS NDVI against CHIRPS)", loc="left")

    ax = fig.add_subplot(gs[1, 0])
    order = np.argsort(beta)
    for k, i in enumerate(order):
        lo, hi = out["coefficients"][cols[i]]["ci90"]
        col = GREY if lo < 0 < hi else (BLUE if beta[i] > 0 else RED)
        ax.plot([lo, hi], [k, k], color=col, lw=2.2, solid_capstyle="round")
        ax.plot(beta[i], k, "o", color=col, ms=8, mec="white", mew=1.1)
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_yticks(range(len(cols)), [dict(COVS)[cols[i]] for i in order], fontsize=9)
    ax.set_xlabel("change in trend per standard deviation\n(% of mean NDVI per decade)")
    ax.set_title("Effect of each covariate, 90% interval", loc="left", fontsize=10.5)

    for j, (c, lab, logx) in enumerate([("crop", "cropland share (ESA WorldCover 2021)", False),
                                        ("tlu_km2", "livestock units per km² (FAO GLW4, 2015)", True)]):
        ax = fig.add_subplot(gs[1, j + 1])
        x = t[c] + (1 if logx else 0)
        ax.scatter(x, t.resid_trend, s=16, color=NAVY, alpha=0.45, lw=0)
        xs = np.log(x) if logx else x
        bins = np.quantile(xs, np.linspace(0, 1, 9))
        idx = np.clip(np.digitize(xs, bins[1:-1]), 0, 7)
        med = [np.median(t.resid_trend.values[idx == b]) for b in range(8)]
        mid = [np.median(x.values[idx == b]) for b in range(8)]
        ax.plot(mid, med, "-o", color=RED, lw=1.8, ms=5, label="median of each eighth")
        ax.axhline(0, color="0.6", lw=0.8)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(lab); ax.set_ylabel("trend not explained by rain (%/decade)" if j == 0 else "")
        ax.set_ylim(-lim * 1.6, lim * 1.6)
        ax.set_title(["Rain-adjusted trend against cropland", "Rain-adjusted trend against livestock"][j], loc="left", fontsize=10.5)
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.savefig(os.path.join(FIG, "fig9_covariates.png"), dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
