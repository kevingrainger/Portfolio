#!/usr/bin/env python3
#-------- make_cover.py ---------------------------------------------------------------
# The 008 cover: how far the weather reaches from Seville, season by season, and
# correlation against distance for all 300 site pairs. Kept here rather than in
# ../make_covers.py because it needs the anemoi environment.
#
#   python make_cover.py      (in .venv-anemoi)

import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import figstyle
import solar_sites as ss

warnings.filterwarnings("ignore", category=RuntimeWarning)
figstyle.apply()

import json


def basemap_lines():
    """Coastlines and land borders from Natural Earth (public domain), as lon/lat polylines."""
    out = []
    for name in ["ne_50m_coastline", "ne_50m_admin_0_boundary_lines_land"]:
        for feat in json.load(open(os.path.join(HERE, "basemap", f"{name}.geojson")))["features"]:
            g = feat["geometry"]
            parts = g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]
            for p in parts:
                p = np.asarray(p)
                if ((p[:, 0] > -12) & (p[:, 0] < 6) & (p[:, 1] > 34) & (p[:, 1] < 46)).any():
                    out.append(p)
    return out


LINES = basemap_lines()
D = ss.load()
A_sites = ss.deseasonalise(D["k_sites"], D["times"])
A_grid = ss.deseasonalise(D["k_grid"], D["times"])
lat, lon = D["grid_lat"], D["grid_lon"]
ulat, ulon = np.unique(lat), np.unique(lon)
ref_i = D["names"].index("Sevilla")

fig = plt.figure(figsize=(13.0, 6.5), dpi=200)
colours = ["#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b"]          # winter, spring, summer, autumn
ax_fit = fig.add_axes([0.66, 0.12, 0.32, 0.74])
for k, ((name, months), col) in enumerate(zip(ss.SEASONS.items(), colours)):
    m = ss.season_mask(D["times"], months)
    #pairwise: each grid point uses every hour where it and Seville both have daylight
    ref = A_sites[ref_i, m]
    G = A_grid[:, m]
    both = np.isfinite(G) & np.isfinite(ref)[None, :]
    n = both.sum(1)
    Gz, rz = np.where(both, G, 0.0), np.where(both, ref[None, :], 0.0)
    mg, mr = Gz.sum(1) / n, rz.sum(1) / n
    Gc, rc = np.where(both, G - mg[:, None], 0.0), np.where(both, ref[None, :] - mr[:, None], 0.0)
    r = (Gc * rc).sum(1) / np.sqrt((Gc ** 2).sum(1) * (rc ** 2).sum(1))
    r[n < 50] = np.nan
    img = np.full((len(ulat), len(ulon)), np.nan)
    img[np.searchsorted(ulat, lat), np.searchsorted(ulon, lon)] = r

    ax = fig.add_axes([0.03 + (k % 2) * 0.30, 0.52 - (k // 2) * 0.44, 0.28, 0.38])
    im = ax.imshow(img, origin="lower", extent=[ulon[0], ulon[-1], ulat[0], ulat[-1]],
                   cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto", interpolation="bilinear", alpha=0.7)
    for p in LINES:
        ax.plot(p[:, 0], p[:, 1], color="0.15", lw=0.6)
    ax.set_xlim(ulon[0], ulon[-1]); ax.set_ylim(ulat[0], ulat[-1])
    ax.contour(ulon, ulat, img, levels=[np.exp(-1)], colors="k", linewidths=1.0, linestyles="--")
    ax.scatter(D["lon"], D["lat"], s=10, c="w", edgecolors="k", linewidths=0.5)
    ax.plot(D["lon"][ref_i], D["lat"][ref_i], "*", color="#c0392b", ms=12, mec="k")
    ax.set_title(name, loc="left", fontsize=11, color=col, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

    C, X = ss.correlation(A_sites[:, m])
    L, p, se, d, rr = ss.fit_correlation_length(C, D["lat"], D["lon"])
    ax_fit.scatter(d, rr, s=7, color=col, alpha=0.35, lw=0)
    dd = np.linspace(0, 1100, 200)
    ax_fit.plot(dd, ss.stretched_exponential(dd, L, p), color=col, lw=2.2, label=f"{name.split()[0]}: {L:.0f} km")

cax = fig.add_axes([0.03, 0.035, 0.58, 0.022])
fig.colorbar(im, cax=cax, orientation="horizontal", label="correlation of clear-sky index anomaly with Seville (dashed line: 1/e)")
ax_fit.axhline(np.exp(-1), color="0.5", lw=0.8, ls=":")
ax_fit.set_xlim(0, 1100); ax_fit.set_ylim(-0.1, 1.0)
ax_fit.set_xlabel("distance between sites (km)")
ax_fit.set_ylabel("correlation")
ax_fit.set_title("Correlation against distance, all 300 site pairs", loc="left", fontsize=11)
ax_fit.legend(title="1/e distance", fontsize=9)
fig.text(0.03, 0.95, "Correlation of solar output with Seville, ERA5 2023, by season", fontsize=12.5)
figstyle.save(fig, os.path.join(HERE, "figures", "cover"))
print("solar cover done")
