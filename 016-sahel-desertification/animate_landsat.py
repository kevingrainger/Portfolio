#!/usr/bin/env python3
#-------- animate_landsat.py --------------------------------------------------------
#-----------------------------------------------------------------------------
# The lattice beside forty years of real Landsat images, at Gourma and Fakara.
#
# Each frame is one year, 1985-2024. For each site:
#   left     the satellite picture of the 6 x 6 km site in true colour: the clearest
#            October-November scene (fetch_site_pictures.py), Landsat at 30 m to 2015 and
#            Sentinel-2 at 10 m from 2016, one brightness stretch for all three bands. Bright
#            green marks pixels clearly greener than the scene's own background (NDVI 0.05
#            above its median): shrubs, trees and wet hollows. Landsat 5's archive over West Africa has
#            long gaps, so the most recent available scene is shown, labelled with its
#            sensor and date. Each scene is brightness-stretched on its own,
#            because dry-season dust haze varies from year to year; the chart is the measure
#   middle   the lattice driven by the site's own rain to that year: one cell per 30 m
#            Landsat pixel (200 x 200), the rates from joint_fit_1982.py (b = 0.45 x 3-year
#            mean CHIRPS June-September rain / 400 mm, degradation 0.3, 3% seed rain a year),
#            settled 50 years in the early-1980s climate
#   right    the lattice's cover and Landsat's mean dry-season (February-May) NDVI over the box,
#            to that year: the dry season shows the woody cover the lattice models
#
#   python animate_landsat.py
#
# Output: figures/fig15_landsat_animation.gif, figures/fig15_landsat_final.png

import io
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image

from sahel_ca import Landscape, VEG, EMPTY

HERE = os.path.dirname(os.path.abspath(__file__))
NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"
CMAP = ListedColormap(["#b08850", "#f2e6cf", "#2e7d32"])        # degraded, empty, vegetated
SITES = {"gourma": "Gourma, Mali", "fakara": "Fakara, Niger"}
B0, D0, SEED_RAIN, N = 0.45, 0.3, 0.03, 200
VEG_MARGIN = 0.05                      # overlay: NDVI this far above the scene's median
VEG_RGBA = (0.0, 0.9, 0.35, 0.8)       # bright green
YEARS = list(range(1985, 2025))


def site_rain(site):
    r = pd.read_csv(os.path.join(HERE, "data", f"chirps_{site}.csv"), index_col=0, parse_dates=True).squeeze()
    jjas = r[r.index.month.isin([6, 7, 8, 9])]
    P = jjas.groupby(jjas.index.year).sum()
    return P.rolling(3, min_periods=1).mean().loc[1981:2024]


def run_lattice(site):
    p3 = site_rain(site)
    land = Landscape(n=N, b=float(min(1, B0 * p3.loc[1981:1983].mean() / 400)), cover=0.5, seed=7, d=D0, f=0.9)
    def seed():
        e = (land.grid == EMPTY) & (land.rng.random(land.grid.shape) < SEED_RAIN); land.grid[e] = VEG
    for _ in range(50):
        land.step(1); seed()
    grids, cover = {}, {}
    for y, p in p3.items():
        land.b = float(min(1, B0 * p / 400))
        land.step(1); seed()
        if y in YEARS:
            grids[y] = land.grid.copy(); cover[y] = land.cover()
    return grids, pd.Series(cover), p3


def veg_overlay(img):
    """Pixels clearly greener than the scene's own background (shrubs, trees, wet hollows).
    Relative to each scene, because October-November grass is green in some years and
    dry in others; the overlay shows where vegetation is, not how much."""
    nd = (img[..., 3] - img[..., 0]) / (img[..., 3] + img[..., 0] + 1e-6)
    mask = nd > np.nanmedian(nd) + VEG_MARGIN
    over = np.zeros(mask.shape + (4,)); over[mask] = VEG_RGBA
    return over


def stretch(rgb):
    """One stretch for all three bands, so the colours stay as Landsat saw them."""
    lo, hi = np.nanpercentile(rgb, [1, 99.5])
    return np.nan_to_num(np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.9)


def main():
    data = {}
    for site in SITES:
        d = np.load(os.path.join(HERE, "data", f"landsat_{site}.npz"))
        pics = np.load(os.path.join(HERE, "data", f"pictures_{site}.npz"))
        grids, cover, p3 = run_lattice(site)
        nd = pd.Series([float(np.nanmean(x)) for x in d["ndvi"]], index=d["years"])
        data[site] = dict(years=list(pics["years"]), dates=list(pics["dates"]), sensors=list(pics["sensors"]),
                          rgb=[pics[f"rgb_{y}"] for y in pics["years"]], grids=grids, cover=cover, ndvi=nd)
        print(site, "cover 1985", round(cover.loc[1985], 2), "2024", round(cover.loc[2024], 2), "| Landsat NDVI first", round(nd.iloc[0], 3), "last", round(nd.iloc[-1], 3))

    plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    def frame(year):
        fig = plt.figure(figsize=(13, 8.2))
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.35], wspace=0.28, hspace=0.25)
        for r, site in enumerate(SITES):
            D = data[site]
            avail = [k for k, y in enumerate(D["years"]) if y <= year]
            ax = fig.add_subplot(gs[r, 0])
            if avail:
                k = avail[-1]
                im = D["rgb"][k]
                ax.imshow(stretch(im[..., :3]), extent=(0, 200, 200, 0))
                ax.imshow(veg_overlay(im), extent=(0, 200, 200, 0), interpolation="nearest")
                ax.set_title(f"{SITES[site]}: {D['sensors'][k]}, {D['dates'][k]}", loc="left", fontsize=10)
            else:
                ax.text(0.5, 0.5, "no clear scene yet", ha="center", va="center", transform=ax.transAxes, color=GREY)
                ax.set_title(f"{SITES[site]}: satellite", loc="left", fontsize=10)
            ax.plot([10, 10 + 1000 / 30], [188, 188], color="w", lw=3); ax.text(10 + 500 / 30, 182, "1 km", color="w", ha="center", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            ax = fig.add_subplot(gs[r, 1])
            ax.imshow(D["grids"][year], cmap=CMAP, vmin=-1, vmax=1, interpolation="nearest")
            ax.set_title(f"Lattice driven by the site's rain, {year}", loc="left", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            ax = fig.add_subplot(gs[r, 2])
            c = D["cover"].loc[:year]
            ax.plot(D["cover"].index, 100 * D["cover"].values, color=RED, lw=1, alpha=0.15)
            ax.plot(c.index, 100 * c.values, color=RED, lw=1.8, label="lattice cover (%)")
            ax.plot(year, 100 * c.iloc[-1], "o", color=RED, ms=6)
            ax.set_xlim(1984, 2025); ax.set_ylim(0, 100 * D["cover"].max() * 1.25)
            ax.set_ylabel("lattice cover, %", color=RED)
            ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
            nd = D["ndvi"]; nd_s = nd[nd.index <= year]
            ax2.plot(nd.index, nd.values, "o", color=NAVY, ms=4, alpha=0.15)
            ax2.plot(nd_s.index, nd_s.values, "o-", color=NAVY, ms=4.5, lw=0.8, label="Landsat dry-season NDVI")
            ax2.set_ylim(nd.min() * 0.85, nd.max() * 1.1); ax2.set_ylabel("Landsat NDVI", color=NAVY)
            ax.set_title("Model cover and Landsat greenness", loc="left", fontsize=10)
            if r == 0:
                h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
                ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, loc="upper left")
        fig.patches.append(plt.Rectangle((0.125, 0.038), 0.012, 0.016, transform=fig.transFigure, color=VEG_RGBA[:3], ec="0.5"))
        fig.text(0.14, 0.04, "satellite: vegetation (NDVI 0.05 above the scene's median)", fontsize=9)
        fig.text(0.50, 0.04, "lattice:", fontsize=9)
        for k, (lab, col) in enumerate([("shrubs and trees", "#2e7d32"), ("bare, fertile", "#f2e6cf"), ("degraded", "#b08850")]):
            fig.patches.append(plt.Rectangle((0.54 + 0.11 * k, 0.038), 0.012, 0.016, transform=fig.transFigure, color=col, ec="0.5"))
            fig.text(0.555 + 0.11 * k, 0.04, lab, fontsize=9)
        return fig

    images = []
    for y in YEARS:
        fig = frame(y)
        buf = io.BytesIO(); fig.savefig(buf, dpi=78, bbox_inches="tight"); plt.close(fig)
        images.append(Image.open(buf).convert("RGB").quantize(colors=200, method=Image.Quantize.MEDIANCUT))
    images[0].save(os.path.join(HERE, "figures", "fig15_landsat_animation.gif"), save_all=True, append_images=images[1:] + [images[-1]] * 4,
                   duration=650, loop=0, optimize=True)
    fig = frame(2024); fig.savefig(os.path.join(HERE, "figures", "fig15_landsat_final.png"), dpi=140, bbox_inches="tight"); plt.close(fig)
    print("gif MB", round(os.path.getsize(os.path.join(HERE, "figures", "fig15_landsat_animation.gif")) / 1e6, 1))


if __name__ == "__main__":
    main()
