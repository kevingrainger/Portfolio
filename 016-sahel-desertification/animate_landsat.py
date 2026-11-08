#!/usr/bin/env python3
#-------- animate_landsat.py --------------------------------------------------------
#-----------------------------------------------------------------------------
# The lattice beside forty years of real satellite pictures, at two 12 x 12 km Sahel sites
# with woody vegetation and no fields: a wooded valley and shrubland in the Gourma (Mali,
# 15.20 N 1.40 W), and a tiger-bush plateau north of Niamey (Niger, 13.70 N 1.95 E), where
# the bush has been cut for firewood.
#
# Each frame is one year, 1987-2024. For each site:
#   left     the satellite picture of the 12 x 12 km site in true colour: the clearest
#            October-November scene (fetch_site_pictures.py), Landsat at 30 m to 2015 and
#            Sentinel-2 at 20 m from 2016, only scenes covering the whole box, one brightness stretch for all three bands. Bright
#            green marks pixels clearly greener than the scene's own background (NDVI 0.05
#            above its median). Landsat 5's archive over West Africa has gaps, so the latest
#            available scene is shown, labelled with its sensor and date
#   middle   the lattice, one cell per 30 m pixel, started from the 1987 picture's vegetation
#            pattern (its greenest pixels, up to the cover the model supports in that
#            climate; the degraded share scattered among the bare pixels) and driven by the site's CHIRPS rain (rates from joint_fit_1982.py:
#            b = 0.45 x 3-year mean June-September rain / 400 mm, degradation 0.3, 3% seed
#            rain a year, local seed spread delta 0.03). The ground is uneven, from the
#            Copernicus 30 m elevation model: hollows collect run-off and get better site
#            quality; its strength is the weakest that clumps about as well as the best
#            (clumping.py), so patches aren't pinned to the same spots
#   right    the lattice's cover and Landsat's mean dry-season (February-May) NDVI over the
#            box, to that year: the dry season shows the woody cover the lattice models
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

from sahel_ca import Landscape, VEG, EMPTY, quality_field

HERE = os.path.dirname(os.path.abspath(__file__))
NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"
CMAP = ListedColormap(["#b08850", "#f2e6cf", "#2e7d32"])        # degraded, empty, vegetated
SITES = {"gourma_wadi": "Gourma (Mali)", "tigerbush_plateau": "Tiger bush (Niger)"}
B0, D0, SEED_RAIN, N = 0.45, 0.3, 0.03, 400           # 400 x 400 cells of 30 m: a 12 x 12 km site
DELTA, GROUND_LENGTH = 0.03, 8                   # local seed spread; hollows judged against the land within ~240 m
STRENGTHS = [0.25, 0.5, 0.75, 1.0]               # uneven-ground strength: the weakest that clumps about as well as the best
START = 1987                                     # both runs start from that year's satellite picture
VEG_MARGIN = 0.05                      # overlay: NDVI this far above the scene's median
VEG_RGBA = (0.0, 0.9, 0.35, 0.8)       # bright green
YEARS = list(range(START, 2025))


def site_rain(site):
    P = pd.read_csv(os.path.join(HERE, "data", f"rain_jjas_{site}.csv"), index_col=0).squeeze()
    return P.rolling(3, min_periods=1).mean()


def ground_quality(site, strength):
    """Site quality from the elevation model: cells lower than the land around them (within
    about GROUND_LENGTH cells) collect run-off and get better quality. Mean 1, standard
    deviation `strength`, floored at 0."""
    from scipy.ndimage import gaussian_filter
    z = np.load(os.path.join(HERE, "data", f"dem_{site}.npy")).astype(float)
    hollow = gaussian_filter(z, GROUND_LENGTH, mode="reflect") - z          # positive in hollows
    hollow = (hollow - hollow.mean()) / hollow.std()
    return np.maximum(1 + strength * hollow, 0.0)


def picture_ndvi(site, year):
    from clumping import to_grid
    pics = np.load(os.path.join(HERE, "data", f"pictures_{site}.npz"))
    a = to_grid(pics[f"rgb_{year}"].astype(float), N)
    return (a[..., 3] - a[..., 0]) / (a[..., 3] + a[..., 0] + 1e-6)


def run_lattice(site, strength):
    """Start from the real vegetation pattern: a short spin-up in the mid-1980s climate gives
    the cover and the share of degraded ground the model supports; the START picture's
    greenest pixels become vegetated up to that cover, and the degraded share is scattered at
    random among the rest (a picture can't tell degraded from fertile bare ground). Then the site's rain, year by year."""
    p3 = site_rain(site)
    Q = ground_quality(site, strength)
    b0 = float(min(1, B0 * p3.loc[START - 2:START].mean() / 400))
    spin = Landscape(n=N, b=b0, cover=0.5, seed=7, d=D0, f=0.9, delta=DELTA, quality=Q)
    for _ in range(50):
        spin.step(1)
        e = (spin.grid == EMPTY) & (spin.rng.random(spin.grid.shape) < SEED_RAIN); spin.grid[e] = VEG
    rho, deg = spin.cover(), spin.degraded()
    nd = np.nan_to_num(picture_ndvi(site, START), nan=np.nanmedian(picture_ndvi(site, START)))
    order = np.argsort(nd.ravel())[::-1]
    grid = np.zeros(N * N, dtype=np.int8)
    nveg = int(rho * N * N)
    grid[order[:nveg]] = VEG
    bare = order[nveg:]                                   # a picture can't tell degraded from fertile bare ground,
    rng = np.random.default_rng(3)                        # so the degraded share is scattered among the bare pixels
    grid[rng.choice(bare, min(int(deg * N * N), len(bare)), replace=False)] = -1
    land = Landscape(n=N, b=b0, cover=0.5, seed=7, d=D0, f=0.9, delta=DELTA, quality=Q)
    land.grid = grid.reshape(N, N)
    grids, cover = {START: land.grid.copy()}, {START: land.cover()}
    for y in YEARS[1:]:
        land.b = float(min(1, B0 * p3.loc[y] / 400))
        land.step(1)
        e = (land.grid == EMPTY) & (land.rng.random(land.grid.shape) < SEED_RAIN); land.grid[e] = VEG
        grids[y] = land.grid.copy(); cover[y] = land.cover()
    return grids, pd.Series(cover), p3


def choose_strength(site):
    """Uneven ground strong enough to clump like the satellite vegetation, but no stronger:
    the weakest strength whose clumping gap (log ratio, lattice to satellite at the same
    density, clumping.py) is within 0.05 of the best strength's. Stronger ground pins the
    patches to the same spots."""
    from clumping import clump, satellite_clumping
    res = []
    for st in STRENGTHS:
        grids, cover, _ = run_lattice(site, st)
        sat = satellite_clumping(site, {int(k): v for k, v in cover.items()}, n=N)
        yrs = [y for y in sat if y in grids and y > START]
        lat_c = np.mean([clump(grids[y] == VEG) for y in yrs]); sat_c = np.nanmean([sat[y] for y in yrs])
        res.append((st, abs(float(np.log(lat_c / sat_c))), lat_c, sat_c))
        print(site, "strength", st, "lattice clumping", round(lat_c, 2), "satellite", round(sat_c, 2))
    best = min(r[1] for r in res)
    st, _, lat_c, sat_c = [r for r in res if r[1] <= best + 0.05][0]
    return st, lat_c, sat_c


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
        st, lat_c, sat_c = choose_strength(site)
        grids, cover, p3 = run_lattice(site, st)
        nd = pd.Series([float(np.nanmean(x)) for x in d["ndvi"]], index=d["years"])
        data[site] = dict(strength=st, clump=(lat_c, sat_c), years=list(pics["years"]), dates=list(pics["dates"]), sensors=list(pics["sensors"]),
                          rgb=[pics[f"rgb_{y}"] for y in pics["years"]], grids=grids, cover=cover, ndvi=nd)
        print(site, "strength", st, f"cover {START}", round(cover.loc[START], 2), "2024", round(cover.loc[2024], 2), "| Landsat NDVI first", round(nd.iloc[0], 3), "last", round(nd.iloc[-1], 3))

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
                ax.imshow(stretch(im[..., :3]), extent=(0, N, N, 0))
                ax.imshow(veg_overlay(im), extent=(0, N, N, 0), interpolation="nearest")
                ax.set_title(f"{SITES[site]}: {D['sensors'][k]}, {D['dates'][k]}", loc="left", fontsize=10)
            else:
                ax.text(0.5, 0.5, "no clear scene yet", ha="center", va="center", transform=ax.transAxes, color=GREY)
                ax.set_title(f"{SITES[site]}: satellite", loc="left", fontsize=10)
            ax.plot([0.05 * N, 0.05 * N + 2000 / 30], [0.94 * N] * 2, color="w", lw=3); ax.text(0.05 * N + 1000 / 30, 0.91 * N, "2 km", color="w", ha="center", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            ax = fig.add_subplot(gs[r, 1])
            ax.imshow(D["grids"][year], cmap=CMAP, vmin=-1, vmax=1, interpolation="nearest")
            ax.set_title(f"Lattice from the {START} picture: {year}", loc="left", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            ax = fig.add_subplot(gs[r, 2])
            c = D["cover"].loc[:year]
            ax.plot(D["cover"].index, 100 * D["cover"].values, color=RED, lw=1, alpha=0.15)
            ax.plot(c.index, 100 * c.values, color=RED, lw=1.8, label="lattice cover (%)")
            ax.plot(year, 100 * c.iloc[-1], "o", color=RED, ms=6)
            ax.set_xlim(START - 1, 2025); ax.set_ylim(0, 100 * D["cover"].max() * 1.25)
            ax.set_ylabel("lattice cover, %", color=RED)
            ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
            nd = D["ndvi"][D["ndvi"].index >= START]; nd_s = nd[nd.index <= year]
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
