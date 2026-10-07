#-------- plots.py ---------------------------------------------------------------------
#-----------------------------------------------------------------------------
# Figures. Kept out of the notebook so that each cell there is one call and the
# notebook reads as an argument rather than as plotting code.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LightSource, ListedColormap, Normalize, TwoSlopeNorm

from .grid import CLASSES, GAMMA

LAND_COLOURS = dict(zip(CLASSES, ["#a4534b", "#e3d59a", "#3f7d4e", "#a9c98a", "#86a9c9", "#bdb7ac"]))
MODEL_COLOURS = {"B0": "#9a9a9a", "B1": "#c2b280", "B2": "#7f6a9e", "B3": "#5a9bd4", "M1": "#a8d5ba", "M2": "#3f9b7a", "M3": "#14532d"}
TEMP = sns.color_palette("rocket_r", as_cmap=True)
DIVERGING = sns.color_palette("vlag", as_cmap=True)


def land_cover_image(frac):
    """Blend the class colours by their fractions: mixed cells show as mixed colours."""
    rgb = np.array([plt.matplotlib.colors.to_rgb(LAND_COLOURS[c]) for c in CLASSES])
    return np.tensordot(frac, rgb, axes=([0], [0]))


def hillshade(elev_km, dx, strength=1.0):
    return LightSource(azdeg=315, altdeg=45).hillshade(elev_km * 1000, vert_exag=6 * strength, dx=dx * 1000, dy=dx * 1000)


def shaded(values, elev_km, dx, cmap, norm, blend=0.45):
    """Colour a field and darken it with the terrain's shadow."""
    rgb = cmap(norm(values))[..., :3]
    shade = hillshade(elev_km, dx)
    flat = np.median(shade[elev_km < 0.05]) if (elev_km < 0.05).any() else 0.7        # level ground keeps its colour
    return np.clip(rgb * (1 - blend + blend * (shade / flat)[..., None]), 0, 1)


def map_axes(ax, grid, title=None):
    ax.set_aspect("equal"); ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, loc="left")


def scale_bar(ax, grid, km=100):
    x0, y0 = grid.x0 + 12, grid.y0 + 9
    ax.plot([x0, x0 + km], [y0, y0], color="k", lw=1.5, solid_capstyle="butt", scalex=False, scaley=False)
    ax.annotate(f"{km} km", (x0 + km + 5, y0), va="center", fontsize=8)


def colourbar(fig, ax, cmap, norm, label, **kw):
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03, aspect=40, **kw)
    cb.set_label(label); cb.outline.set_visible(False)
    return cb


#-------- Step 1: where the stations are ---------------------------------------------------
def stations(ds, fold=None):
    g = ds.grid
    fig = plt.figure(figsize=(12, 4.6))
    ax = fig.add_axes([0.0, 0.02, 0.62, 0.9]); hx = fig.add_axes([0.70, 0.16, 0.28, 0.68])
    terrain = sns.color_palette("blend:#f4f1e8,#cdbf9c,#8d7a5b,#5b4a3a", as_cmap=True)
    img = shaded(np.where(ds.sea, np.nan, ds.elev), ds.elev, g.dx, terrain, Normalize(0, 3.2), 0.5)
    img[ds.sea | (ds.frac[CLASSES.index("water")] > 0.5)] = plt.matplotlib.colors.to_rgb("#c9d9e6")
    ax.imshow(img, origin="lower", extent=g.extent, interpolation="bilinear")
    st = ds.stations
    marks = ListedColormap(sns.color_palette("mako_r", 256)[70:])
    sc = ax.scatter(st.x, st.y, c=st.elevation, cmap=marks, norm=Normalize(0, 2500), s=46, edgecolor="white", linewidth=0.9, zorder=3)
    map_axes(ax, g, f"{len(st)} stations with a 2021-2023 record (Meteostat list)"); scale_bar(ax, g)
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.045, pad=0.02, aspect=45)
    cb.set_label("station elevation (m)"); cb.outline.set_visible(False)

    bins = np.arange(0, 3600, 200)
    land = ~ds.sea
    hx.hist(ds.elev[land] * 1000, bins=bins, weights=np.full(land.sum(), 100 / land.sum()), color="#cdbf9c", label="land area")
    hx.hist(st.elevation, bins=bins, weights=np.full(len(st), 100 / len(st)), histtype="step", color="#1f4e5f", lw=1.8, label="stations")
    hx.set_xlabel("elevation (m)"); hx.set_ylabel("share (%)")
    hx.set_title("Stations sit low", loc="left")
    hx.legend(loc="upper right")
    return fig


#-------- Step 0: recovery of planted parameters -----------------------------------------------
def recovery(tables, labels):
    fig, axes = plt.subplots(1, len(tables), figsize=(5.4 * len(tables), 4.2), sharey=True)
    for ax, t, label in zip(np.atleast_1d(axes), tables, labels):
        rows = [r for r in t.index if r.startswith("tau ") and r.endswith("(h)")] + ["tau_a (h)", "kappa (m^2/s)"]
        ratio = t.loc[rows, "fitted / true"]
        share = t.loc[rows, "station land-cover share"].fillna(1.0)
        y = np.arange(len(rows))[::-1]
        ax.axvline(1, color="0.3", lw=1)
        ax.axvspan(0.5, 2, color="0.93", zorder=0)
        ax.scatter(ratio, y, s=30 + 260 * share.clip(0, 0.5), color="#1f4e5f", zorder=3)
        ax.set_xscale("log"); ax.set_xlim(0.07, 14)
        ax.set_xticks([0.1, 0.5, 1, 2, 10]); ax.set_xticklabels(["0.1", "0.5", "1", "2", "10"])
        ax.set_yticks(y); ax.set_yticklabels([r.replace(" (h)", "").replace(" (m^2/s)", "").replace("tau_a", "tau atmosphere") for r in rows])
        ax.set_xlabel("fitted / true"); ax.set_title(label, loc="left")
    return fig


#-------- the heatwave day ------------------------------------------------------------------------
def terrain_surface(ax, grid, elev_km, values, cmap, norm, stride=2, exaggeration=14.0, shade=0.55):
    """A 3D terrain with a field draped over it."""
    X, Y = np.meshgrid(grid.x[::stride], grid.y[::stride])
    Z = elev_km[::stride, ::stride]
    rgb = cmap(norm(values[::stride, ::stride]))[..., :3]
    light = LightSource(azdeg=300, altdeg=40).hillshade(Z * 1000, vert_exag=5, dx=grid.dx * stride * 1000, dy=grid.dx * stride * 1000)
    rgb = np.clip(rgb * (1 - shade + shade * (light / np.median(light))[..., None]), 0, 1)
    ax.plot_surface(X, Y, Z * exaggeration, facecolors=rgb, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ax.set_box_aspect((grid.nx, grid.ny, 0.16 * grid.nx)); ax.set_axis_off()
    ax.set_zlim(0, elev_km.max() * exaggeration)


def temperature_surface(ax, grid, theta, cells_xy, y, cmap, norm, title, stride=2, zlim=None, hide=None):
    """The PDE's own surface: sea-level temperature as height, stations as pins.
    A pin's head is the reading; the surface either reaches it or it does not."""
    X, Y = np.meshgrid(grid.x[::stride], grid.y[::stride])
    Z = np.where(hide, np.nan, theta)[::stride, ::stride] if hide is not None else theta[::stride, ::stride]
    colours = cmap(norm(np.nan_to_num(Z, nan=norm.vmin)))
    ax.plot_surface(X, Y, Z, facecolors=colours, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=True,
                    lightsource=LightSource(azdeg=300, altdeg=55))
    ix, iy, ok = cells_xy
    for x_, y_, obs, model in zip(grid.x[ix[ok]], grid.y[iy[ok]], y[ok], theta[iy[ok], ix[ok]]):
        ax.plot([x_, x_], [y_, y_], [min(obs, model), max(obs, model)], color="k", lw=1.0, zorder=10)
    ax.scatter(grid.x[ix[ok]], grid.y[iy[ok]], y[ok], color="k", s=14, depthshade=False, zorder=11)
    ax.set_zlim(*zlim); ax.set_box_aspect((grid.nx, grid.ny, 0.42 * grid.nx), zoom=1.14)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zlabel("sea-level temperature (°C)")
    ax.xaxis.pane.set_visible(False); ax.yaxis.pane.set_visible(False); ax.zaxis.pane.set_visible(False)
    ax.grid(False); ax.set_title(title, loc="left", y=0.98)


#-------- skill --------------------------------------------------------------------------------------
def skill_by_band(tables, titles, models):
    fig, axes = plt.subplots(1, len(tables), figsize=(5.6 * len(tables), 4.0), sharey=True)
    for ax, t, title in zip(axes, tables, titles):
        groups = list(dict.fromkeys(t.group))
        width = 0.8 / len(models)
        for j, m in enumerate(models):
            v = t[t.model == m].set_index("group").rmse.reindex(groups)
            ax.bar(np.arange(len(groups)) + (j - len(models) / 2 + 0.5) * width, v, width, color=MODEL_COLOURS[m], label=m)
        counts = t[t.model == models[0]].set_index("group").n.reindex(groups)
        ax.set_xticks(np.arange(len(groups))); ax.set_xticklabels([f"{g}\n(n = {int(c)})" for g, c in zip(groups, counts)])
        ax.set_title(title, loc="left"); ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("RMSE at held-out stations (K)")
    axes[-1].legend(ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5))
    return fig


#-------- figures built from results/maps.npz ----------------------------------------------------
def cover(ds, maps, width=13.0, height=6.5):
    """The hottest July 2022 day: downscaled Tmax draped over the terrain."""
    from scipy import ndimage
    d = int(maps["days"][0])
    tmax = maps["hot_M3"] - GAMMA * ds.elev
    norm = Normalize(16, 40)
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([-0.19, -0.50, 1.38, 1.86], projection="3d")
    cmap = TEMP.copy(); cmap.set_bad("#c9d9e6")
    relief = ndimage.gaussian_filter(ds.elev, 1.2)                      # drawn slightly smoothed; the colours are not
    terrain_surface(ax, ds.grid, relief, np.ma.masked_where(ds.sea, tmax), cmap, norm, stride=2, exaggeration=11.0, shade=0.5)
    ax.view_init(elev=44, azim=-78)
    cax = fig.add_axes([0.72, 0.905, 0.24, 0.022])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=TEMP), cax=cax, orientation="horizontal")
    cb.set_label("daily maximum temperature (°C)"); cb.outline.set_visible(False)
    fig.text(0.035, 0.925, f"Daily maximum temperature at 1 km, {ds.days[d].strftime('%d %B %Y').lstrip('0')}", fontsize=15)
    fig.text(0.035, 0.882, "Po Valley from the south. A heat-budget equation constrained to equal every station reading.", fontsize=10.5, color="0.3")
    return fig


def pde_surfaces(ds, maps):
    """Sea-level temperature as a surface, before and after anchoring, with stations as pins."""
    d = int(maps["days"][0])
    y = ds.y[d]
    ok = np.isfinite(y)
    cells = (ds.stations.ix.values, ds.stations.iy.values, ok)
    fields = [("Physics only: the surface misses the pins", maps["hot_M1"]),
              ("Anchored: the surface passes through every pin", maps["hot_M3"])]
    land = ~ds.sea
    lo = min(np.nanmin(y), min(f[land].min() for _, f in fields)) - 0.3
    hi = max(np.nanmax(y), max(f[land].max() for _, f in fields)) + 0.3
    norm = Normalize(lo, hi)
    fig = plt.figure(figsize=(13, 5.6))
    for i, (title, f) in enumerate(fields):
        ax = fig.add_axes([0.0 + 0.5 * i, 0.0, 0.5, 0.94], projection="3d")
        temperature_surface(ax, ds.grid, f, cells, y, TEMP, norm, title, stride=2, zlim=(lo, hi), hide=ds.sea)
        ax.view_init(elev=30, azim=-70)
    return fig


def downscaling(ds, maps):
    """What ERA5-Land sees, what the anchored model produces, and (simulation only) the truth."""
    d = int(maps["days"][0])
    panels = [("ERA5-Land, lapse-corrected", maps["hot_theta_E"]), ("Physics, anchored, XGBoost prior (1 km)", maps["hot_M3"])]
    if "truth" in ds._maps:
        panels.append(("Known truth of the simulation", ds.field("truth", d) + GAMMA * ds.elev))
    norm = Normalize(18, 40)
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.6))
    for ax, (title, theta) in zip(axes, panels):
        img = shaded(theta - GAMMA * ds.elev, ds.elev, ds.grid.dx, TEMP, norm, 0.35)
        ax.imshow(img, origin="lower", extent=ds.grid.extent, interpolation="bilinear")
        ok = np.isfinite(ds.y[d])
        ax.scatter(ds.stations.x[ok], ds.stations.y[ok], c=ds.tmax[d][ok], cmap=TEMP, norm=norm, s=26, edgecolor="k", linewidth=0.7)
        map_axes(ax, ds.grid, title)
    colourbar(fig, axes, TEMP, norm, f"daily maximum temperature, {ds.days[d].date()} (°C); circles are stations")
    return fig


def missing_physics(ds, maps):
    """Mean q beside the land-cover map."""
    q = np.where(ds.sea, np.nan, maps["mean_q_m3"])
    lim = float(np.nanquantile(np.abs(q), 0.99))
    norm = TwoSlopeNorm(0, -lim, lim)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    cmap = DIVERGING.copy(); cmap.set_bad("#e9eef2")
    axes[0].imshow(np.ma.masked_invalid(q), origin="lower", extent=ds.grid.extent, cmap=cmap, norm=norm, interpolation="bilinear")
    axes[0].scatter(ds.stations.x, ds.stations.y, s=10, color="k", zorder=3)
    map_axes(axes[0], ds.grid, "Mean missing heating q (dots are stations)")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING), ax=axes[0], orientation="horizontal", fraction=0.05, pad=0.03, aspect=40)
    cb.set_label("K per hour"); cb.outline.set_visible(False)
    axes[1].imshow(land_cover_image(ds.frac), origin="lower", extent=ds.grid.extent, interpolation="bilinear")
    map_axes(axes[1], ds.grid, "Land cover")
    handles = [plt.Rectangle((0, 0), 1, 1, color=LAND_COLOURS[c]) for c in CLASSES]
    axes[1].legend(handles, CLASSES, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.04), handlelength=1.2, columnspacing=1.0)
    return fig


def footprints(ds, maps):
    """Station footprints (rows of G) on the calmest and the windiest day."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.0))
    blues = sns.color_palette("blend:#ffffff,#8f97e0,#2430bc,#0b1150", as_cmap=True)
    st = ds.stations.iloc[maps["footprint_stations"]]
    half = 45
    for ax, key, label in zip(axes, ("still", "windy"), ("Calmest day", "Windiest day")):
        d = int(maps["days"][{"windy": 1, "still": 2}[key]])
        f = maps[f"{key}_footprints"]
        total = (f / f.max(axis=(1, 2), keepdims=True)).max(0)
        base = shaded(np.zeros_like(ds.elev), ds.elev, ds.grid.dx, ListedColormap(["#f3f1ea"]), Normalize(0, 1), 0.35)
        ax.imshow(base, origin="lower", extent=ds.grid.extent)
        ax.imshow(np.ma.masked_less(total, 0.01), origin="lower", extent=ds.grid.extent, cmap=blues, norm=Normalize(0, 0.6), alpha=0.9, interpolation="bilinear")
        u, v = maps[f"{key}_u"], maps[f"{key}_v"]
        s = 24
        ax.quiver(ds.grid.x[s // 2::s], ds.grid.y[s // 2::s], u[s // 2::s, s // 2::s], v[s // 2::s, s // 2::s], color="0.35", scale=420, width=0.0022)
        ax.scatter(st.x, st.y, s=16, color="k", zorder=3)
        ax.set_xlim(st.x.min() - half, st.x.max() + half); ax.set_ylim(st.y.min() - half, st.y.max() + half)
        map_axes(ax, ds.grid, f"{label}: mean wind {maps['speed'][d]:.0f} km/h")
    return fig


def skill_bars(by_elevation, by_distance, models):
    return skill_by_band([by_elevation, by_distance], ["By station elevation", "By distance to the nearest anchor"], models)
