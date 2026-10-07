#!/usr/bin/env python3
#-------- make_maps.py -----------------------------------------------------------------
#-----------------------------------------------------------------------------
# Interactive maps (folium / Leaflet) from the pipeline's results. Each is one
# self-contained HTML file with switchable layers and a marker per station.
#
#   python make_maps.py
#
#   maps/heatwave.html          the hottest July 2022 day: ERA5-Land at its own
#                               resolution against the 1 km anchored surface
#   maps/missing_physics.html   mean q beside land cover, and station footprints on the
#                               calmest and the windiest day

import os
import sys

import branca.colormap as bcm
import folium
import numpy as np
import seaborn as sns
from matplotlib.colors import Normalize, TwoSlopeNorm, to_hex

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tmax1km.data import Dataset
from tmax1km.grid import BBOX, GAMMA
from tmax1km.plots import DIVERGING, TEMP, land_cover_image

OUT = os.path.join(HERE, "maps")
STEP = 0.01                                            # degrees: resolution of the overlays
NOTE = " (simulated test data)"


def to_lonlat(ds, field):
    """Resample a field from the UTM grid to a regular latitude-longitude raster
    (row 0 in the south), NaN outside the domain."""
    lat = np.arange(BBOX["lat0"], BBOX["lat1"], STEP) + STEP / 2
    lon = np.arange(BBOX["lon0"], BBOX["lon1"], STEP) + STEP / 2
    LON, LAT = np.meshgrid(lon, lat)
    iy, ix, inside = ds.grid.cell(LON, LAT)
    out = np.full(LON.shape + field.shape[2:], np.nan)
    out[inside] = field[iy[inside], ix[inside]]
    return out


def overlay(ds, field, name, cmap=None, norm=None, show=True, opacity=0.8, rgb=False):
    a = to_lonlat(ds, field)
    if rgb:
        rgba = np.dstack([a, np.isfinite(a[..., 0]).astype(float)])
    else:
        rgba = cmap(norm(a)); rgba[..., 3] = np.isfinite(a)
    rgba = np.nan_to_num(rgba)
    return folium.raster_layers.ImageOverlay((rgba * 255).astype(np.uint8), [[BBOX["lat0"], BBOX["lon0"]], [BBOX["lat1"], BBOX["lon1"]]],
                                             origin="lower", mercator_project=True, opacity=opacity, name=name, show=show)


def legend(cmap, norm, caption, n=9):
    v = np.linspace(norm.vmin, norm.vmax, n)
    return bcm.LinearColormap([to_hex(cmap(norm(x))) for x in v], index=list(v), vmin=norm.vmin, vmax=norm.vmax, caption=caption)


def base_map(ds):
    m = folium.Map(location=[45.1, 9.9], zoom_start=8, tiles=None, control_scale=True, min_zoom=6, max_zoom=11)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                     attr="Tiles &copy; Esri, HERE, Garmin, OpenStreetMap contributors", name="light grey base", max_zoom=11).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap base", show=False).add_to(m)
    return m


def heatwave(ds, maps):
    d = int(maps["days"][0])
    norm = Normalize(20, 40)
    tmax = maps["hot_M3"] - GAMMA * ds.elev
    era5 = maps["hot_theta_E"] - GAMMA * ds.elev
    m = base_map(ds)
    overlay(ds, np.where(ds.sea, np.nan, tmax), "Tmax at 1 km (physics, anchored, XGBoost prior)", TEMP, norm).add_to(m)
    overlay(ds, np.where(ds.sea, np.nan, era5), "ERA5-Land with lapse-rate correction", TEMP, norm, show=False).add_to(m)
    layer = folium.FeatureGroup("stations", show=True)
    obs = ds.tmax[d]
    for i, s in ds.stations.iterrows():
        if not np.isfinite(obs[i]):
            continue
        model = maps["hot_M3"][s.iy, s.ix] - GAMMA * s.z_km
        before = maps["hot_theta_E"][s.iy, s.ix] - GAMMA * s.z_km
        text = (f"<b>{s['name']}</b> ({s.elevation:.0f} m)<br>station {obs[i]:.1f} °C<br>anchored surface {model:.1f} °C"
                f"<br>ERA5-Land, lapse-corrected {before:.1f} °C")
        folium.CircleMarker([s.latitude, s.longitude], radius=6, color="#1b1b1f", weight=1.2, fill=True, fill_opacity=1,
                            fill_color=to_hex(TEMP(norm(obs[i]))), popup=folium.Popup(text, max_width=260), tooltip=s["name"]).add_to(layer)
    layer.add_to(m)
    legend(TEMP, norm, f"Daily maximum temperature, {ds.days[d].date()} (°C)" + (NOTE if ds.simulated else "")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(os.path.join(OUT, "heatwave.html"))


def missing_physics(ds, maps):
    q = maps["mean_q_m3"]
    lim = float(np.quantile(np.abs(q[~ds.sea]), 0.99))
    norm = TwoSlopeNorm(0, -lim, lim)
    m = base_map(ds)
    overlay(ds, np.where(ds.sea, np.nan, q), "mean q: heating the equation lacks", DIVERGING, norm).add_to(m)
    overlay(ds, land_cover_image(ds.frac), "land cover", rgb=True, show=False, opacity=0.75).add_to(m)
    for key, label in (("still", "calmest"), ("windy", "windiest")):
        d = int(maps["days"][{"windy": 1, "still": 2}[key]])
        f = maps[f"{key}_footprints"].sum(0)
        f = f / f.max()
        rgba_norm = Normalize(0.0, 0.25, clip=True)
        lay = overlay(ds, np.where(f > 0.004, f, np.nan), f"station footprints, {label} day ({maps['speed'][d]:.0f} km/h)",
                      sns.color_palette("blend:#dfe2f7,#2430bc,#0b1150", as_cmap=True), rgba_norm, show=False, opacity=0.85)
        lay.add_to(m)
    layer = folium.FeatureGroup("stations", show=True)
    for _, s in ds.stations.iterrows():
        folium.CircleMarker([s.latitude, s.longitude], radius=4, color="#1b1b1f", weight=1, fill=True, fill_color="#ffffff", fill_opacity=1,
                            tooltip=f"{s['name']} ({s.elevation:.0f} m)").add_to(layer)
    layer.add_to(m)
    legend(DIVERGING, Normalize(-lim, lim), "Mean missing heating q, all days (K per hour)" + (NOTE if ds.simulated else "")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(os.path.join(OUT, "missing_physics.html"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    ds = Dataset(verbose=False)
    maps = np.load(os.path.join(HERE, "results", "maps.npz"))
    heatwave(ds, maps)
    missing_physics(ds, maps)
    for f in sorted(os.listdir(OUT)):
        print(f, os.path.getsize(os.path.join(OUT, f)) // 1024, "kB")
