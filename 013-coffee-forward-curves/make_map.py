#!/usr/bin/env python3
#-------- make_map.py --------------------------------------------------------------
#-----------------------------------------------------------------------------
# Interactive map of the coffee belt for the website: the eight regions of the frost
# index, sized by their share of output and coloured by the ERA5 minimum temperature
# on each of the worst frost nights since 1990 (one switchable layer per night).
#
#   python make_map.py                 # writes maps/coffee_belt.html

import os

import folium
import matplotlib
import numpy as np
import pandas as pd

from coffee_data import DATA, T0, cold_actual
from coffee_regions import REGIONS

HERE = os.path.dirname(os.path.abspath(__file__))
NIGHTS = 5                                                     # the worst nights, at least 30 days apart


def worst_nights(C, n):
    picked = []
    for d in C.sort_values(ascending=False).index:
        if all(abs((d - p).days) > 30 for p in picked):
            picked.append(d)
        if len(picked) == n:
            break
    return sorted(picked)


def main():
    e = pd.read_csv(os.path.join(DATA, "weather_era5.csv"), parse_dates=["time"])
    C = cold_actual()
    nights = worst_nights(C, NIGHTS)
    cmap = matplotlib.colormaps["RdPu_r"]
    m = folium.Map(location=[-20.2, -45.0], zoom_start=6, tiles=None, control_scale=True, min_zoom=5, max_zoom=10)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                     attr="Tiles &copy; Esri, HERE, Garmin, OpenStreetMap contributors", name="light grey base", max_zoom=10).add_to(m)
    for k, d in enumerate(nights):
        night = e[e.time == d].set_index("region").temperature_2m_min
        layer = folium.FeatureGroup(name=f"{d:%d %b %Y}: frost index {C[d]:.2f}", show=(k == len(nights) - 1))
        for name, (lat, lon, w) in REGIONS.items():
            t = night.get(name, np.nan)
            c = matplotlib.colors.to_hex(cmap(np.clip((t + 1) / 12, 0, 1)))
            folium.CircleMarker([lat, lon], radius=6 + 40 * np.sqrt(w), color="#333333", weight=1, fill=True, fill_color=c,
                                fill_opacity=0.85,
                                tooltip=f"{name}: {w:.0%} of the index; ERA5 minimum {t:.1f} C "
                                        f"({'frost weather' if t < T0 else 'above the 5 C line'})").add_to(layer)
        layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    os.makedirs(os.path.join(HERE, "maps"), exist_ok=True)
    m.save(os.path.join(HERE, "maps", "coffee_belt.html"))
    print("nights:", [f"{d:%Y-%m-%d}" for d in nights])


if __name__ == "__main__":
    main()
