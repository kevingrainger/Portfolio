#!/usr/bin/env python3
#-------- verify_smelters.py -------------------------------------------------------
#-----------------------------------------------------------------------------
# Find each copper smelter's furnaces from space before measuring them.
#
# Published smelter coordinates are rough (a town, a company address). For each
# candidate below, the clearest Sentinel-2 scenes of 2024-2025 are searched in an
# 8 x 8 km box for hot pixels, using the test of Murphy et al. (2016), which bright
# desert, salt and clay never pass but furnaces, slag and flares do:
#     B12 / B11 >= 1.4   and   B12 / B8A >= 1.4   and   B12 >= 0.15
# A site is VERIFIED when at least 2 hot pixels turn up in two or more of its scenes;
# its coordinates are then moved to the centre of the hottest cluster. VERIFIED means
# only that something hot runs within 4 km: the closed Tuticorin plant, kept as a
# negative control, also lights up (from neighbouring industry), so a site's hot spot
# must still be matched to the smelter's buildings on imagery before it is trusted.
# Tuticorin is never measured. Nothing hot
# means a wrong location, a closed plant (Tuticorin, shut since 2018, is included as a
# check) or furnaces hidden under a roof; those rows stay unverified and are not used.
#
# Sentinel-2 comes from Microsoft Planetary Computer, read without an account.
#
#   python verify_smelters.py
#
# Output: smelters.csv (name, country, lat, lon, verified, hot pixels, scenes, source)

import json
import os
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"
HALF = 4000
N_SCENES = 4

# name: (country, approximate lat, lon), from company and press descriptions of each site
CANDIDATES = {
    "Chuquicamata (Codelco)":          ("Chile", -22.300, -68.905),
    "Altonorte (Glencore)":            ("Chile", -23.790, -70.335),
    "Potrerillos (Codelco)":           ("Chile", -26.433, -69.457),
    "Hernan Videla Lira (ENAMI)":      ("Chile", -27.393, -70.272),
    "Chagres (Anglo American)":        ("Chile", -32.790, -70.940),
    "Ventanas (Codelco)":              ("Chile", -32.750, -71.480),
    "Caletones (Codelco)":             ("Chile", -34.110, -70.450),
    "Ilo (Southern Copper)":           ("Peru", -17.555, -71.363),
    "Guixi (Jiangxi Copper)":          ("China", 28.285, 117.180),
    "Tongling Jinguan":                ("China", 30.900, 117.830),
    "Daye Nonferrous":                 ("China", 30.090, 114.960),
    "Xiangguang Copper (Yanggu)":      ("China", 36.110, 115.770),
    "Guangxi Jinchuan (Fangchenggang)": ("China", 21.650, 108.400),
    "Yunnan Copper (Kunming)":         ("China", 25.030, 102.620),
    "Saganoseki (JX Nippon)":          ("Japan", 33.240, 131.880),
    "Onsan (LS MnM)":                  ("South Korea", 35.430, 129.350),
    "Gresik (Smelting)":               ("Indonesia", -7.150, 112.650),
    "Pirdop (Aurubis)":                ("Bulgaria", 42.680, 24.180),
    "Huelva (Atlantic Copper)":        ("Spain", 37.245, -6.945),
    "Hamburg (Aurubis)":               ("Germany", 53.520, 10.020),
    "Glogow (KGHM)":                   ("Poland", 51.665, 16.075),
    "Ronnskar (Boliden)":              ("Sweden", 64.680, 21.255),
    "Mount Isa (Glencore)":            ("Australia", -20.730, 139.480),
    "Olympic Dam (BHP)":               ("Australia", -30.440, 136.880),
    "Garfield (Rio Tinto Kennecott)":  ("USA", 40.720, -112.200),
    "Miami (Freeport)":                ("USA", 33.410, -110.870),
    "Balkhash (Kazakhmys)":            ("Kazakhstan", 46.840, 74.970),
    "Kansanshi (First Quantum)":       ("Zambia", -12.100, 26.430),
    "Mufulira (Mopani)":               ("Zambia", -12.550, 28.240),
    "Tuticorin (Sterlite, closed 2018)": ("India", 8.790, 78.130),
}


def post_json(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def scenes(lat, lon):
    body = {"collections": ["sentinel-2-l2a"], "intersects": {"type": "Point", "coordinates": [lon, lat]},
            "datetime": "2024-01-01/2025-12-31", "limit": 100, "query": {"eo:cloud_cover": {"lt": 5}},
            "sortby": [{"field": "eo:cloud_cover", "direction": "asc"}]}
    return post_json(STAC, body)["features"][:N_SCENES]


def read_box(href, token, lat, lon):
    import rasterio
    from rasterio.warp import transform
    from rasterio.windows import from_bounds
    with rasterio.open(f"{href}?{token}") as src:
        x, y = transform("EPSG:4326", src.crs, [lon], [lat])
        win = from_bounds(x[0] - HALF, y[0] - HALF, x[0] + HALF, y[0] + HALF, src.transform)
        a = src.read(1, window=win, boundless=True, fill_value=0).astype(float)
        T = src.window_transform(win)
        crs = src.crs
    return a, T, crs


def hot_mask(scene, token, lat, lon):
    offset = 1000 if scene["properties"].get("s2:processing_baseline", "00.00") >= "04.00" else 0
    b11, T, crs = read_box(scene["assets"]["B11"]["href"], token, lat, lon)
    b12, _, _ = read_box(scene["assets"]["B12"]["href"], token, lat, lon)
    b8a, _, _ = read_box(scene["assets"]["B8A"]["href"], token, lat, lon)
    b8a, b11, b12 = [np.maximum((b - offset) / 1e4, 1e-4) for b in (b8a, b11, b12)]
    return (b12 / b11 >= 1.4) & (b12 / b8a >= 1.4) & (b12 >= 0.15), T, crs


def main():
    from rasterio.warp import transform
    from scipy.ndimage import label
    token = json.loads(urllib.request.urlopen(TOKEN, timeout=60).read())["token"]
    rows = []
    for name, (country, lat, lon) in CANDIDATES.items():
        hits, total, best = 0, 0, None
        for sc in scenes(lat, lon):
            try:
                m, T, crs = hot_mask(sc, token, lat, lon)
            except Exception as e:
                print(f"  {name}: {sc['id']} unreadable ({e})")
                continue
            n = int(m.sum())
            total += n
            hits += n >= 2
            if n >= 2 and (best is None or n > best[0]):
                lab, k = label(m)
                big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
                r, c = np.argwhere(lab == big).mean(axis=0)
                px, py = T * (c + 0.5, r + 0.5)
                lo, la = transform(crs, "EPSG:4326", [px], [py])
                best = (n, la[0], lo[0])
        ok = hits >= 2 and "closed" not in name
        rows.append(dict(name=name, country=country, lat=round(best[1], 4) if ok else lat,
                         lon=round(best[2], 4) if ok else lon, verified="yes" if ok else "no",
                         hot_pixels=total, scenes_hot=hits, source="Sentinel-2 SWIR hot spot" if ok else "unconfirmed"))
        print(f"{name:36s} {'VERIFIED' if ok else 'not seen':9s} hot pixels {total:5d} in {hits} scenes")
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "smelters.csv"), index=False)


if __name__ == "__main__":
    main()
