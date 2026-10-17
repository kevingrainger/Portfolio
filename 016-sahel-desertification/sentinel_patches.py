#!/usr/bin/env python3
#-------- sentinel_patches.py ------------------------------------------------------
#-----------------------------------------------------------------------------
# Vegetation patches seen from space, at 10 m, to compare with the lattice.
#
# Sentinel-2 surface reflectance from Microsoft Planetary Computer (free, no account:
# the read token is requested anonymously). For each site, the clearest scene of the
# early dry season (November-December, when grass has dried and the woody patches that
# carry the facilitation story still stand out green) is cut to a 6 x 6 km box, and
#     NDVI = (B08 - B04) / (B08 + B04)
# is thresholded by Otsu's method into vegetated and bare. Patches are 4-connected
# clusters of vegetated pixels, as in the model.
#
#   python sentinel_patches.py
#
# Output: data/s2_<site>.npz (NDVI, mask, scene id), results/s2_patches.csv

import json
import os
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"
SITES = {"gourma": (15.33, -1.55), "fakara": (13.50, 2.70)}
HALF = 3000                                                     # m: a 6 x 6 km box
YEARS = ["2023", "2024"]


def post_json(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def clearest_scene(lat, lon):
    best = None
    for y in YEARS:
        body = {"collections": ["sentinel-2-l2a"], "intersects": {"type": "Point", "coordinates": [lon, lat]},
                "datetime": f"{y}-11-01/{y}-12-31", "limit": 50, "query": {"eo:cloud_cover": {"lt": 2}}}
        for f in post_json(STAC, body)["features"]:
            if best is None or f["properties"]["eo:cloud_cover"] < best["properties"]["eo:cloud_cover"]:
                best = f
    return best


def read_box(href, token, lat, lon):
    import rasterio
    from rasterio.warp import transform
    from rasterio.windows import from_bounds
    with rasterio.open(f"{href}?{token}") as src:
        x, y = transform("EPSG:4326", src.crs, [lon], [lat])
        win = from_bounds(x[0] - HALF, y[0] - HALF, x[0] + HALF, y[0] + HALF, src.transform)
        return src.read(1, window=win).astype(float)


def otsu(v):
    v = v[np.isfinite(v)]
    hist, edges = np.histogram(v, 256)
    p = hist / hist.sum()
    mids = 0.5 * (edges[1:] + edges[:-1])
    w0 = np.cumsum(p)
    m0 = np.cumsum(p * mids)
    between = (m0[-1] * w0 - m0) ** 2 / np.maximum(w0 * (1 - w0), 1e-12)
    return mids[np.argmax(between)]


def main():
    from scipy.ndimage import label
    os.makedirs(DATA, exist_ok=True)
    token = urllib.request.urlopen(TOKEN, timeout=60).read()
    token = json.loads(token)["token"]
    rows = []
    for site, (lat, lon) in SITES.items():
        scene = clearest_scene(lat, lon)
        # scenes processed with baseline 04.00 or later (from January 2022) store reflectance
        # with a +1000 offset, which must come off before any ratio is taken
        offset = 1000 if scene["properties"].get("s2:processing_baseline", "00.00") >= "04.00" else 0
        red = read_box(scene["assets"]["B04"]["href"], token, lat, lon) - offset
        nir = read_box(scene["assets"]["B08"]["href"], token, lat, lon) - offset
        ndvi = (nir - red) / np.maximum(nir + red, 1)
        thr = otsu(ndvi)
        mask = ndvi > thr
        lab, k = label(mask)
        sizes = np.bincount(lab.ravel())[1:]
        np.savez_compressed(os.path.join(DATA, f"s2_{site}.npz"), ndvi=ndvi, mask=mask, sizes=sizes,
                            scene=scene["id"], threshold=thr)
        rows.append(dict(site=site, scene=scene["id"], date=scene["properties"]["datetime"][:10],
                         cloud=scene["properties"]["eo:cloud_cover"], threshold=thr, cover=mask.mean(),
                         n_patches=k, largest_share=sizes.max() / sizes.sum()))
        print(rows[-1])
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "results", "s2_patches.csv"), index=False)


if __name__ == "__main__":
    main()
