#!/usr/bin/env python3
#-------- fetch_dry_season.py -------------------------------------------------------
#-----------------------------------------------------------------------------
# Dry-season greenness and long rain records for the Sahel grid sites.
#
# August-September greenness is mostly annual grass and follows that year's rain. By
# March-May the grass has died back, and what is still green is mostly shrubs and trees:
# the slow, woody cover the lattice is about (the separation used in Brandt et al. 2016
# and others). For each grid site (fetch_covariates.py) and year:
#
#   dry NDVI   MODIS MOD13Q1 (250 m) mean of the March-May composites, 2001-2025
#   rain       CHIRPS June-September total, 1981-2024, to drive the lattice
#
#   python fetch_dry_season.py      (Earth Engine account; EE_PROJECT)
#
# Output: data/sahel_dry_ndvi.csv, data/sahel_chirps_jjas.csv

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
HALF = 0.1


def main():
    import ee
    ee.Initialize(project=os.environ.get("EE_PROJECT", "zinc-interface-468219-a4"))
    g = pd.read_csv(os.path.join(DATA, "sahel_static.csv"))[["id", "lat", "lon"]]
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Rectangle([r.lon - HALF, r.lat - HALF, r.lon + HALF, r.lat + HALF]), {"id": int(r.id)})
                               for r in g.itertuples()])
    modis = ee.ImageCollection("MODIS/061/MOD13Q1").select("NDVI")
    bands = [modis.filterDate(f"{y}-03-01", f"{y}-06-01").mean().multiply(0.0001).rename(f"d{y}") for y in range(2001, 2026)]
    rows = ee.Image.cat(bands).reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=500, tileScale=4).getInfo()
    w = pd.DataFrame([f["properties"] for f in rows["features"]])
    pd.concat([w[["id", f"d{y}"]].rename(columns={f"d{y}": "dry_ndvi"}).assign(year=y) for y in range(2001, 2026)]) \
        .sort_values(["id", "year"]).to_csv(os.path.join(DATA, "sahel_dry_ndvi.csv"), index=False)
    print("dry NDVI ok", flush=True)
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select("precipitation")
    out = []
    for start in range(1981, 2025, 11):                          # in chunks so each request stays small
        yrs = range(start, min(start + 11, 2025))
        im = ee.Image.cat([chirps.filterDate(f"{y}-06-01", f"{y}-10-01").sum().rename(f"r{y}") for y in yrs])
        rows = im.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=5000, tileScale=4).getInfo()
        w = pd.DataFrame([f["properties"] for f in rows["features"]])
        out += [w[["id", f"r{y}"]].rename(columns={f"r{y}": "rain_jjas"}).assign(year=y) for y in yrs]
        print("rain", start, "ok", flush=True)
    pd.concat(out).sort_values(["id", "year"]).to_csv(os.path.join(DATA, "sahel_chirps_jjas.csv"), index=False)


if __name__ == "__main__":
    main()
