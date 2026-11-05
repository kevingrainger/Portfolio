#!/usr/bin/env python3
#-------- fetch_covariates.py -------------------------------------------------------
#-----------------------------------------------------------------------------
# Greenness, rain and the pressures on the land for many Sahel sites, not just two.
#
# A grid of sites every 0.5 deg of latitude (13-17 N) and 1 deg of longitude (16 W - 16 E),
# each a 0.2 x 0.2 deg box (about 22 km). For each box:
#
#   greenness   MODIS MOD13Q1 NDVI (250 m), the mean of the August-September composites,
#               each year 2001-2024 (Earth Engine MODIS/061/MOD13Q1)
#   rain        CHIRPS daily rainfall summed over June-October, each year 2001-2024
#               (UCSB-CHG/CHIRPS/DAILY)
#   cropping    share of the box mapped as cropland by ESA WorldCover 2021 at 10 m
#               (ESA/WorldCover/v200, class 40)
#   people      population density in 2000 and 2020, GPW v4.11 (CIESIN/GPWv411)
#   soil        sand fraction and organic carbon of the topsoil, OpenLandMap (Hengl, 250 m)
#   grazing     livestock in 2015 from FAO's Gridded Livestock of the World (GLW4, about
#               9 km): cattle, sheep and goats, as tropical livestock units per km2
#               (cattle 0.7, sheep and goats 0.1). Read locally from data/glw4/
#
#   python fetch_covariates.py      (Earth Engine account; EE_PROJECT)
#
# Output: data/sahel_grid.csv (one row per site and year), data/sahel_static.csv

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
YEARS = range(2001, 2025)
HALF = 0.1


def grid():
    pts = [(round(la, 2), float(lo)) for la in np.arange(13.0, 17.01, 0.5) for lo in np.arange(-16, 16.01, 1.0)]
    return pd.DataFrame(pts, columns=["lat", "lon"]).assign(id=lambda d: range(len(d)))


def ee_tables(g):
    import ee
    ee.Initialize(project=os.environ.get("EE_PROJECT", "zinc-interface-468219-a4"))
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Rectangle([r.lon - HALF, r.lat - HALF, r.lon + HALF, r.lat + HALF]), {"id": int(r.id)})
                               for r in g.itertuples()])
    modis = ee.ImageCollection("MODIS/061/MOD13Q1").select("NDVI")
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select("precipitation")
    bands = []
    for y in YEARS:
        bands.append(modis.filterDate(f"{y}-08-01", f"{y}-10-01").mean().multiply(0.0001).rename(f"ndvi_{y}"))
        bands.append(chirps.filterDate(f"{y}-06-01", f"{y}-11-01").sum().rename(f"rain_{y}"))
    yearly = ee.Image.cat(bands)
    rows = yearly.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=1000, tileScale=4).getInfo()
    yr = pd.DataFrame([f["properties"] for f in rows["features"]])
    crop = ee.Image("ESA/WorldCover/v200/2021").eq(40).rename("crop")
    gpw = ee.ImageCollection("CIESIN/GPWv411/GPW_Population_Density")
    pop00 = gpw.filter(ee.Filter.eq("system:index", "gpw_v4_population_density_rev11_2000_30_sec")).first().rename("pop2000")
    pop20 = gpw.filter(ee.Filter.eq("system:index", "gpw_v4_population_density_rev11_2020_30_sec")).first().rename("pop2020")
    sand = ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02").select("b0").rename("sand")
    soc = ee.Image("OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02").select("b0").multiply(5).rename("soc")   # g/kg
    static = ee.Image.cat([crop, pop00, pop20, sand, soc])
    rows = static.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=250, tileScale=8).getInfo()
    st = pd.DataFrame([f["properties"] for f in rows["features"]])
    return yr, st


def livestock(g):
    import rasterio
    tlu = None
    for f, w in [("5_Ct_2015_Da.tif", 0.7), ("5_Sh_2015_Da.tif", 0.1), ("5_Gt_2015_Da.tif", 0.1)]:
        with rasterio.open(os.path.join(DATA, "glw4", f)) as src:
            vals = []
            for r in g.itertuples():
                win = rasterio.windows.from_bounds(r.lon - HALF, r.lat - HALF, r.lon + HALF, r.lat + HALF, src.transform)
                a = src.read(1, window=win, boundless=True, fill_value=0).astype(float)
                a[(a < 0) | ~np.isfinite(a)] = 0
                px_km2 = (abs(src.res[0]) * 111.32) * (abs(src.res[1]) * 111.32 * np.cos(np.radians(r.lat)))
                vals.append(a.mean() / px_km2)
            tlu = w * np.array(vals) if tlu is None else tlu + w * np.array(vals)
    return tlu


def main():
    g = grid()
    yr, st = ee_tables(g)
    st = g.merge(st, on="id")
    st["tlu_km2"] = livestock(g)
    st.to_csv(os.path.join(DATA, "sahel_static.csv"), index=False)
    long = []
    for y in YEARS:
        long.append(yr[["id", f"ndvi_{y}", f"rain_{y}"]].rename(columns={f"ndvi_{y}": "ndvi", f"rain_{y}": "rain"}).assign(year=y))
    pd.concat(long).sort_values(["id", "year"]).to_csv(os.path.join(DATA, "sahel_grid.csv"), index=False)
    print(st.describe().round(2).to_string())


if __name__ == "__main__":
    main()
