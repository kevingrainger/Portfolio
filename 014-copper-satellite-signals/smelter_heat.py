#!/usr/bin/env python3
#-------- smelter_heat.py ----------------------------------------------------------
#-----------------------------------------------------------------------------
# How hot each copper smelter runs, from three satellite products on Google Earth Engine.
# All the image work happens on Google's side; only a small table per site comes back.
#
#   Sentinel-2 surface reflectance, 20 m, every ~5 days (COPERNICUS/S2_SR_HARMONIZED).
#     A pixel is "hot" when shortwave infrared rises steeply from band 8A (0.86 um) and
#     band 11 (1.6 um) to band 12 (2.2 um), which only happens above a few hundred
#     degrees C (Murphy et al. 2016; the same test as verify_smelters.py):
#         B12 / B11 >= 1.4   and   B12 / B8A >= 1.4   and   B12 >= 0.15
#     Recorded per pass: hot-pixel count, and the share of the site seen cloud-free.
#   Landsat 8 and 9 surface temperature, 100 m, ~8 days combined (LANDSAT/LC0[89]/C02/T1_L2,
#     band ST_B10). Recorded: mean temperature inside the site minus a 3-6 km ring around it.
#   FIRMS active fire, MODIS 1 km, daily, near real time (FIRMS). Recorded: the number of
#     fire pixels (confidence >= 50) touching the site, and their brightest T21.
#
# Smelter sites come from smelters.csv, written by verify_smelters.py: each is a 1.5 km
# circle around the hot spot found there. Rows with verified != yes are skipped.
#
# One-time setup (Kevin): a Google account registered for non-commercial Earth Engine
# use, with a Cloud project, then
#   pip install earthengine-api
#   earthengine authenticate
#   set EE_PROJECT=<your cloud project id>
#
#   python smelter_heat.py 2019-01-01 2026-10-01
#
# Output: data/s2_hot.csv, data/landsat_dT.csv, data/firms.csv

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SITE_RADIUS = 1500                                   # m
RING = (3000, 6000)                                  # m, background for Landsat


def sites_fc(ee, sites):
    feats = [ee.Feature(ee.Geometry.Point([r.lon, r.lat]).buffer(SITE_RADIUS), {"site": r.name})
             for r in sites.itertuples()]
    return ee.FeatureCollection(feats)


def sentinel2(ee, fc, start, end):
    def per_image(img):
        b8a = img.select("B8A").divide(1e4).max(1e-4)
        b11 = img.select("B11").divide(1e4).max(1e-4)
        b12 = img.select("B12").divide(1e4)
        hot = b12.divide(b11).gte(1.4).And(b12.divide(b8a).gte(1.4)).And(b12.gte(0.15)).rename("hot")
        scl = img.select("SCL")
        clear = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).rename("clear")
        stats = hot.addBands(clear).reduceRegions(fc, ee.Reducer.sum().combine(ee.Reducer.count(), sharedInputs=True), 20)
        return stats.map(lambda f: f.set("date", img.date().format("YYYY-MM-dd")))
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(fc.geometry()).filterDate(start, end))
    return col.map(per_image).flatten()


def landsat(ee, fc, start, end):
    def to_kelvin(img):
        return img.select("ST_B10").multiply(0.00341802).add(149.0).rename("T").copyProperties(img, ["system:time_start"])
    col = (ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").merge(ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"))
           .filterBounds(fc.geometry()).filterDate(start, end).map(to_kelvin))

    def per_image(img):
        img = ee.Image(img)
        def per_site(f):
            c = f.geometry().centroid()
            ring = c.buffer(RING[1]).difference(c.buffer(RING[0]))
            inside = img.reduceRegion(ee.Reducer.mean(), f.geometry(), 100).get("T")
            outside = img.reduceRegion(ee.Reducer.median(), ring, 100).get("T")
            return f.set({"T_site": inside, "T_ring": outside, "date": img.date().format("YYYY-MM-dd")})
        return fc.filterBounds(img.geometry()).map(per_site)
    return col.map(per_image).flatten()


def firms(ee, fc, start, end):
    def per_image(img):
        fire = img.select("confidence").gte(50).rename("fire")
        t21 = img.select("T21").updateMask(fire).rename("T21")
        stats = fire.addBands(t21).reduceRegions(fc, ee.Reducer.sum().combine(ee.Reducer.max(), sharedInputs=True), 1000)
        return stats.map(lambda f: f.set("date", img.date().format("YYYY-MM-dd")))
    return ee.ImageCollection("FIRMS").filterBounds(fc.geometry()).filterDate(start, end).map(per_image).flatten()


def to_frame(ee, fc):
    """Pull a FeatureCollection back as a table. getInfo is capped at 5,000 features, so
    the caller works in yearly chunks."""
    feats = fc.getInfo()["features"]
    return pd.DataFrame([f["properties"] for f in feats])


def main():
    import ee
    ee.Initialize(project=os.environ.get("EE_PROJECT"))
    start, end = sys.argv[1:3]
    sites = pd.read_csv(os.path.join(HERE, "smelters.csv"))
    sites = sites[sites["verified"].astype(str).str.lower() == "yes"].set_index("name")
    if sites.empty:
        sys.exit("No verified sites in smelters.csv yet: check each furnace location on imagery first.")
    fc = sites_fc(ee, sites)
    os.makedirs(DATA, exist_ok=True)
    years = pd.date_range(start, end, freq="YS").strftime("%Y-%m-%d").tolist() + [end]
    for name, fn in [("s2_hot", sentinel2), ("landsat_dT", landsat), ("firms", firms)]:
        path = os.path.join(DATA, f"{name}.csv")
        parts = []
        for a, b in zip(years[:-1], years[1:]):
            df = to_frame(ee, fn(ee, fc, a, b))
            parts.append(df)
            print(f"{name} {a[:4]}: {len(df)} site-passes")
        pd.concat(parts).to_csv(path, index=False)


if __name__ == "__main__":
    main()
