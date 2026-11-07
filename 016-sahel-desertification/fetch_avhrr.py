#!/usr/bin/env python3
#-------- fetch_avhrr.py ------------------------------------------------------------
#-----------------------------------------------------------------------------
# Dry-season greenness back to the 1980s droughts.
#
# MODIS starts in 2000, after the droughts. NOAA's AVHRR NDVI climate data record (CDR v5,
# daily, 0.05 deg, Earth Engine NOAA/CDR/AVHRR/NDVI/V5) runs from 1981 to May 2022. For
# each Sahel grid site and year, the mean March-May NDVI over the site's 0.2 x 0.2 deg box,
# with cloudy and cloud-shadow days masked (QA bits 1 and 2), 1982-2022.
#
#   python fetch_avhrr.py      (Earth Engine account; EE_PROJECT)
#
# Output: data/sahel_avhrr_dry.csv

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
    col = ee.ImageCollection("NOAA/CDR/AVHRR/NDVI/V5")

    def clean(im):
        qa = im.select("QA")
        clear = qa.bitwiseAnd(1 << 1).eq(0).And(qa.bitwiseAnd(1 << 2).eq(0))
        nd = im.select("NDVI")
        return nd.updateMask(clear.And(nd.gt(-1000))).multiply(0.0001)

    out = []
    for start in range(1982, 2023, 7):
        yrs = range(start, min(start + 7, 2023))
        im = ee.Image.cat([col.filterDate(f"{y}-03-01", f"{y}-06-01").map(clean).mean().rename(f"a{y}") for y in yrs])
        rows = im.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=5000, tileScale=4).getInfo()
        w = pd.DataFrame([f["properties"] for f in rows["features"]])
        out += [w[["id", f"a{y}"]].rename(columns={f"a{y}": "avhrr_dry"}).assign(year=y) for y in yrs if f"a{y}" in w]
        print(start, "ok", flush=True)
    pd.concat(out).sort_values(["id", "year"]).to_csv(os.path.join(DATA, "sahel_avhrr_dry.csv"), index=False)


if __name__ == "__main__":
    main()
