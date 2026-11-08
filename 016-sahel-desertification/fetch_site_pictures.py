#!/usr/bin/env python3
#-------- fetch_site_pictures.py ----------------------------------------------------
#-----------------------------------------------------------------------------
# Clear true-colour pictures of Gourma and Fakara for the animation.
#
# The dry season (fetch_landsat_sites.py) is right for measuring woody cover but hazy with
# Harmattan dust. October-November, just after the rains, is clearer and the vegetation is
# green against the sand. For each year, the clearest October-November scene over the
# site's 6 x 6 km box (ranked by the scene's cloud cover, clouds masked):
#   1985-2015   Landsat 5 (to 2011) and Landsat 8 (2013-15), 30 m, 200 x 200 pixels
#   2016-2025   Sentinel-2 (harmonised surface reflectance), 10 m, 600 x 600 pixels
#
#   python fetch_site_pictures.py      (Earth Engine account; EE_PROJECT)
#
# Output: data/pictures_<site>.npz (years, dates, sensors, and one rgb array per year)

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SITES = {"gourma": (15.33, -1.55, "EPSG:32630"), "fakara": (13.50, 2.70, "EPSG:32631")}
BOX = 6000


def main():
    import ee
    ee.Initialize(project=os.environ.get("EE_PROJECT", "zinc-interface-468219-a4"))

    def landsat(bands):
        r, g, b = bands
        def f(im):
            qa = im.select("QA_PIXEL")
            ok = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
            return im.select([r, g, b], ["r", "g", "b"]).multiply(0.0000275).add(-0.2).updateMask(ok) \
                .copyProperties(im, ["system:time_start"]).set("cc", im.get("CLOUD_COVER"))
        return f

    def s2(im):
        scl = im.select("SCL")
        ok = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
        return im.select(["B4", "B3", "B2"], ["r", "g", "b"]).multiply(0.0001).updateMask(ok) \
            .copyProperties(im, ["system:time_start"]).set("cc", im.get("CLOUDY_PIXEL_PERCENTAGE"))

    for site, (lat, lon, crs) in SITES.items():
        pt = ee.Geometry.Point([lon, lat])
        x0, y0 = pt.transform(crs, 1).coordinates().getInfo()
        x0 -= BOX / 2; y0 += BOX / 2
        out = {}
        for y in range(1985, 2026):
            if y == 2012:
                continue
            if y <= 2011:
                col = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").filterDate(f"{y}-10-01", f"{y}-12-01").filterBounds(pt).map(landsat(("SR_B3", "SR_B2", "SR_B1"))); px, sensor = 30, "Landsat 5"
            elif y <= 2015:
                col = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterDate(f"{y}-10-01", f"{y}-12-01").filterBounds(pt).map(landsat(("SR_B4", "SR_B3", "SR_B2"))); px, sensor = 30, "Landsat 8"
            else:
                col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate(f"{y}-10-01", f"{y}-12-01").filterBounds(pt).map(s2); px, sensor = 10, "Sentinel-2"
            if col.size().getInfo() == 0:
                continue
            best = ee.Image(col.sort("cc").first())
            n = int(BOX / px)
            arr = ee.data.computePixels({"expression": best.unmask(-9), "fileFormat": "NUMPY_NDARRAY",
                                         "grid": {"dimensions": {"width": n, "height": n},
                                                  "affineTransform": {"scaleX": px, "shearX": 0, "translateX": x0,
                                                                      "shearY": 0, "scaleY": -px, "translateY": y0},
                                                  "crsCode": crs}})
            rgb = np.stack([np.array(arr[b], dtype=np.float32) for b in "rgb"], axis=-1)
            rgb[rgb == -9] = np.nan
            if np.isnan(rgb).mean() > 0.1:
                print(site, y, "too cloudy, skipped", flush=True)
                continue
            date = ee.Date(best.get("system:time_start")).format("YYYY-MM-dd").getInfo()
            out[y] = (date, sensor, rgb)
            print(site, y, date, sensor, flush=True)
        np.savez_compressed(os.path.join(HERE, "data", f"pictures_{site}.npz"),
                            years=np.array(list(out)), dates=np.array([v[0] for v in out.values()]),
                            sensors=np.array([v[1] for v in out.values()]),
                            **{f"rgb_{y}": v[2] for y, v in out.items()})


if __name__ == "__main__":
    main()
