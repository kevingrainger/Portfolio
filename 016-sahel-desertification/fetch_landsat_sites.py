#!/usr/bin/env python3
#-------- fetch_landsat_sites.py ----------------------------------------------------
#-----------------------------------------------------------------------------
# Forty years of Landsat at Gourma and Fakara, one picture a year, for the animation.
#
# For each site, a 6 x 6 km box at 30 m (200 x 200 pixels). For each year 1985-2024, all
# clear dry-season (February-May) Landsat views are ranked by cloud cover and the clearest
# single scene is used (a median of several part-cloudy views leaves seams): Landsat 5 to
# 2011, Landsat 8 and 9 from 2013. Two products from the same pixels:
#   true colour   red, green and blue surface reflectance, what the land looks like
#   NDVI          (near-infrared - red) / (near-infrared + red), the greenness
# The dry season shows the shrubs and trees, the woody cover the lattice models, without
# the grass. 2012 has only Landsat 7, whose scan-line corrector failed in 2003 and leaves
# stripes, so it is skipped; so are years with no clear view. Clouds and shadows are
# masked with QA_PIXEL. Landsat 5 and 8 see colour and NDVI slightly differently, so the
# 2011-2013 step is partly the sensor.
#
#   python fetch_landsat_sites.py      (Earth Engine account; EE_PROJECT)
#
# Output: data/landsat_<site>.npz (years, dates, ndvi[year, y, x], rgb[year, y, x, 3])

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SITES = {"gourma": (15.33, -1.55, "EPSG:32630"), "fakara": (13.50, 2.70, "EPSG:32631"),
         "agoufou": (15.34, -1.48, "EPSG:32630"), "tigerbush": (13.60, 2.05, "EPSG:32631")}   # the last two: the animation sites
YEARS = [y for y in range(1985, 2025) if y != 2012]
PX, SIZE = 30, 200


def main():
    import ee
    ee.Initialize(project=os.environ.get("EE_PROJECT", "zinc-interface-468219-a4"))

    def prep(bands):
        nir, red, green, blue = bands
        def f(im):
            qa = im.select("QA_PIXEL")
            ok = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
            sr = im.select([nir, red, green, blue]).multiply(0.0000275).add(-0.2)
            nd = sr.normalizedDifference([nir, red]).rename("ndvi")
            return sr.select([red, green, blue], ["r", "g", "b"]).addBands(nd).updateMask(ok)
        return f

    L5 = ("SR_B4", "SR_B3", "SR_B2", "SR_B1")
    L89 = ("SR_B5", "SR_B4", "SR_B3", "SR_B2")
    for site, (lat, lon, crs) in SITES.items():
        path = os.path.join(HERE, "data", f"landsat_{site}.npz")
        if os.path.exists(path):
            continue
        pt = ee.Geometry.Point([lon, lat])
        x0, y0 = pt.transform(crs, 1).coordinates().getInfo()
        x0 -= SIZE * PX / 2; y0 += SIZE * PX / 2
        nd_stack, rgb_stack, got, dates = [], [], [], []
        for y in YEARS:
            if y <= 2011:
                raw = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").filterDate(f"{y}-02-01", f"{y}-06-01").filterBounds(pt)
                col = raw.map(lambda im: prep(L5)(im).copyProperties(im, ["CLOUD_COVER", "system:time_start"]))
            else:
                raw = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")) \
                    .filterDate(f"{y}-02-01", f"{y}-06-01").filterBounds(pt)
                col = raw.map(lambda im: prep(L89)(im).copyProperties(im, ["CLOUD_COVER", "system:time_start"]))
            n = raw.size().getInfo()
            if n == 0:
                print(site, y, "no images", flush=True)
                continue
            img = ee.Image(col.sort("CLOUD_COVER").first()).unmask(-9)
            arr = ee.data.computePixels({"expression": img, "fileFormat": "NUMPY_NDARRAY",
                                         "grid": {"dimensions": {"width": SIZE, "height": SIZE},
                                                  "affineTransform": {"scaleX": PX, "shearX": 0, "translateX": x0,
                                                                      "shearY": 0, "scaleY": -PX, "translateY": y0},
                                                  "crsCode": crs}})
            nd = np.array(arr["ndvi"], dtype=float); nd[nd == -9] = np.nan
            rgb = np.stack([np.array(arr[b], dtype=float) for b in "rgb"], axis=-1); rgb[rgb == -9] = np.nan
            if np.isnan(nd).mean() > 0.2:
                print(site, y, f"{n} views, {np.isnan(nd).mean():.0%} missing: skipped", flush=True)
                continue
            date = ee.Date(ee.Image(col.sort("CLOUD_COVER").first()).get("system:time_start")).format("YYYY-MM-dd").getInfo()
            nd_stack.append(nd); rgb_stack.append(rgb); got.append(y); dates.append(date)
            print(site, y, f"{n} views", flush=True)
        np.savez_compressed(path, years=np.array(got), dates=np.array(dates), ndvi=np.array(nd_stack, dtype=np.float32),
                            rgb=np.array(rgb_stack, dtype=np.float32))


if __name__ == "__main__":
    main()
