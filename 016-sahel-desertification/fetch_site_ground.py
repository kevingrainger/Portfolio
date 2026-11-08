#!/usr/bin/env python3
#-------- fetch_site_ground.py ------------------------------------------------------
#-----------------------------------------------------------------------------
# Rain and ground for the two animation sites (a wooded valley and shrubland in the Gourma,
# and a tiger-bush plateau north of Niamey), 12 x 12 km:
#   rain        CHIRPS June-September totals, 1981-2024, the mean over a 0.2 x 0.2 deg box
#   elevation   Copernicus DEM GLO-30 (30 m), on the same 12 x 12 km, 30 m grid as the
#               Landsat pictures, for the lattice's uneven ground: water runs into the
#               hollows, so lower-than-surroundings cells get better site quality
#
#   python fetch_site_ground.py      (Earth Engine account; EE_PROJECT)
#
# Output: data/rain_jjas_<site>.csv, data/dem_<site>.npy

import os

import numpy as np
import pandas as pd

from fetch_site_pictures import SITES

HERE = os.path.dirname(os.path.abspath(__file__))
ANIM = ["gourma_wadi", "tigerbush_plateau"]


def main():
    import ee
    ee.Initialize(project=os.environ.get("EE_PROJECT", "zinc-interface-468219-a4"))
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select("precipitation")
    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM").mosaic()
    for site in ANIM:
        lat, lon, crs = SITES[site]
        box = ee.Geometry.Rectangle([lon - 0.1, lat - 0.1, lon + 0.1, lat + 0.1])
        im = ee.Image.cat([chirps.filterDate(f"{y}-06-01", f"{y}-10-01").sum().rename(f"r{y}") for y in range(1981, 2025)])
        v = im.reduceRegion(ee.Reducer.mean(), box, 5000).getInfo()
        pd.Series({y: v[f"r{y}"] for y in range(1981, 2025)}, name="rain_jjas").rename_axis("year").to_csv(
            os.path.join(HERE, "data", f"rain_jjas_{site}.csv"))
        x0, y0 = ee.Geometry.Point([lon, lat]).transform(crs, 1).coordinates().getInfo()
        x0 -= 6000; y0 += 6000
        arr = ee.data.computePixels({"expression": dem.rename("dem"), "fileFormat": "NUMPY_NDARRAY",
                                     "grid": {"dimensions": {"width": 400, "height": 400},
                                              "affineTransform": {"scaleX": 30, "shearX": 0, "translateX": x0,
                                                                  "shearY": 0, "scaleY": -30, "translateY": y0},
                                              "crsCode": crs}})
        z = np.array(arr["dem"], dtype=np.float32)
        np.save(os.path.join(HERE, "data", f"dem_{site}.npy"), z)
        print(site, "rain mean", round(np.mean(list(v.values())), 1), "elevation", round(float(z.min()), 1), "-", round(float(z.max()), 1))


if __name__ == "__main__":
    main()
