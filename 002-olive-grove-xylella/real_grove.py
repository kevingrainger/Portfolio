#!/usr/bin/env python3
#-------- real_grove.py ------------------------------------------------------------
#-----------------------------------------------------------------------------
# A real Salento olive grove, alive and then dead, for the "next steps": the kind of
# picture a data-assimilation version of the model would be fitted to.
#
# The grove is near Gallipoli (Lecce), about 40.0935 N, 18.0414 E, in the zone where
# Xylella spread from 2013. It is a regular planting of roughly 15 x 20 trees, close to
# the simulated 16 x 16 grove. Imagery: Esri World Imagery Wayback (satellite, about
# 0.5 m), the release of 20 February 2014 and that of 24 September 2026.
#
#   python real_grove.py          # downloads the two images once, then draws the figure
#
# Output: data/satellite/grove_<date>.png, figures/fig_real_grove

import io
import math
import os
import sys
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import figstyle                                # noqa: E402

LAT, LON, HALF_M, ZOOM = 40.09351, 18.04139, 110, 18
RELEASES = {"2014-02": 10, "2026-09": 63116}  # Wayback release numbers (waybackconfig.json)
SAT = os.path.join(HERE, "data", "satellite")
ATTRIB = "Satellite imagery: Esri World Imagery Wayback (source: Esri, Maxar, Earthstar Geographics)"


def pixel(lat, lon, z):
    n = 256 * 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def wayback(release, path):
    """Stitch the Web Mercator tiles covering the box and crop to it."""
    dlat = HALF_M / 111320.0
    dlon = HALF_M / (111320.0 * math.cos(math.radians(LAT)))
    x0, y0 = pixel(LAT + dlat, LON - dlon, ZOOM)
    x1, y1 = pixel(LAT - dlat, LON + dlon, ZOOM)
    tx0, ty0, tx1, ty1 = int(x0 // 256), int(y0 // 256), int(x1 // 256), int(y1 // 256)
    canvas = Image.new("RGB", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256))
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            url = ("https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/"
                   f"default028mm/MapServer/tile/{release}/{ZOOM}/{ty}/{tx}")
            req = urllib.request.Request(url, headers={"User-Agent": "portfolio-research-script"})
            tile = Image.open(io.BytesIO(urllib.request.urlopen(req, timeout=60).read())).convert("RGB")
            canvas.paste(tile, ((tx - tx0) * 256, (ty - ty0) * 256))
    canvas.crop((int(x0 - tx0 * 256), int(y0 - ty0 * 256), int(x1 - tx0 * 256), int(y1 - ty0 * 256))).save(path)


def main():
    os.makedirs(SAT, exist_ok=True)
    figstyle.apply()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.0))
    for ax, (date, rel) in zip(axes, RELEASES.items()):
        path = os.path.join(SAT, f"grove_{date}.png")
        if not os.path.exists(path):
            wayback(rel, path)
        im = np.asarray(Image.open(path))
        ax.imshow(im, extent=[0, 2 * HALF_M, 0, 2 * HALF_M])
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.plot([12, 62], [12, 12], color="white", lw=3)
        ax.text(37, 17, "50 m", color="white", ha="center", fontsize=9)
    axes[0].set_title("February 2014", loc="left", fontsize=10)
    axes[1].set_title("September 2026", loc="left", fontsize=10)
    fig.suptitle("Olive grove near Gallipoli, Lecce", x=0.01, ha="left", fontsize=11)
    fig.text(0.01, 0.01, ATTRIB, fontsize=7.5, color="0.4")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    figstyle.save(fig, os.path.join(HERE, "figures", "fig_real_grove"))


if __name__ == "__main__":
    main()
