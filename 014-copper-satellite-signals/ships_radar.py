#!/usr/bin/env python3
#-------- ships_radar.py ------------------------------------------------------------
#-----------------------------------------------------------------------------
# Ships at San Antonio, Chile's central copper port, seen by radar through the July 2026
# storm. Sentinel-1 (radiometrically terrain-corrected, VV polarisation, Microsoft
# Planetary Computer) sees through cloud: the sea is dark, and a ship's steel hull is a
# bright point. Every pass from 1 July to 5 August, a 16 x 16 km box over the port and its
# offshore anchorage. Ships show as bright points on the dark sea. A simple count of bright,
# compact objects on water (backscatter
# more than 12 dB above the sea around them, at least 3 pixels, at least 250 m off the
# land so harbour walls and cranes don't count). Passes that cover less than 90% of the box
# are skipped. The count is kept in results/ for reference but not drawn: rough storm seas
# hide ships from a fixed threshold and strong port reflectors leave streaks, so reliable
# counting needs a CFAR detector, the standard method for radar ship detection.
#
#   python ships_radar.py
#
# Output: figures/fig9_ships_radar.gif, figures/fig9_ships_radar.png, results/ships_radar.csv

import io
import json
import os
import urllib.request

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage

import verify_smelters as vs

HERE = os.path.dirname(os.path.abspath(__file__))
LAT, LON = -33.575, -71.66                       # between the port and the anchorage to its west
TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-rtc"
STORM = ("2026-07-15", "2026-07-21")
RED = "#c0392b"


def scenes():
    body = {"collections": ["sentinel-1-rtc"], "intersects": {"type": "Point", "coordinates": [LON, LAT]},
            "datetime": "2026-07-01T00:00:00Z/2026-08-05T23:59:59Z", "limit": 100}
    return sorted(vs.post_json(vs.STAC, body)["features"], key=lambda f: f["properties"]["datetime"])


def main():
    vs.HALF = 8000
    sas = json.loads(urllib.request.urlopen(TOKEN, timeout=60).read())["token"]
    frames, rows = [], []
    for sc in scenes():
        a = vs.read_box(sc["assets"]["vv"]["href"], sas, LAT, LON)[0]
        valid = np.isfinite(a) & (a > 0)
        if valid.mean() < 0.9:                                                  # the pass must cover the whole box
            continue
        db = 10 * np.log10(np.maximum(a, 1e-5))
        land = ndimage.uniform_filter((db > -14).astype(float), 25) > 0.5       # bright, textured ground
        land = ndimage.binary_dilation(land, iterations=25)                     # and a 250 m buffer: harbour walls, cranes, surf
        sea_level = ndimage.median_filter(db, size=31)
        bright = (db - sea_level > 12) & ~land
        lab, n = ndimage.label(bright)
        sizes = ndimage.sum(bright, lab, range(1, n + 1))
        ships = [i + 1 for i, s in enumerate(sizes) if s >= 3]
        centres = ndimage.center_of_mass(bright, lab, ships) if ships else []
        day = sc["properties"]["datetime"][:10]
        storm = STORM[0] <= day <= STORM[1]
        rows.append(dict(date=day, ships=len(ships), storm=storm))
        frames.append((day, db, centres, storm))
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "results", "ships_radar.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False))

    images = []
    for day, db, centres, storm in frames:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(np.clip(db, -24, -4), cmap="gray")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"San Antonio, Sentinel-1 radar, {day}", loc="left", fontsize=11)
        if storm:
            ax.text(0.02, 0.03, "storm", transform=ax.transAxes, color="w", fontsize=11,
                    bbox=dict(facecolor=RED, alpha=0.7, edgecolor="none", pad=3))
        buf = io.BytesIO(); fig.savefig(buf, dpi=110, bbox_inches="tight"); plt.close(fig)
        images.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
    images[0].save(os.path.join(HERE, "figures", "fig9_ships_radar.gif"), save_all=True, append_images=images[1:],
                   duration=900, loop=0)

    pick = [i for i, f in enumerate(frames) if f[0] < STORM[0]][-1:] + [i for i, f in enumerate(frames) if f[3]] +            [i for i, f in enumerate(frames) if f[0] > STORM[1]][:2]
    pick = pick[:4]
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    px_km = 2 * vs.HALF / 1000 / frames[0][1].shape[1]          # km per pixel
    for ax, i in zip(axes.ravel(), pick):
        day, db, centres, storm = frames[i]
        ax.imshow(np.clip(db, -24, -4), cmap="gray")             # brighter stretch: ships stand out on the sea
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(day + ("  (storm)" if storm else ""), loc="left", fontsize=12, color=RED if storm else "k")
        h, w = db.shape
        n2 = 2 / px_km                                            # a 2 km scale bar
        ax.plot([0.05 * w, 0.05 * w + n2], [0.94 * h] * 2, color="w", lw=3)
        ax.text(0.05 * w + n2 / 2, 0.915 * h, "2 km", color="w", ha="center", fontsize=10)
        ax.annotate("N", xy=(0.93 * w, 0.06 * h), xytext=(0.93 * w, 0.16 * h), color="w", ha="center", fontsize=11,
                    arrowprops=dict(arrowstyle="-|>", color="w", lw=1.5))
        ax.text(0.80 * w, 0.62 * h, "port", color="#ffcc00", fontsize=11, ha="center")
        ax.text(0.30 * w, 0.40 * h, "anchorage", color="#ffcc00", fontsize=11, ha="center")
    fig.suptitle("San Antonio port and anchorage through the July 2026 storm, Sentinel-1 radar (ships are bright points)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "fig9_ships_radar.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
