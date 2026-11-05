#!/usr/bin/env python3
#-------- kazakhstan_scenes.py ------------------------------------------------------
#-----------------------------------------------------------------------------
# Real Landsat 7 scenes of the Balkhash copper smelter around the 2012 cold snap.
# Top row: true colour (red, green, blue), where the snow shows. Bottom row: shortwave
# infrared false colour (2.2 um, 1.6 um, near infrared), where a furnace glows, with the
# pixels passing the hot-spot test outlined. The black stripes are Landsat 7's missing
# scan lines (since 2003). 6 x 6 km around the smelter, Microsoft Planetary Computer.
#
#   python kazakhstan_scenes.py
#
# Output: figures/fig7_balkhash_scenes.png

import json
import os
import urllib.request

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import verify_smelters as vs

HERE = os.path.dirname(os.path.abspath(__file__))
LAT, LON = 46.823, 74.93
TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token/landsat-c2-l2"
ITEM = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/landsat-c2-l2/items/{}"
SCENES = [("LE07_L2SP_151027_20110722_02_T1", "22 Jul 2011"), ("LE07_L2SP_151027_20111111_02_T1", "11 Nov 2011"),
          ("LE07_L2SP_151027_20120130_02_T1", "30 Jan 2012"), ("LE07_L2SP_151028_20120302_02_T1", "2 Mar 2012"),
          ("LE07_L2SP_151027_20120403_02_T1", "3 Apr 2012"), ("LE07_L2SP_152027_20120816_02_T1", "16 Aug 2012")]


def stretch(a, lo=2, hi=98):
    v = a[np.isfinite(a) & (a > 0)]
    if v.size == 0:
        return np.zeros_like(a)
    p1, p2 = np.percentile(v, [lo, hi])
    return np.clip((a - p1) / (p2 - p1 + 1e-9), 0, 1)


def main():
    vs.HALF = 3000
    sas = json.loads(urllib.request.urlopen(TOKEN, timeout=60).read())["token"]
    fig, axes = plt.subplots(2, len(SCENES), figsize=(3.0 * len(SCENES), 6.4))
    counts = {}
    for k, (sid, label) in enumerate(SCENES):
        it = json.loads(urllib.request.urlopen(ITEM.format(sid), timeout=60).read())
        rd = lambda b: vs.read_box(it["assets"][b]["href"], sas, LAT, LON)[0]
        sc = lambda a: np.where(a > 0, a * 2.75e-5 - 0.2, np.nan)
        r, g, b, nir, s1, s2 = (sc(rd(x)) for x in ["red", "green", "blue", "nir08", "swir16", "swir22"])
        rgb = np.dstack([stretch(r), stretch(g), stretch(b)])
        swir = np.dstack([stretch(s2, 2, 99.5), stretch(s1), stretch(nir)])
        s1c, s2c, nirc = (np.maximum(np.nan_to_num(x, nan=1e-4), 1e-4) for x in (s1, s2, nir))
        hot = (s2c / s1c >= 1.4) & (s2c / nirc >= 1.4) & (s2c >= 0.15)
        counts[label] = int(hot.sum())
        axes[0, k].imshow(rgb); axes[0, k].set_title(label, loc="left", fontsize=10.5)
        axes[1, k].imshow(swir)
        if hot.any():
            axes[1, k].contour(hot.astype(float), levels=[0.5], colors="#ff3b30", linewidths=0.9)
        axes[1, k].text(0.03, 0.04, f"{hot.sum()} hot pixels", transform=axes[1, k].transAxes, color="w", fontsize=9,
                        bbox=dict(facecolor="k", alpha=0.5, edgecolor="none", pad=2))
        for ax in axes[:, k]:
            ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel("true colour", fontsize=10.5)
    axes[1, 0].set_ylabel("shortwave infrared", fontsize=10.5)
    fig.suptitle("Balkhash copper smelter, Landsat 7, 6 x 6 km", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "fig7_balkhash_scenes.png"), dpi=170, bbox_inches="tight")
    print(counts)


if __name__ == "__main__":
    main()
