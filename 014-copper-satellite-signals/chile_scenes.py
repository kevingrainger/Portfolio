#!/usr/bin/env python3
#-------- chile_scenes.py -----------------------------------------------------------
#-----------------------------------------------------------------------------
# Real Sentinel-2 scenes of the Caletones smelter (Codelco, which smelts El Teniente's
# ore) around the central Chile storm of 15-21 July 2026, including passes during the
# storm whatever their cloud cover. Top row: true colour. Bottom row: shortwave infrared
# false colour (B12, B11, B8A) with the pixels passing the hot-spot test outlined.
# 3 x 3 km around the smelter, Microsoft Planetary Computer.
#
#   python chile_scenes.py
#
# Output: figures/fig8_caletones_scenes.png

import json
import os
import urllib.request

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import verify_smelters as vs

HERE = os.path.dirname(os.path.abspath(__file__))
LAT, LON = -34.1065, 70.4577 * -1
WANT = ["2026-06-26", "2026-07-03", "2026-07-11", "STORM", "2026-08-19", "2026-08-30"]


def scenes(start, end, max_cloud=100):
    body = {"collections": ["sentinel-2-l2a"], "intersects": {"type": "Point", "coordinates": [LON, LAT]},
            "datetime": f"{start}/{end}", "limit": 100, "query": {"eo:cloud_cover": {"lte": max_cloud}}}
    return sorted(vs.post_json(vs.STAC, body)["features"], key=lambda f: f["properties"]["datetime"])


def stretch(a, lo=2, hi=98):
    v = a[np.isfinite(a)]
    p1, p2 = np.percentile(v, [lo, hi])
    return np.clip((a - p1) / (p2 - p1 + 1e-9), 0, 1)


def main():
    vs.HALF = 1500
    sas = json.loads(urllib.request.urlopen(vs.TOKEN, timeout=60).read())["token"]
    picks = []
    for d in WANT:
        if d == "STORM":
            st = scenes("2026-07-14T00:00:00Z", "2026-07-24T23:59:59Z")
            picks.append(max(st, key=lambda f: f["properties"]["eo:cloud_cover"]) if st else None)
        else:
            st = scenes(f"{d}T00:00:00Z", f"{d}T23:59:59Z")
            picks.append(st[0] if st else None)
    picks = [p for p in picks if p is not None]
    fig, axes = plt.subplots(2, len(picks), figsize=(3.0 * len(picks), 6.4), squeeze=False)
    for k, sc in enumerate(picks):
        offset = 1000 if sc["properties"].get("s2:processing_baseline", "00.00") >= "04.00" else 0
        rd = lambda b: np.maximum((vs.read_box(sc["assets"][b]["href"], sas, LAT, LON)[0] - offset) / 1e4, 1e-4)
        r, g, b = rd("B04"), rd("B03"), rd("B02")
        hot, _, _ = vs.hot_mask(sc, sas, LAT, LON)
        b8a, b11, b12 = rd("B8A"), rd("B11"), rd("B12")
        if b12.shape != hot.shape:
            hot = np.zeros(b12.shape, bool)
        axes[0, k].imshow(np.dstack([stretch(r), stretch(g), stretch(b)]))
        axes[1, k].imshow(np.dstack([stretch(b12, 2, 99.5), stretch(b11), stretch(b8a)]))
        if hot.any():
            axes[1, k].contour(hot.astype(float), levels=[0.5], colors="#ff3b30", linewidths=0.9)
        day = sc["properties"]["datetime"][:10]
        cc = sc["properties"]["eo:cloud_cover"]
        axes[0, k].set_title(day + (f"  (storm, {cc:.0f}% cloud)" if "2026-07-14" <= day <= "2026-07-24" else ""), loc="left", fontsize=10)
        axes[1, k].text(0.03, 0.04, f"{int(hot.sum())} hot pixels", transform=axes[1, k].transAxes, color="w", fontsize=9,
                        bbox=dict(facecolor="k", alpha=0.5, edgecolor="none", pad=2))
        for ax in axes[:, k]:
            ax.set_xticks([]); ax.set_yticks([])
        print(day, cc, int(hot.sum()))
    axes[0, 0].set_ylabel("true colour", fontsize=10.5); axes[1, 0].set_ylabel("shortwave infrared", fontsize=10.5)
    fig.suptitle("Caletones smelter, Sentinel-2, 3 x 3 km", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "fig8_caletones_scenes.png"), dpi=170, bbox_inches="tight")


if __name__ == "__main__":
    main()
