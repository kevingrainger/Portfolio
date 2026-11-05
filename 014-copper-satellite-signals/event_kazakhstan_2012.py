#!/usr/bin/env python3
#-------- event_kazakhstan_2012.py --------------------------------------------------
#-----------------------------------------------------------------------------
# Event study: the cold snap of early 2012 in Kazakhstan, when Kazakhmys reported that a
# severe winter hit transport and processing and first-quarter cathode-equivalent output
# fell to 65,100 tonnes from 74,200 a year earlier (City AM, April 2012).
#
# Sentinel-2 starts in 2015, so the Balkhash smelter is read from Landsat 7 (Collection 2
# Level-2, Microsoft Planetary Computer, no account) for 2010-2013, with the same hot-spot
# test as the Sentinel-2 series, on the matching bands: SWIR2 (2.2 um) against SWIR1
# (1.6 um) and NIR, all ratios at least 1.4 and SWIR2 reflectance at least 0.15. Landsat 7
# has had stripes of missing data since 2003, and winter scenes are often snow-covered,
# so each pass also records the share of the box that is valid, clear and snow-free.
#
#   python event_kazakhstan_2012.py
#
# Output: data/balkhash_landsat7.csv, results/kazakhstan_2012.json, figures/fig6_kazakhstan_2012.png

import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import verify_smelters as vs

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "balkhash_landsat7.csv")
LAT, LON = 46.823, 74.93
TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token/landsat-c2-l2"
NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"
COLD = ("2012-01-01", "2012-03-31")


def scenes():
    items, token = [], None
    while True:
        body = {"collections": ["landsat-c2-l2"], "intersects": {"type": "Point", "coordinates": [LON, LAT]},
                "datetime": "2010-01-01/2013-12-31", "limit": 500,
                "query": {"platform": {"eq": "landsat-7"}, "eo:cloud_cover": {"lt": 40}}}
        if token:
            body["token"] = token
        r = vs.post_json(vs.STAC, body)
        items += r["features"]
        nxt = [l for l in r.get("links", []) if l.get("rel") == "next"]
        token = nxt[0].get("body", {}).get("token") if nxt else None
        if not token:
            return items


def measure(args):
    sc, sas = args
    try:
        rd = lambda k: vs.read_box(sc["assets"][k]["href"], sas, LAT, LON)[0]
        nir, sw1, sw2, qa = rd("nir08"), rd("swir16"), rd("swir22"), rd("qa_pixel")
        scale = lambda b: np.maximum(b * 2.75e-5 - 0.2, 1e-4)
        nir, sw1, sw2 = scale(nir), scale(sw1), scale(sw2)
        qa = qa.astype(int)
        valid = (qa & 1) == 0                                       # not fill (the SLC-off stripes)
        clear = valid & ((qa >> 3) & 1 == 0) & ((qa >> 4) & 1 == 0) & ((qa >> 5) & 1 == 0)   # no cloud, shadow, snow
        hot = clear & (sw2 / sw1 >= 1.4) & (sw2 / nir >= 1.4) & (sw2 >= 0.15)
        return dict(date=sc["properties"]["datetime"][:10], scene=sc["id"], hot=int(hot.sum()),
                    clear=float(clear.mean()), valid=float(valid.mean()), snow=float((valid & ((qa >> 5) & 1 == 1)).mean()))
    except Exception as e:
        return dict(date=sc["properties"]["datetime"][:10], scene=sc["id"], hot=-1, clear=np.nan, error=str(e)[:80])


def main():
    vs.HALF = 1500
    if not os.path.exists(OUT):
        sas = json.loads(urllib.request.urlopen(TOKEN, timeout=60).read())["token"]
        sc = scenes()
        with ThreadPoolExecutor(12) as ex:
            rows = list(ex.map(measure, [(s, sas) for s in sc]))
        pd.DataFrame(rows).sort_values("date").to_csv(OUT, index=False)
    d = pd.read_csv(OUT, parse_dates=["date"])
    d = d[d.hot >= 0]
    good = d[d.clear >= 0.3].copy()
    good["hot_per_clear"] = good.hot / good.clear                    # hot pixels per fully clear box
    q = good.set_index("date").hot_per_clear.resample("QE").agg(["mean", "count"])
    cold = good[(good.date >= COLD[0]) & (good.date <= COLD[1])]
    same_q = good[good.date.dt.quarter == 1]
    out = dict(passes=len(d), usable=len(good), quarterly=q.reset_index().astype({"date": str}).to_dict(orient="records"),
               q1_2012=dict(n=len(cold), mean=float(cold.hot_per_clear.mean()) if len(cold) else None),
               q1_other_years=float(same_q[same_q.date.dt.year != 2012].hot_per_clear.mean()),
               all_other_quarters=float(good[(good.date < COLD[0]) | (good.date > COLD[1])].hot_per_clear.mean()))
    json.dump(out, open(os.path.join(HERE, "results", "kazakhstan_2012.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "quarterly"}, indent=1))

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw=dict(height_ratios=[1.4, 1], hspace=0.15))
    for ax in axes:
        ax.axvspan(pd.Timestamp(COLD[0]), pd.Timestamp(COLD[1]), color=LIGHT, alpha=0.35, lw=0)
    ax = axes[0]
    ax.plot(good.date, good.hot_per_clear, "o", color=RED, ms=4, alpha=0.7)
    ax.plot(q.index, q["mean"], color=NAVY, lw=1.6, drawstyle="steps-pre")
    ax.set_ylabel("hot pixels per clear pass")
    ax.set_title("Balkhash smelter heat, Landsat 7", loc="left")
    ax.text(pd.Timestamp(COLD[0]), 0.98, " 2012 cold snap", transform=ax.get_xaxis_transform(), color=BLUE, va="top", fontsize=9)
    ax = axes[1]
    ax.fill_between(d.date, 0, d.snow, step="mid", color=LIGHT, alpha=0.8, lw=0, label="snow")
    ax.step(d.date, d.snow, where="mid", color=BLUE, lw=0.8)
    ax.plot(d.date, d.clear, ".", color=GREY, ms=4, label="clear and snow-free")
    ax.set_ylabel("share of the box"); ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title("What the satellite could see", loc="left")
    fig.savefig(os.path.join(HERE, "figures", "fig6_kazakhstan_2012.png"), dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
