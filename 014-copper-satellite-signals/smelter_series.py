#!/usr/bin/env python3
#-------- smelter_series.py --------------------------------------------------------
#-----------------------------------------------------------------------------
# How hot each smelter ran, pass by pass, 2019-2026, from Sentinel-2 on Microsoft
# Planetary Computer (free, no account). The same measurement as smelter_heat.py, which
# does it on Google Earth Engine; this version needs no sign-in at all.
#
# For every verified site in smelters.csv and every Sentinel-2 pass with under 20% cloud
# on its tile, a 3 x 3 km box around the hot spot is read and two numbers kept:
#   hot     pixels passing the Murphy et al. (2016) test (see verify_smelters.py)
#   clear   share of the box not flagged as cloud, cloud shadow or cirrus (scene
#           classification), so a cloudy pass is not mistaken for a cold furnace
#
#   python smelter_series.py            # about an hour, 16 reads at a time; resumable
#
# Output: data/smelter_passes.csv, one row per site and pass

import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import verify_smelters as vs

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "smelter_passes.csv")
START, END = "2019-01-01", "2026-10-15"
CLOUDY = {3, 8, 9, 10}                         # cloud shadow, cloud medium, cloud high, cirrus


def all_scenes(lat, lon):
    items, token = [], None
    while True:
        body = {"collections": ["sentinel-2-l2a"], "intersects": {"type": "Point", "coordinates": [lon, lat]},
                "datetime": f"{START}/{END}", "limit": 1000, "query": {"eo:cloud_cover": {"lt": 20}}}
        if token:
            body["token"] = token
        r = vs.post_json(vs.STAC, body)
        items += r["features"]
        nxt = [l for l in r.get("links", []) if l.get("rel") == "next"]
        if not nxt:
            return items
        token = nxt[0].get("body", {}).get("token")
        if not token:
            return items


def measure(args):
    site, lat, lon, sc, sas = args
    try:
        hot, _, _ = vs.hot_mask(sc, sas, lat, lon)
        scl = vs.read_box(sc["assets"]["SCL"]["href"], sas, lat, lon)[0]
        clear = float(np.mean(~np.isin(scl, list(CLOUDY)) & (scl > 0)))
        return dict(site=site, date=sc["properties"]["datetime"][:10], scene=sc["id"],
                    hot=int(hot.sum()), clear=clear)
    except Exception as e:
        return dict(site=site, date=sc["properties"]["datetime"][:10], scene=sc["id"], hot=-1, clear=np.nan,
                    error=str(e)[:80])


def main():
    vs.HALF = 1500
    sites = pd.read_csv(os.path.join(HERE, "smelters.csv"))
    sites = sites[sites.verified == "yes"]
    done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["site", "scene"])
    seen = set(zip(done.site, done.scene))
    rows = done.to_dict("records")
    for r in sites.itertuples():
        sas = json.loads(urllib.request.urlopen(vs.TOKEN, timeout=60).read())["token"]     # tokens expire: one per site
        todo = [(r.name, r.lat, r.lon, sc, sas) for sc in all_scenes(r.lat, r.lon) if (r.name, sc["id"]) not in seen]
        with ThreadPoolExecutor(16) as ex:
            got = list(ex.map(measure, todo))
        rows += got
        pd.DataFrame(rows).to_csv(OUT, index=False)
        ok = [g for g in got if g["hot"] >= 0]
        print(f"{r.name:36s} {len(todo):4d} passes, {len(ok)} read, mean hot {np.mean([g['hot'] for g in ok]) if ok else 0:.1f}", flush=True)


if __name__ == "__main__":
    main()
