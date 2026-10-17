#!/usr/bin/env python3
#-------- fetch_sites.py -----------------------------------------------------------
#-----------------------------------------------------------------------------
# Rainfall and greenness at the two Sahel sites of Dardel et al. (2014), which had
# similar rainfall histories and opposite fates after the 1980s droughts:
#   Gourma, Mali     re-greened strongly from the 1990s
#   Fakara, Niger    kept losing vegetation, under heavy cropping and grazing
#
# All free, no account:
#   rain    ERA5 daily precipitation, 1950-, from the Open-Meteo archive (averaged over
#           a 3 x 3 grid of points 0.25 deg apart around each site)
#   NDVI    MODIS MOD13Q1, 250 m, 16-day composites, 2000-, from the ORNL DAAC subset
#           service: the median over a 5 x 5 km box. Only the growing season (July-
#           October) is fetched, since the analysis uses each year's peak; each composite
#           is already the clearest view of its 16 days, and the median over 400 pixels
#           shrugs off what cloud is left.
#
#   python fetch_sites.py
#
# Output: data/rain_<site>.csv, data/ndvi_<site>.csv

import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SITES = {"gourma": (15.33, -1.55), "fakara": (13.50, 2.70)}
ORNL = "https://modis.ornl.gov/rst/api/v1/MOD13Q1"


def get_json(url, tries=5):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def fetch_rain(site, lat, lon):
    path = os.path.join(DATA, f"rain_{site}.csv")
    if os.path.exists(path):
        return
    lats = [lat + d for d in (-0.25, 0, 0.25)]
    lons = [lon + d for d in (-0.25, 0, 0.25)]
    pts = [(a, o) for a in lats for o in lons]
    q = dict(latitude=",".join(f"{a:.2f}" for a, _ in pts), longitude=",".join(f"{o:.2f}" for _, o in pts),
             start_date="1950-01-01", end_date="2025-12-31", daily="precipitation_sum", timezone="GMT")
    r = get_json("https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(q))
    series = [pd.Series(x["daily"]["precipitation_sum"], index=pd.to_datetime(x["daily"]["time"])) for x in r]
    pd.concat(series, axis=1).mean(axis=1).rename("precip_mm").to_csv(path)
    print(f"rain {site}: done")


def fetch_ndvi(site, lat, lon):
    path = os.path.join(DATA, f"ndvi_{site}.csv")
    if os.path.exists(path):
        return
    dates = get_json(f"{ORNL}/dates?latitude={lat}&longitude={lon}")["dates"]
    modis = [d["modis_date"] for d in dates if d["calendar_date"][5:7] in ("07", "08", "09", "10")]
    rows = []
    for y in sorted({m[1:5] for m in modis}):               # one call per growing season (<= 10 dates)
        chunk = [m for m in modis if m[1:5] == y]
        base = f"{ORNL}/subset?latitude={lat}&longitude={lon}&startDate={chunk[0]}&endDate={chunk[-1]}&kmAboveBelow=2&kmLeftRight=2"
        for a in get_json(base + "&band=250m_16_days_NDVI")["subset"]:
            v = np.array(a["data"], float) * 1e-4
            v = v[v > -0.2]                                  # fill values are -0.3
            rows.append(dict(date=a["calendar_date"], ndvi=np.median(v) if len(v) > 5 else np.nan))
        print(f"ndvi {site}: {y}", flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"ndvi {site}: {len(rows)} composites")


def main():
    os.makedirs(DATA, exist_ok=True)
    for site, (lat, lon) in SITES.items():
        fetch_rain(site, lat, lon)
        fetch_ndvi(site, lat, lon)


if __name__ == "__main__":
    main()
