#!/usr/bin/env python3
#-------- fetch_openmeteo.py -------------------------------------------------------
#-----------------------------------------------------------------------------
# Past AIFS and IFS solar forecasts at the 25 sites, and what actually happened, from
# Open-Meteo. Free, no account.
#
# Why not fetch_forecasts.py? That script reads ECMWF's global GRIB files on AWS. It
# works, but the bucket throttles hard (503 Slow Down, under 10 kB/s), and the IFS files
# before 2025 have no solar radiation. Open-Meteo keeps the same two models' forecasts as
# point series, stored by lead time: "previous_dayN" is the value for each hour as
# forecast N days earlier, which is exactly what a trader had in hand.
#
#   forecasts   ECMWF AIFS Single (the AI model, archive from early 2025) and ECMWF IFS
#               (the physics model), hourly surface solar radiation, 1-7 days ahead
#   truth       ERA5 daily surface solar radiation, 2000- (also the climatology)
#
#   python aifs/fetch_openmeteo.py
#
# Output: aifs/data/forecast_<model>.csv, aifs/data/era5_daily.csv

import datetime
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from solar_sites import SITES                                    # noqa: E402

OUT = os.path.join(HERE, "data")
START = "2025-03-01"
END = str(datetime.date.today() - datetime.timedelta(days=8))  # ERA5 runs about five days behind
MODELS = {"aifs": "ecmwf_aifs025_single", "ifs": "ecmwf_ifs025"}
LEADS = range(1, 8)


def get_json(url, tries=5):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                return json.load(r)
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(10 * (k + 1))


def fetch_forecasts(name, model):
    path = os.path.join(OUT, f"forecast_{name}.csv")
    if os.path.exists(path):
        return
    frames = []
    hourly = ",".join(f"shortwave_radiation_previous_day{n}" for n in LEADS)
    for site, (lat, lon) in SITES.items():
        q = dict(latitude=lat, longitude=lon, start_date=START, end_date=END, hourly=hourly, models=model, timezone="GMT")
        h = pd.DataFrame(get_json("https://previous-runs-api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(q))["hourly"])
        h["date"] = pd.to_datetime(h.pop("time")).dt.date
        # daily energy in MJ/m2: hourly mean W/m2 x 3600 s, summed; a day missing any hour is dropped
        d = h.groupby("date").agg(lambda s: s.sum() * 3600 / 1e6 if s.notna().all() else float("nan"))
        d.columns = [f"lead{n}" for n in LEADS]
        frames.append(d.reset_index().assign(site=site))
        print(f"{name} {site}: {d.notna().all(axis=1).sum()} complete days", flush=True)
    pd.concat(frames).to_csv(path, index=False)


def fetch_truth():
    path = os.path.join(OUT, "era5_daily.csv")
    if os.path.exists(path):
        return
    frames = []
    for site, (lat, lon) in SITES.items():
        q = dict(latitude=lat, longitude=lon, start_date="2000-01-01", end_date=END,
                 daily="shortwave_radiation_sum", models="era5", timezone="GMT")
        d = pd.DataFrame(get_json("https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(q))["daily"])
        frames.append(d.rename(columns={"time": "date", "shortwave_radiation_sum": "era5"}).assign(site=site))
        print(f"era5 {site}: {len(d)} days", flush=True)
    pd.concat(frames).to_csv(path, index=False)


def main():
    os.makedirs(OUT, exist_ok=True)
    fetch_truth()
    for name, model in MODELS.items():
        fetch_forecasts(name, model)


if __name__ == "__main__":
    main()
