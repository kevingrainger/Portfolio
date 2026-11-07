#!/usr/bin/env python3
#-------- fetch_era5_daily.py -------------------------------------------------------
#-----------------------------------------------------------------------------
# Thirty years of daily solar radiation at the 25 sites, so the event set can see whole
# cloudy years and not only the day-to-day weather of 2023.
#
# ERA5 daily shortwave radiation sums (MJ/m2), 1995-2024, from the Open-Meteo archive API
# (free, no account; one request per site, spaced out to stay inside its rate limit).
# Output per MW of panels: MJ/m2 / 3.6 = kWh/m2, x 0.8 performance ratio = MWh per MW,
# the same proxy as the hourly 2023 analysis.
#
#   python fetch_era5_daily.py
#
# Output: data/era5_daily_1995_2024.csv (one column per site, MWh per MW per day)

import json
import os
import time
import urllib.request

import pandas as pd

import solar_sites as ss

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "era5_daily_1995_2024.csv")
URL = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
       "&start_date=1995-01-01&end_date=2024-12-31&daily=shortwave_radiation_sum&timezone=UTC")


def get(lat, lon, tries=6):
    for k in range(tries):
        try:
            req = urllib.request.Request(URL.format(lat=lat, lon=lon), headers={"User-Agent": "portfolio-research-script"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)["daily"]
        except Exception as e:
            print("retry", lat, lon, e, flush=True)
            time.sleep(30 * (k + 1))
    raise RuntimeError("Open-Meteo failed")


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cols = {}
    for name, (lat, lon) in ss.SITES.items():
        d = get(lat, lon)
        cols[name] = pd.Series(d["shortwave_radiation_sum"], index=pd.to_datetime(d["time"])) / 3.6 * 0.8
        print(name, "ok", flush=True)
        time.sleep(8)
    df = pd.DataFrame(cols)
    df.index.name = "date"
    df.to_csv(OUT)
    print(df.describe().T[["mean", "min", "max"]].round(2))


if __name__ == "__main__":
    main()
