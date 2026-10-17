#!/usr/bin/env python3
#-------- fetch_data.py ------------------------------------------------------------
#-----------------------------------------------------------------------------
# Everything 013 needs, from free sources with no account:
#
#   prices    ICE arabica futures from Yahoo Finance: the continuous front month (KC=F)
#             since 2000, and each individual contract month still trading
#             (KCH27.NYB, ...). Expired contracts are delisted on Yahoo, so the panel of
#             full curves reaches back only as far as today's live contracts do.
#   weather   from Open-Meteo, at each point in coffee_regions.py:
#               era5        what actually happened (ERA5 reanalysis), 1990-
#               ecmwf       ECMWF IFS forecasts by lead time, 1-7 days, 2024-
#               gfs         NOAA GFS forecasts by lead time, 1-7 days, 2021-
#             The forecast archives keep the value forecast for each hour as issued
#             n days earlier ("previous_dayN"), which is exactly what a trader saw.
#
#   python fetch_data.py            # all of it, a few minutes
#   python fetch_data.py prices     # one part
#
# Output in data/ (git-ignored). Each file is skipped if it already exists.

import datetime
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from coffee_regions import REGIONS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TODAY = datetime.date.today()
MONTH_CODES = "HKNUZ"                                  # March, May, July, September, December
LEADS = range(1, 8)


def get_json(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))                     # the free API rate-limits bursts


#-------- prices ---------------------------------------------------------------------
def fetch_prices():
    import yfinance as yf
    front = os.path.join(DATA, "kc_front.csv")
    if not os.path.exists(front):
        s = yf.download("KC=F", start="2000-01-01", progress=False, auto_adjust=False)["Close"].squeeze()
        s.rename("KC").to_csv(front)
        print(f"front month: {len(s)} days")
    curves = os.path.join(DATA, "kc_contracts.csv")
    if not os.path.exists(curves):
        cols = {}
        for year in range(TODAY.year, TODAY.year + 4):
            for code in MONTH_CODES:
                tic = f"KC{code}{year % 100:02d}.NYB"
                h = yf.Ticker(tic).history(period="max", auto_adjust=False)
                if len(h):
                    h.index = h.index.tz_localize(None).normalize()
                    cols[tic] = h["Close"]
        df = pd.DataFrame(cols).sort_index()
        df.to_csv(curves)
        print(f"contracts: {df.shape[1]} live, {df.index.min().date()} to {df.index.max().date()}")


#-------- weather --------------------------------------------------------------------
def fetch_era5(start="1990-01-01"):
    path = os.path.join(DATA, "weather_era5.csv")
    if os.path.exists(path):
        return
    frames = []
    for name, (lat, lon, _) in REGIONS.items():
        q = dict(latitude=lat, longitude=lon, start_date=start, end_date=str(TODAY - datetime.timedelta(days=6)),
                 daily="temperature_2m_min,precipitation_sum", timezone="America/Sao_Paulo")
        r = get_json("https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(q))
        d = pd.DataFrame(r["daily"]).assign(region=name)
        frames.append(d)
        print(f"era5 {name}: {len(d)} days")
    pd.concat(frames).to_csv(path, index=False)


def fetch_forecasts(model, start):
    """Daily minimum temperature at each region as forecast 1..7 days ahead. The archive is
    hourly; the daily minimum of the hourly values forecast n days earlier is taken."""
    path = os.path.join(DATA, f"forecast_{model.split('_')[0]}.csv")
    if os.path.exists(path):
        return
    hourly = ",".join(f"temperature_2m_previous_day{n}" for n in LEADS)
    frames = []
    for name, (lat, lon, _) in REGIONS.items():
        q = dict(latitude=lat, longitude=lon, start_date=start, end_date=str(TODAY - datetime.timedelta(days=1)),
                 hourly=hourly, models=model, timezone="America/Sao_Paulo")
        r = get_json("https://previous-runs-api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(q))
        h = pd.DataFrame(r["hourly"])
        h["date"] = pd.to_datetime(h.pop("time")).dt.date
        d = h.groupby("date").min()                                 # NaN-aware daily minimum
        d.columns = [f"tmin_lead{n}" for n in LEADS]
        frames.append(d.reset_index().assign(region=name))
        print(f"{model} {name}: {d.notna().all(axis=1).sum()} complete days")
    pd.concat(frames).to_csv(path, index=False)


def main():
    os.makedirs(DATA, exist_ok=True)
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    if part in ("all", "prices"):
        fetch_prices()
    if part in ("all", "weather"):
        fetch_era5()
        fetch_forecasts("ecmwf_ifs025", "2024-02-01")
        fetch_forecasts("gfs_seamless", "2021-03-23")


if __name__ == "__main__":
    main()
