#!/usr/bin/env python3
#-------- fetch_real.py ------------------------------------------------------------
#-----------------------------------------------------------------------------
# The real series behind the simulated ones in power.py, from free sources:
#
#   price       day-ahead price, Italy North bidding zone, EUR/MWh, hourly -> daily mean
#               Energy-Charts API (Fraunhofer ISE), CC BY 4.0, from ENTSO-E/SMARD
#   load, wind, solar   Italian public net generation and load, Energy-Charts, -> daily
#               totals; renewables = (wind + solar) / load
#   temp        daily mean 2 m temperature, ERA5 via Open-Meteo, the mean of eight
#               cities spread over the country (equal weights)
#   gas         Dutch TTF front-month futures, EUR/MWh, Yahoo Finance (TTF=F, from
#               October 2017; not redistributable, so not committed)
#
#   python fetch_real.py
#
# Output: data/price.csv, data/power.csv, data/temp.csv, data/ttf.csv (ignored by git),
#         data/real_daily.csv: everything joined, in the column layout of simulate()

import datetime
import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
YEARS = range(2015, datetime.date.today().year + 1)
CITIES = {"Milano": (45.46, 9.19), "Torino": (45.07, 7.69), "Bologna": (44.49, 11.34), "Firenze": (43.77, 11.25),
          "Roma": (41.90, 12.50), "Napoli": (40.85, 14.27), "Bari": (41.12, 16.87), "Palermo": (38.12, 13.36)}


def get_json(url, tries=5):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                return json.load(r)
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(10 * (k + 1))


def daily(unix, values, how):
    s = pd.Series(values, index=pd.to_datetime(unix, unit="s", utc=True).tz_convert("Europe/Rome").tz_localize(None),
                  dtype=float)
    return getattr(s.resample("D"), how)()


def fetch_price():
    path = os.path.join(DATA, "price.csv")
    if os.path.exists(path):
        return
    parts = []
    for y in YEARS:
        r = get_json(f"https://api.energy-charts.info/price?bzn=IT-North&start={y}-01-01&end={y}-12-31")
        parts.append(daily(r["unix_seconds"], r["price"], "mean"))
        print(f"price {y}", flush=True)
    pd.concat(parts).rename("price").to_csv(path)


def fetch_power():
    path = os.path.join(DATA, "power.csv")
    if os.path.exists(path):
        return
    parts = []
    for y in YEARS:
        r = get_json(f"https://api.energy-charts.info/public_power?country=it&start={y}-01-01&end={y}-12-31")
        cols = {}
        for pt in r["production_types"]:
            name = pt["name"].lower()
            if name in ("load", "solar", "wind onshore", "wind offshore"):
                step_h = np.median(np.diff(r["unix_seconds"])) / 3600
                cols[name] = daily(r["unix_seconds"], pt["data"], "sum") * step_h        # MW x h = MWh
        parts.append(pd.DataFrame(cols))
        print(f"power {y}: {sorted(cols)}", flush=True)
    pd.concat(parts).to_csv(path)


def fetch_temp():
    path = os.path.join(DATA, "temp.csv")
    if os.path.exists(path):
        return
    lat = ",".join(str(v[0]) for v in CITIES.values())
    lon = ",".join(str(v[1]) for v in CITIES.values())
    q = dict(latitude=lat, longitude=lon, start_date="2015-01-01", end_date=str(datetime.date.today() - datetime.timedelta(days=7)),
             daily="temperature_2m_mean", timezone="Europe/Rome")
    r = get_json("https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(q))
    s = pd.concat([pd.Series(x["daily"]["temperature_2m_mean"], index=pd.to_datetime(x["daily"]["time"])) for x in r], axis=1)
    s.mean(axis=1).rename("temp").to_csv(path)


def fetch_gas():
    path = os.path.join(DATA, "ttf.csv")
    if os.path.exists(path):
        return
    import yfinance as yf
    s = yf.download("TTF=F", start="2017-01-01", progress=False, auto_adjust=False)["Close"].squeeze()
    s.rename("gas").to_csv(path)


def join():
    price = pd.read_csv(os.path.join(DATA, "price.csv"), index_col=0, parse_dates=True).price
    pw = pd.read_csv(os.path.join(DATA, "power.csv"), index_col=0, parse_dates=True)
    temp = pd.read_csv(os.path.join(DATA, "temp.csv"), index_col=0, parse_dates=True).temp
    gas = pd.read_csv(os.path.join(DATA, "ttf.csv"), index_col=0, parse_dates=True).squeeze()
    days = pd.date_range("2018-01-01", min(price.index.max(), temp.index.max(), pw.index.max()), freq="D")
    df = pd.DataFrame(index=days)
    df["price"] = price.reindex(days)
    df["temp"] = temp.reindex(days)
    df["gas"] = gas.reindex(days).ffill()                           # weekends and holidays: last close
    df["log_gas"] = np.log(df.gas)
    load = pw["load"].reindex(days)
    df["load"] = load / load.mean()
    wind = pw.filter(like="wind").sum(axis=1).reindex(days)
    df["renewables"] = (pw["solar"].reindex(days) + wind) / load
    df["weekend"] = (days.dayofweek >= 5).astype(int)
    df["month"] = days.month
    df = df[df.price > 0].dropna()
    extreme = ((df.temp > 28) | (df.temp < 3)).groupby(df.index.to_period("M")).sum()
    df["extreme_month"] = extreme.reindex(df.index.to_period("M")).values
    df.to_csv(os.path.join(DATA, "real_daily.csv"))
    print(f"real_daily: {len(df)} days, {df.index.min().date()} to {df.index.max().date()}")


def main():
    os.makedirs(DATA, exist_ok=True)
    fetch_price()
    fetch_power()
    fetch_temp()
    fetch_gas()
    join()


if __name__ == "__main__":
    main()
