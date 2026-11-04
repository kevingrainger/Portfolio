#!/usr/bin/env python3
#-------- fetch_omie.py ------------------------------------------------------------
#-----------------------------------------------------------------------------
# Spanish day-ahead electricity prices, hour by hour, from OMIE (the Iberian market
# operator). OMIE publishes one small text file per day of marginal prices, free to
# download with no account, so no ENTSO-E key is needed.
#
#   File:   marginalpdbc_YYYYMMDD.1   (semicolon separated)
#           year;month;day;period;price_PT;price_ES;
#   Periods are hours 1-24 (23 or 25 on clock-change days). From 1 October 2025 OMIE
#   moved to 15-minute periods (96 a day); those are averaged to hours here.
#
#   python fetch_omie.py 2023-01-01 2023-12-31
#
# Output: data/omie_es_hourly.csv  (time in local Spanish clock hours, price in EUR/MWh)
# Source: OMIE, https://www.omie.es (cite OMIE when the data are used).

import os
import sys
import time
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
URL = "https://www.omie.es/es/file-download?parents%5B0%5D=marginalpdbc&filename=marginalpdbc_{d}.{v}"


def day_file(day):
    """The text of one day's file. Nearly always version .1, but some days are only
    published under a later version (2023-05-28 is .2)."""
    for version in (1, 2, 3):
        url = URL.format(d=day.strftime("%Y%m%d"), v=version)
        for k in range(4):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    return r.read().decode("latin-1")
            except Exception as err:
                if getattr(err, "code", None) == 404:
                    break                                    # not this version: try the next
                if k == 3:
                    raise
                time.sleep(5 * (k + 1))
    raise FileNotFoundError(day.date())


def one_day(day):
    text = day_file(day)
    rows = []
    for line in text.splitlines():
        f = line.strip().split(";")
        if len(f) < 6 or not f[0].isdigit():
            continue                                         # header "MARGINALPDBC;" and the closing "*"
        rows.append(dict(period=int(f[3]), price_es=float(f[5].replace(",", "."))))
    df = pd.DataFrame(rows)
    per_hour = 4 if df.period.max() > 25 else 1
    df["hour"] = (df.period - 1) // per_hour
    df = df.groupby("hour", as_index=False).price_es.mean()
    df.insert(0, "date", day.date())
    return df


def main():
    start, end = (pd.Timestamp(a) for a in sys.argv[1:3])
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "omie_es_hourly.csv")
    parts = []
    done = set()
    if os.path.exists(path):
        have = pd.read_csv(path)
        have["date"] = pd.to_datetime(have.date).dt.date     # plain dates, the same as one_day
        done = set(have.date)
        parts.append(have)
    missing = []
    for day in pd.date_range(start, end, freq="D"):
        if day.date() in done:
            continue
        try:
            parts.append(one_day(day))
        except FileNotFoundError:
            missing.append(str(day.date()))                  # left as a gap, never filled in
            continue
        if day.day == 1:
            print(day.strftime("%Y-%m"), flush=True)
            pd.concat(parts).to_csv(path, index=False)       # resumable
        time.sleep(0.3)                                      # be polite to a free service
    pd.concat(parts).sort_values(["date", "hour"]).to_csv(path, index=False)
    print("not published by OMIE:", ", ".join(missing) if missing else "none")
    print("done:", path)


if __name__ == "__main__":
    main()
