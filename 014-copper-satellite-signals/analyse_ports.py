#!/usr/bin/env python3
#-------- analyse_ports.py ---------------------------------------------------------
#-----------------------------------------------------------------------------
# Is the ship signal any good, and does it say anything about the copper price?
#
# 1. Validation. Peru's central bank publishes monthly copper export volumes (BCRP
#    series PN38783BM, thousand tonnes, free API). Sum PortWatch's dry-bulk export
#    tonnes over Peru's copper ports by month and compare, in levels and in year-on-
#    year changes; then do the same with Peru's placebo ports, which do not ship copper.
#
# 2. Prediction, out of sample. Each week, the signal is the copper ports' dry-bulk
#    exports over the last four weeks, as a log ratio to the same four weeks a year
#    earlier (so the seasons cancel). Does it predict the next four weeks' log change
#    in COMEX copper? Fitted on an expanding window from 2021, forecasting each week
#    only with data available then; scored by out-of-sample R^2 against a forecast of
#    zero (Campbell & Thompson 2008). The placebo ports get the same treatment.
#
# 3. Grasberg. Daily dry-bulk exports from Amamapare, Grasberg's port, around the
#    mudslide of 8 September 2025.
#
#   python analyse_ports.py
#
# Output: results/validation_peru.csv, results/prediction.csv, results/amamapare.csv

import json
import os
import urllib.request

import numpy as np
import pandas as pd

from fetch_ports import EXPORT_PORTS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RES = os.path.join(HERE, "results")
PERU_COPPER = ["port1064", "port88", "port1047", "port1045"]
PERU_PLACEBO = ["port873", "port1026", "port1040"]
MONTHS_ES = {"Ene": 1, "Feb": 2, "Mar": 3, "Abr": 4, "May": 5, "Jun": 6, "Jul": 7, "Ago": 8,
             "Sep": 9, "Set": 9, "Oct": 10, "Nov": 11, "Dic": 12}


def bcrp_copper():
    path = os.path.join(DATA, "bcrp_copper_exports.csv")
    if not os.path.exists(path):
        url = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api/PN38783BM/json/2019-1/2026-12"
        d = json.load(urllib.request.urlopen(url, timeout=120))
        rows = []
        for p in d["periods"]:
            m, y = p["name"].split(".")
            v = p["values"][0]
            if v not in ("n.d.", ""):
                rows.append(dict(month=pd.Timestamp(int(y), MONTHS_ES[m], 1), kt=float(v)))
        pd.DataFrame(rows).to_csv(path, index=False)
    return pd.read_csv(path, parse_dates=["month"]).set_index("month").kt


def copper_price():
    path = os.path.join(DATA, "hg_front.csv")
    if not os.path.exists(path):
        import yfinance as yf
        s = yf.download("HG=F", start="2018-01-01", progress=False, auto_adjust=False)["Close"].squeeze()
        s.rename("HG").to_csv(path)
    return pd.read_csv(path, index_col=0, parse_dates=True).squeeze()


def oos_r2(x, y, start):
    """Expanding-window regression y = a + b x, forecasting each date with data up to the
    date before; R^2 out of sample against a zero forecast."""
    pred, actual = [], []
    for i in range(len(y)):
        if y.index[i] < start:
            continue
        past = slice(0, max(i - 4, 0))                   # the target overlaps 4 weeks ahead
        X, Y = x.iloc[past], y.iloc[past]
        if len(X) < 52:
            continue
        b, a = np.polyfit(X.values, Y.values, 1)
        pred.append(a + b * x.iloc[i])
        actual.append(y.iloc[i])
    pred, actual = np.array(pred), np.array(actual)
    return 1 - np.sum((actual - pred) ** 2) / np.sum(actual ** 2), len(actual)


def main():
    os.makedirs(RES, exist_ok=True)
    ports = pd.read_csv(os.path.join(DATA, "ports_daily.csv"), parse_dates=["date"])

    #-------- 1. validation against Peru's official exports ---------------------------
    off = bcrp_copper()
    monthly = ports.assign(month=ports.date.dt.to_period("M").dt.to_timestamp())
    def peru(ids):
        return monthly[monthly.portid.isin(ids)].groupby("month").export_dry_bulk.sum() / 1e3   # kt
    v = pd.DataFrame({"official_kt": off, "ships_copper_ports_kt": peru(PERU_COPPER),
                      "ships_placebo_ports_kt": peru(PERU_PLACEBO)}).dropna()
    v = v[v.index < monthly.month.max()]                   # drop the part-month at the end
    v.to_csv(os.path.join(RES, "validation_peru.csv"))
    yoy = np.log(v).diff(12).dropna()
    stats = {"months": len(v),
             "corr_level_copper": v.official_kt.corr(v.ships_copper_ports_kt),
             "corr_level_placebo": v.official_kt.corr(v.ships_placebo_ports_kt),
             "corr_yoy_copper": yoy.official_kt.corr(yoy.ships_copper_ports_kt),
             "corr_yoy_placebo": yoy.official_kt.corr(yoy.ships_placebo_ports_kt)}
    print({k: round(x, 3) for k, x in stats.items()})

    #-------- 2. prediction of copper's price ---------------------------------------
    price = copper_price().resample("W-FRI").last()
    target = np.log(price).shift(-4) - np.log(price)       # next four weeks
    rows = []
    for role in ["copper", "placebo"]:
        daily = ports[ports.role == role].groupby("date").export_dry_bulk.sum()
        weekly = daily.resample("W-FRI").sum()
        four = weekly.rolling(4).sum()
        signal = np.log(four / four.shift(52)).replace([np.inf, -np.inf], np.nan)
        df = pd.DataFrame({"x": signal, "y": target}).dropna()
        r2, n = oos_r2(df.x, df.y, pd.Timestamp("2021-06-01"))
        same = df.x.corr(np.log(price).diff(4).reindex(df.index))
        rows.append(dict(ports=role, oos_r2=r2, weeks=n, corr_with_next_4w=df.x.corr(df.y),
                         corr_with_past_4w=same))
    pred = pd.DataFrame(rows)
    pred.to_csv(os.path.join(RES, "prediction.csv"), index=False)
    print(pred.round(4))

    #-------- 3. Grasberg -----------------------------------------------------------------
    a = ports[ports.portname == "Amamapare"].set_index("date")[["export_dry_bulk", "portcalls_dry_bulk"]]
    a.loc["2025-06-01":"2026-01-31"].to_csv(os.path.join(RES, "amamapare.csv"))
    before = a.loc["2025-06-08":"2025-09-07"].export_dry_bulk.mean()
    after = a.loc["2025-09-08":"2025-12-07"].export_dry_bulk.mean()
    print(f"Amamapare dry-bulk exports: {before:,.0f} t/day in the 3 months before, {after:,.0f} after")
    stats.update(amamapare_before_t_per_day=before, amamapare_after_t_per_day=after)
    json.dump(stats, open(os.path.join(RES, "validation_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
