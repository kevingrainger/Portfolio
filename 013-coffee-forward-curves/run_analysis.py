#!/usr/bin/env python3
#-------- run_analysis.py ----------------------------------------------------------
#-----------------------------------------------------------------------------
# Three questions, in order.
#
# 1. When does the price move: with the forecast, or with the frost?
#    Event study on the front-month contract. Events are
#      - frost nights in ERA5, 1990-2026: C(day) >= 0.3 after 30 quiet days;
#      - frost warnings in GFS forecasts, 2021-2026: W(t) >= 0.15 after 10 quiet days.
#    For each, the cumulative log return from 10 trading days before to 10 after.
#
# 2. Is a new warning priced the same day? (Roll's 1984 test for orange juice.) In the
#    frost season (May-August), regress each day's return on the change in that day's
#    warning and on the two previous days' changes. Forecasts from the 00 UTC run are
#    out before ICE opens, so a market that reads them prices the change the same day
#    and the lagged coefficients are zero. Standard errors: Newey-West (5 lags).
#
# 3. How far along the curve does a frost reach? The Schwartz two-factor model, fitted
#    to the 2024-2026 curves with and without the warning in the convenience yield.
#
#   python run_analysis.py
#
# Output: results/events.csv, results/timing.csv, results/schwartz.json

import json
import sys
import os

import numpy as np
import pandas as pd

from coffee_data import cold_actual, curve_panel, front_month, warning
from schwartz import NAMES, fit, kalman

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
WINDOW = 10


def first_crossings(series, threshold, quiet_days):
    s = series.sort_index()
    above = s[s >= threshold].index
    events, last = [], None
    for d in above:
        if last is None or (d - last).days > quiet_days:
            events.append(d)
        last = d
    return events


def event_paths(price, events, kind):
    lr = np.log(price)
    rows = []
    for e in events:
        i = price.index.searchsorted(e)                       # first trading day on or after the event
        if i - WINDOW < 1 or i + WINDOW >= len(price):
            continue
        base = lr.iloc[i - WINDOW - 1]
        for k in range(-WINDOW, WINDOW + 1):
            rows.append(dict(kind=kind, event=e.date(), k=k, cum=lr.iloc[i + k] - base))
    return rows


def newey_west(X, y, lags=5):
    X = np.column_stack([np.ones(len(y)), X])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    u = y - X @ b
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (X * u[:, None]).T @ (X * u[:, None])
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = (X[L:] * u[L:, None]).T @ (X[:-L] * u[:-L, None])
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    return b, np.sqrt(np.diag(V))


def main():
    os.makedirs(RES, exist_ok=True)
    price = front_month()

    #-------- 1. events --------------------------------------------------------------
    C = cold_actual()
    W, _ = warning("gfs")
    frosts = first_crossings(C, 0.3, 30)
    warns = first_crossings(W, 0.15, 10)
    rows = event_paths(price, frosts, "frost night (ERA5)") + event_paths(price, warns, "first warning (GFS)")
    ev = pd.DataFrame(rows)
    ev.to_csv(os.path.join(RES, "events.csv"), index=False)
    print("frost nights:", [d.date() for d in frosts])
    print("first warnings:", [d.date() for d in warns])

    #-------- 2. timing --------------------------------------------------------------
    r = np.log(price).diff()
    # the warning as it stood on each trading day, so Monday's change includes the
    # forecasts issued over the weekend (24 July 2021 was a Saturday)
    dW = W.reindex(price.index).diff()
    df = pd.DataFrame({"r": r, "dW0": dW, "dW1": dW.shift(1), "dW2": dW.shift(2)}).dropna()
    df = df[df.index.month.isin([5, 6, 7, 8])]
    b, se = newey_west(df[["dW0", "dW1", "dW2"]].values, df.r.values)
    timing = pd.DataFrame(dict(term=["const", "same day", "1 day later", "2 days later"], coef=b, se=se))
    timing["t"] = timing.coef / timing.se
    timing.to_csv(os.path.join(RES, "timing.csv"), index=False)
    print(timing.round(4), f"(n = {len(df)} frost-season days, 2021-2026)")

    #-------- 3. Schwartz with and without weather -------------------------------------
    if os.path.exists(os.path.join(RES, "schwartz.json")) and "--refit" not in sys.argv:
        return
    dates, logF, tau = curve_panel()
    Wd = W.reindex(dates).fillna(0).values
    p0, ll0 = fit(logF, tau)
    p1, ll1 = fit(logF, tau, W=Wd, start=p0)
    _, states = kalman(p0, logF, tau, return_states=True)
    pd.DataFrame(states, index=dates, columns=["lnS", "delta"]).to_csv(os.path.join(RES, "schwartz_states.csv"))
    out = dict(no_weather=dict(zip(NAMES, p0.tolist())), loglik_no_weather=ll0,
               weather=dict(zip(NAMES, p1.tolist())), loglik_weather=ll1, LR=2 * (ll1 - ll0),
               days=len(dates), start=str(dates[0].date()), end=str(dates[-1].date()))
    json.dump(out, open(os.path.join(RES, "schwartz.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
