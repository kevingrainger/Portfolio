#-------- coffee_data.py -----------------------------------------------------------
#-----------------------------------------------------------------------------
# Loading and shaping 013's data: futures curves, and a frost index from forecasts.
#
# Frost index. ERA5 and forecast models give the minimum air temperature at 2 m over a
# 25 km grid cell. Leaves freeze at ground level in hollows on clear nights, several
# degrees colder than that, so a grid-cell minimum of about 5 C is already frost
# weather in the coffee belt (in July 2021, damaging frost came with ERA5 grid minima of
# 2-5 C in the Sul de Minas). The index is the production-weighted cold excess
#     C(day) = sum over regions r of  w_r * max(0, 5 - Tmin_r(day))      (degree C)
# For a forecast issued on day t, the warning is the largest C forecast for the next
# seven days:
#     W(t) = max over n = 1..7 of C(t + n | issued t)

import os

import numpy as np
import pandas as pd

from coffee_regions import REGIONS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
T0 = 5.0
WEIGHTS = {k: v[2] for k, v in REGIONS.items()}
DELIVERY = {"H": 3, "K": 5, "N": 7, "U": 9, "Z": 12}


def front_month():
    s = pd.read_csv(os.path.join(DATA, "kc_front.csv"), index_col=0, parse_dates=True).squeeze().dropna()
    return s[s > 0]


def cold_actual():
    """C(day) from ERA5, 1990-."""
    e = pd.read_csv(os.path.join(DATA, "weather_era5.csv"), parse_dates=["time"])
    e["c"] = np.maximum(0, T0 - e.temperature_2m_min) * e.region.map(WEIGHTS)
    return e.groupby("time").c.sum()


def warning(model="gfs"):
    """W(t), the frost warning in the forecasts issued on day t (gfs from 2021, ecmwf from 2024).
    Also returns the full table C(t + n | t) by lead n."""
    d = pd.read_csv(os.path.join(DATA, f"forecast_{model}.csv"), parse_dates=["date"])
    w = d.region.map(WEIGHTS)
    out = {}
    for n in range(1, 8):
        c = (np.maximum(0, T0 - d[f"tmin_lead{n}"]) * w).groupby(d.date).sum(min_count=1)
        c.index = c.index - pd.Timedelta(days=n)                  # re-index by issue date
        out[n] = c
    table = pd.DataFrame(out)
    return table.max(axis=1, skipna=False).dropna(), table


def contract_maturity(ticker):
    """Approximate expiry of an ICE coffee contract: mid-month of its delivery month."""
    code, yy = ticker[2], int(ticker[3:5])
    return pd.Timestamp(2000 + yy, DELIVERY[code], 15)


def front_maturity(date):
    """The contract KC=F follows on `date`: the first delivery month whose first notice day
    (about seven business days before the month starts) has not yet passed."""
    for y in (date.year, date.year + 1):
        for m in sorted(DELIVERY.values()):
            first = pd.Timestamp(y, m, 1)
            if date < first - pd.Timedelta(days=10):
                return pd.Timestamp(y, m, 15)


def curve_panel():
    """Log futures prices and maturities (years) for each day: the front month plus every
    live contract. Returns (dates, logF array, tau array), NaN where a contract has no price."""
    c = pd.read_csv(os.path.join(DATA, "kc_contracts.csv"), index_col=0, parse_dates=True)
    front = front_month().reindex(c.index)
    mats = [contract_maturity(t) for t in c.columns]
    logF = np.log(np.column_stack([front.values, c.values]))
    tau = np.empty_like(logF)
    for i, d in enumerate(c.index):
        tau[i, 0] = (front_maturity(d) - d).days / 365.25
        tau[i, 1:] = [(m - d).days / 365.25 for m in mats]
    # the front month and a live contract can be the same contract: keep one
    for j in range(1, logF.shape[1]):
        same = np.isclose(tau[:, 0], tau[:, j])
        logF[same, j] = np.nan
    logF[tau <= 0.02] = np.nan
    return c.index, logF, tau
