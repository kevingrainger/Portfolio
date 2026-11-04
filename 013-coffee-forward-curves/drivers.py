#!/usr/bin/env python3
#-------- drivers.py ---------------------------------------------------------------
#-----------------------------------------------------------------------------
# How much of arabica's price is weather? Monthly returns of the front month,
# 2000-2026, regressed on four candidate drivers:
#
#   frost         the month's largest production-weighted cold excess C (coffee_data.py)
#   Brazil dry    minus the production-weighted rain of the last three months, as a
#                 z-score against the same calendar months of 1990-2026
#   Vietnam dry   the same for the robusta highlands, with the three-month mean daily
#                 maximum temperature as a heat term
#   real          the month's log change in Brazilian reais per US dollar
#
# The R-squared is split between the four by Shapley values (the average gain from
# adding each driver over every order of adding them), so the shares add up to the
# total. A first look at daily data says how much of the variance sits near frost nights.
#
#   python drivers.py
#
# Output: results/drivers.json, results/drivers_monthly.csv

import itertools
import json
import math
import os

import numpy as np
import pandas as pd

from coffee_data import DATA, WEIGHTS, cold_actual, front_month
from fetch_data import VIETNAM

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def dryness(path, weights):
    """Minus the weighted three-month rain, standardised by calendar month, and the three-month heat anomaly."""
    e = pd.read_csv(os.path.join(DATA, path), parse_dates=["time"])
    e["w"] = e.region.map(weights)
    m = e.set_index("time").groupby("region").resample("ME").agg({"precipitation_sum": "sum", "w": "first"})
    rain = (m.precipitation_sum * m.w).groupby(level="time").sum().rolling(3).sum()
    z = rain.groupby(rain.index.month).transform(lambda s: (s - s.mean()) / s.std())
    out = pd.DataFrame({"dry": -z})
    if "temperature_2m_max" in e:
        t = e.set_index("time").groupby("region").resample("ME").temperature_2m_max.mean()
        t = t.groupby(level="time").mean().rolling(3).mean()
        out["heat"] = t.groupby(t.index.month).transform(lambda s: (s - s.mean()) / s.std())
    return out


def r2(X, y):
    X = np.column_stack([np.ones(len(y)), X])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum(), b


def shapley(groups, df, y):
    names = list(groups)
    val = {}
    for k in range(len(names) + 1):
        for S in itertools.combinations(names, k):
            cols = [c for g in S for c in groups[g]]
            val[S] = r2(df[cols].values, y)[0] if cols else 0.0
    phi = {g: 0.0 for g in names}
    n = len(names)
    for g in names:
        rest = [h for h in names if h != g]
        for k in range(n):
            for S in itertools.combinations(rest, k):
                w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                with_g = tuple(sorted(S + (g,), key=names.index))
                phi[g] += w * (val[with_g] - val[S])
    return phi, val[tuple(names)]


def main():
    os.makedirs(RES, exist_ok=True)
    px = front_month()
    C = cold_actual()

    # daily: share of variance within two days before to five after an ERA5 frost night
    r = np.log(px).diff().dropna()
    near = pd.Series(False, index=r.index)
    for d in C[C >= 0.3].index:
        i = r.index.searchsorted(d)
        near.iloc[max(0, i - 2):i + 5] = True
    frost_share_daily = float((r[near] ** 2).sum() / (r ** 2).sum())

    # monthly panel
    ret = np.log(px.resample("ME").last()).diff()
    brl = pd.read_csv(os.path.join(DATA, "brl_usd.csv"), index_col=0, parse_dates=True).squeeze()
    brl = brl[brl > 0.5]                                             # a few zero prints in the Yahoo series
    br = dryness("weather_era5.csv", WEIGHTS)
    vn = dryness("weather_vietnam.csv", {k: v[2] for k, v in VIETNAM.items()})
    df = pd.DataFrame({"ret": ret, "frost": C.resample("ME").max(), "brazil_dry": br.dry,
                       "vietnam_dry": vn.dry, "vietnam_heat": vn.heat,
                       "brl": np.log(brl.resample("ME").last()).diff()}).dropna()
    df = df["2000-02":]
    df.to_csv(os.path.join(RES, "drivers_monthly.csv"))
    groups = {"frost": ["frost"], "Brazil dry": ["brazil_dry"], "Vietnam dry and heat": ["vietnam_dry", "vietnam_heat"],
              "Brazilian real": ["brl"]}
    phi, total = shapley(groups, df, df.ret.values)
    _, b = r2(df[[c for g in groups.values() for c in g]].values, df.ret.values)
    out = dict(months=len(df), start=str(df.index[0].date()), end=str(df.index[-1].date()),
               frost_share_of_daily_variance=frost_share_daily, frost_days_share=float(near.mean()),
               r2_total=total, shapley=phi,
               coef=dict(zip(["const"] + [c for g in groups.values() for c in g], b.tolist())),
               corr_with_return=df.corr().ret.drop("ret").to_dict())
    json.dump(out, open(os.path.join(RES, "drivers.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
