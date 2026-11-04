#!/usr/bin/env python3
#-------- weather_signal.py ---------------------------------------------------------
#-----------------------------------------------------------------------------
# Weather against the two coffee prices, once the currency is taken out.
#
#   1. Monthly log returns of arabica and robusta (World Bank Pink Sheet), 2004-2026.
#   2. Strip the standard driver first: regress each return on the month's change in
#      Brazilian reais per dollar and keep the residual, the move the real leaves.
#   3. Weather hazards for the two origins, from ERA5: Brazil's arabica belt (frost,
#      three-month dryness) and Vietnam's robusta highlands (three-month dryness and
#      heat), plus El Nino (NOAA ONI). Each is correlated with each bean's residual at
#      lags of 0-6 months with corrlib's Gaussian-rank correlation, and its 95% band
#      from the effective sample size.
#   4. Slow weather: six-month residual returns against season-long (six-month) dryness,
#      since a monthly test credits a drought only in the month it moves.
#   5. The events: the largest residual moves of each bean and the weather around them.
#
#   python weather_signal.py
#
# Output: results/weather_signal.json, results/weather_monthly.csv, results/lag_correlations.csv

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "004-corrlib-correlation-toolbox"))
from corrlib import correlation_measures as cm                     # noqa: E402
from corrlib import sampling_estimators as se                      # noqa: E402

from coffee_data import DATA, WEIGHTS, cold_actual                 # noqa: E402
from fetch_data import VIETNAM                                     # noqa: E402

RES = os.path.join(HERE, "results")


def pink_sheet():
    x = pd.read_excel(os.path.join(DATA, "wb_pinksheet_monthly.xlsx"), sheet_name="Monthly Prices", header=4)
    x = x.rename(columns={x.columns[0]: "month"}).dropna(subset=["month"])
    x = x[x.month.astype(str).str.match(r"^\d{4}M\d{2}$")]
    idx = pd.to_datetime(x.month.str.replace("M", "-") + "-01") + pd.offsets.MonthEnd(0)
    return pd.DataFrame({"arabica": pd.to_numeric(x["Coffee, Arabica"], errors="coerce").values,
                         "robusta": pd.to_numeric(x["Coffee, Robusta"], errors="coerce").values}, index=idx)


def oni():
    o = pd.read_csv(os.path.join(DATA, "oni.txt"), sep=r"\s+")
    centre = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8,
              "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}
    idx = pd.to_datetime(dict(year=o.YR, month=o.SEAS.map(centre), day=1)) + pd.offsets.MonthEnd(0)
    return pd.Series(o.ANOM.values, index=idx, name="enso")


def brl():
    path = os.path.join(DATA, "brl_usd.csv")
    if not os.path.exists(path):
        import yfinance as yf
        s = yf.download("BRL=X", start="2003-01-01", progress=False, auto_adjust=True)["Close"].squeeze()
        s.to_csv(path)
    s = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
    return s[s > 0.5]


def rain_heat(path, weights, months):
    e = pd.read_csv(os.path.join(DATA, path), parse_dates=["time"])
    e["w"] = e.region.map(weights)
    m = e.set_index("time").groupby("region").resample("ME").agg({"precipitation_sum": "sum", "w": "first"})
    rain = (m.precipitation_sum * m.w).groupby(level="time").sum().rolling(months).sum()
    out = {"dry": -rain.groupby(rain.index.month).transform(lambda s: (s - s.mean()) / s.std())}
    if "temperature_2m_max" in e:
        t = e.set_index("time").groupby("region").resample("ME").temperature_2m_max.mean()
        t = t.groupby(level="time").mean().rolling(months).mean()
        out["heat"] = t.groupby(t.index.month).transform(lambda s: (s - s.mean()) / s.std())
    return pd.DataFrame(out)


def strip(ret, x):
    X = np.column_stack([np.ones(len(x)), x])
    b = np.linalg.lstsq(X, ret, rcond=None)[0]
    return ret - X @ b, b


def corr_band(x, y):
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    r = cm.gaussian_rank(np.vstack([x, y]))[0, 1]
    n_eff = se.effective_n_pair(x, y)
    return float(r), float(1.96 / np.sqrt(max(n_eff - 3, 1))), float(n_eff)


def full_panel():
    """Every month from 2004, weather and ENSO complete; the real has a few gaps from Yahoo."""
    br3 = rain_heat("weather_era5.csv", WEIGHTS, 3)
    vn3 = rain_heat("weather_vietnam.csv", {k: v[2] for k, v in VIETNAM.items()}, 3)
    fx = np.log(brl().resample("ME").last()).diff().rename("brl")
    return pd.DataFrame({"brl": fx, "brazil_dry": br3.dry, "vietnam_dry": vn3.dry,
                         "vietnam_heat": vn3.heat, "enso": oni()})["2004-01":"2026-09"]


def main():
    os.makedirs(RES, exist_ok=True)
    px = pink_sheet()["2003-12":]
    ret = np.log(px).diff()
    fx = np.log(brl().resample("ME").last()).diff().rename("brl")
    br3 = rain_heat("weather_era5.csv", WEIGHTS, 3)
    vn3 = rain_heat("weather_vietnam.csv", {k: v[2] for k, v in VIETNAM.items()}, 3)
    df = pd.DataFrame({"arabica": ret.arabica, "robusta": ret.robusta, "brl": fx,
                       "brazil_frost": cold_actual().resample("ME").max(),
                       "brazil_dry": br3.dry, "vietnam_dry": vn3.dry, "vietnam_heat": vn3.heat,
                       "enso": oni()}).dropna()["2004-01":]

    # 2. the real out first
    real = {}
    for bean in ["arabica", "robusta"]:
        res, b = strip(df[bean].values, df.brl.values)
        df[f"{bean}_res"] = res
        r2 = 1 - res.var() / df[bean].var()
        real[bean] = dict(beta=float(b[1]), r2=float(r2))

    # 3. hazards against residuals, lags 0-6 (hazard leads the price)
    hazards = ["brazil_frost", "brazil_dry", "vietnam_dry", "vietnam_heat", "enso"]
    rows = []
    for bean in ["arabica", "robusta"]:
        y = df[f"{bean}_res"]
        for h in hazards:
            for lag in range(7):
                r, band, n_eff = corr_band(df[h].shift(lag).values, y.values)
                rows.append(dict(bean=bean, hazard=h, lag=lag, r=r, band=band, n_eff=n_eff,
                                 significant=abs(r) > band))
    lags = pd.DataFrame(rows)
    lags.to_csv(os.path.join(RES, "lag_correlations.csv"), index=False)

    # 4. slow weather: six-month residual returns against six-month dryness (non-overlapping)
    br6 = rain_heat("weather_era5.csv", WEIGHTS, 6)
    vn6 = rain_heat("weather_vietnam.csv", {k: v[2] for k, v in VIETNAM.items()}, 6)
    slow = {}
    for bean, hz in [("arabica", br6.dry), ("robusta", vn6.dry), ("arabica_vs_vietnam", vn6.dry), ("robusta_vs_brazil", br6.dry)]:
        b = bean.split("_")[0]
        r6 = df[f"{b}_res"].rolling(6).sum().shift(-6)            # the next six months
        z = pd.concat([hz.reindex(df.index), r6], axis=1).dropna().iloc[::6]
        r, band, n_eff = corr_band(z.iloc[:, 0].values, z.iloc[:, 1].values)
        slow[bean] = dict(r=r, band=band, n=len(z))

    # 5. events: the ten largest residual months for each bean, with the weather around them
    events = {}
    for bean in ["arabica", "robusta"]:
        top = df[f"{bean}_res"].abs().nlargest(10).index
        events[bean] = [dict(month=str(t.date())[:7], residual=float(df.loc[t, f"{bean}_res"]),
                             raw=float(df.loc[t, bean]), brl=float(df.loc[t, "brl"]),
                             **{h: float(df.loc[t, h]) for h in hazards}) for t in sorted(top)]

    # how the two beans move together, before and after the real is taken out
    together = dict(raw=corr_band(df.arabica.values, df.robusta.values)[0],
                    residual=corr_band(df.arabica_res.values, df.robusta_res.values)[0])

    df.to_csv(os.path.join(RES, "weather_monthly.csv"))
    best = lags.loc[lags.groupby(["bean", "hazard"]).r.apply(lambda s: s.abs().idxmax())]
    out = dict(months=len(df), start=str(df.index[0].date()), end=str(df.index[-1].date()),
               real=real, beans_together=together, slow_weather=slow,
               strongest_lag=best.to_dict(orient="records"), events=events)
    json.dump(out, open(os.path.join(RES, "weather_signal.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ["months", "real", "beans_together", "slow_weather", "strongest_lag"]}, indent=1))


if __name__ == "__main__":
    main()
