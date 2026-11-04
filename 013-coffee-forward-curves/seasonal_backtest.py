#!/usr/bin/env python3
#-------- seasonal_backtest.py ------------------------------------------------------
#-----------------------------------------------------------------------------
# Can an importer use ECMWF's seasonal forecast to decide when to buy coffee?
#
#   forecast   SEAS5, issued each month: ensemble-mean rain and temperature over Brazil's
#              arabica belt and Vietnam's robusta highlands for the next three months,
#              as anomalies against the 1993-2016 hindcast of the same issue month
#   relation   learned only from data before each decision (expanding window): the next
#              three months' residual return of a bean (real removed) regressed on the
#              observed (ERA5) weather of those three months
#   decision   feed the forecast through the relation; if it predicts a rise, buy forward
#              now (lock today's price), otherwise wait and buy in three months
#   scoring    hit rate against the actual three-month price change, and the average
#              saving per decision against always waiting; compared with the same rule
#              driven by persistence (the last three months' observed weather) instead of
#              the forecast, so the forecast's own contribution shows
#
# Decisions are made every three months so outcomes don't overlap. Tested 2008-2026.
#
#   python seasonal_backtest.py
#
# Output: results/seasonal_backtest.json, results/seasonal_decisions.csv

import json
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from coffee_data import WEIGHTS
from fetch_data import VIETNAM
from weather_signal import pink_sheet, rain_heat, brl

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
SEAS = os.path.join(HERE, "data", "seas5")
RES = os.path.join(HERE, "results")
TEST_START = "2008-01"


def load_seas5(origin):
    rows = []
    for f in sorted(os.listdir(SEAS)):
        if not (f.startswith(origin) and f.endswith(".grib")):
            continue
        ds = xr.open_dataset(os.path.join(SEAS, f), engine="cfgrib", backend_kwargs={"indexpath": ""})
        m = ds[["tprate", "t2m"]].mean(dim=["number", "latitude", "longitude"]).to_dataframe().reset_index()
        m = m.dropna(subset=["tprate", "t2m"])
        m["lead"] = (m.step.dt.days / 30.44).round().astype(int)
        rows.append(m[["time", "lead", "tprate", "t2m"]])
    s = pd.concat(rows)
    s = s[s.lead.between(1, 3)]
    g = s.groupby("time").agg(rain=("tprate", "mean"), t2m=("t2m", "mean"), leads=("lead", "nunique"))
    g = g[g.leads == 3]
    g.index = g.index + pd.offsets.MonthEnd(0)
    clim = g["1993":"2016"].groupby(g["1993":"2016"].index.month)
    mu, sd = clim.mean(), clim.std()
    z = (g[["rain", "t2m"]] - mu.loc[g.index.month].values[:, :2]) / sd.loc[g.index.month].values[:, :2]
    return z.rename(columns={"rain": f"{origin}_rain_z", "t2m": f"{origin}_t2m_z"})


def build():
    px = pink_sheet()
    lp = np.log(px)
    fx = np.log(brl().resample("ME").last()).diff().reindex(px.index).fillna(0)
    ret = lp.diff()
    df = pd.DataFrame(index=px.index)
    for bean in ["arabica", "robusta"]:
        ok = ret[bean].notna()
        X = np.column_stack([np.ones(ok.sum()), fx[ok]])
        b = np.linalg.lstsq(X, ret[bean][ok], rcond=None)[0]
        res = ret[bean] - b[0] - b[1] * fx
        df[f"{bean}_next3_res"] = res.rolling(3).sum().shift(-3)      # the next three months, real out
        df[f"{bean}_next3_raw"] = (lp[bean].shift(-3) - lp[bean])     # what the importer actually pays
    br = rain_heat("weather_era5.csv", WEIGHTS, 3)
    vn = rain_heat("weather_vietnam.csv", {k: v[2] for k, v in VIETNAM.items()}, 3)
    obs = pd.DataFrame({"brazil_dry": br.dry, "vietnam_dry": vn.dry, "vietnam_heat": vn.heat})
    for c in obs:
        df[f"{c}_next3"] = obs[c].shift(-3)            # observed weather of the next three months
        df[f"{c}_last3"] = obs[c]                      # persistence: the last three months
    fb, fv = load_seas5("brazil"), load_seas5("vietnam")
    df["brazil_dry_fc"] = -fb.brazil_rain_z
    df["vietnam_dry_fc"] = -fv.vietnam_rain_z
    df["vietnam_heat_fc"] = fv.vietnam_t2m_z
    return df


def backtest(df, bean, hazards, source):
    """Expanding-window relation on observed weather; decide on `source` (fc or last3)."""
    dates = df[TEST_START:].index[::3]
    rows = []
    for t in dates:
        train = df[:t - pd.offsets.MonthEnd(3)].dropna(subset=[f"{bean}_next3_res"] + [f"{h}_next3" for h in hazards])
        x_now = df.loc[t, [f"{h}_{source}" for h in hazards]]
        if len(train) < 36 or x_now.isna().any() or np.isnan(df.loc[t, f"{bean}_next3_raw"]):
            continue
        X = np.column_stack([np.ones(len(train)), train[[f"{h}_next3" for h in hazards]].values])
        b = np.linalg.lstsq(X, train[f"{bean}_next3_res"].values, rcond=None)[0]
        pred = b[0] + x_now.values @ b[1:]
        actual = df.loc[t, f"{bean}_next3_raw"]
        hedge = pred > 0
        rows.append(dict(month=t, pred=pred, actual=actual, hedge=hedge,
                         saving=actual if hedge else 0.0))       # vs always waiting
    d = pd.DataFrame(rows)
    up = d.actual > 0
    return d, dict(decisions=len(d), hedged_share=float(d.hedge.mean()),
                   hit_rate=float((d.hedge == up).mean()), base_rate_up=float(up.mean()),
                   mean_saving_vs_wait=float(d.saving.mean()),
                   mean_saving_always_hedge=float(d.actual.mean()),
                   saving_when_hedged=float(d.loc[d.hedge, "actual"].mean()) if d.hedge.any() else None,
                   saving_when_waited=float(d.loc[~d.hedge, "actual"].mean()) if (~d.hedge).any() else None)


def main():
    df = build()
    skill = {h: float(df[[f"{h}_fc", f"{h}_next3"]].dropna().corr().iloc[0, 1])
             for h in ["brazil_dry", "vietnam_dry", "vietnam_heat"]}
    skill_persist = {h: float(df[[f"{h}_last3", f"{h}_next3"]].dropna().corr().iloc[0, 1])
                     for h in ["brazil_dry", "vietnam_dry", "vietnam_heat"]}
    out = dict(forecast_skill=skill, persistence_skill=skill_persist, results={})
    keep = []
    setups = {"arabica": ["brazil_dry", "vietnam_heat"], "robusta": ["vietnam_heat", "vietnam_dry", "brazil_dry"]}
    for bean, hz in setups.items():
        for source in ["fc", "last3"]:
            d, s = backtest(df, bean, hz, source)
            out["results"][f"{bean}|{'forecast' if source == 'fc' else 'persistence'}"] = s
            keep.append(d.assign(bean=bean, source=source))
    pd.concat(keep).to_csv(os.path.join(RES, "seasonal_decisions.csv"), index=False)
    json.dump(out, open(os.path.join(RES, "seasonal_backtest.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
