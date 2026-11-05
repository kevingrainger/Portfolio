#!/usr/bin/env python3
#-------- seasonal_test.py ----------------------------------------------------------
#-----------------------------------------------------------------------------
# Does ECMWF's seasonal forecast help predict the Italian power price, once gas is out?
#
#   forecast   SEAS5, issued each month: mean 2 m temperature over northern Italy for the
#              next three months, as an anomaly against the 2008-2022 forecasts of the same
#              issue month (a recent reference, since the region has warmed)
#   targets    for the same three months: the level of power relative to gas,
#              log(price / gas), and its volatility (the standard deviation of daily changes),
#              each as an anomaly for the time of year
#   relation   learned on 2018-2022 from the observed (ERA5) temperature anomaly of those
#              months; a squared term lets both cold and heat matter
#   test       2023-2026: the forecast fed through the relation, against climatology (a
#              normal season, anomaly zero); skill = 1 - MSE(forecast) / MSE(climatology)
#
#   python seasonal_test.py
#
# Output: results/seasonal_test.json, results/seasonal_test.csv, figures/fig11_seasonal.png

import json
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
SEAS = os.path.join(HERE, "data", "seas5")
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"


def seas5():
    rows = []
    for f in sorted(os.listdir(SEAS)):
        if not f.endswith(".grib"):
            continue
        ds = xr.open_dataset(os.path.join(SEAS, f), engine="cfgrib", backend_kwargs={"indexpath": ""})
        m = ds["t2m"].mean(dim=["number", "latitude", "longitude"]).to_dataframe().reset_index().dropna(subset=["t2m"])
        m["lead"] = (m.step.dt.days / 30.44).round().astype(int)
        rows.append(m[["time", "lead", "t2m"]])
    s = pd.concat(rows)
    g = s[s.lead.between(1, 3)].groupby("time").agg(t=("t2m", "mean"), leads=("lead", "nunique"))
    g = g[g.leads == 3].t
    g.index = g.index + pd.offsets.MonthEnd(0)
    ref = g["2008":"2022"]                      # a recent reference with no test years: the region has warmed since 1993-2016
    clim = ref.groupby(ref.index.month)
    return ((g - clim.mean().reindex(g.index.month).values) / clim.std().reindex(g.index.month).values).rename("fc")


def observed():
    d = pd.read_csv(os.path.join(HERE, "data", "real_daily.csv"), index_col=0, parse_dates=True)
    rel = np.log(d.price / d.gas)
    m = pd.DataFrame({"temp": d.temp.resample("ME").mean(), "level": rel.resample("ME").mean(),
                      "vol": rel.diff().resample("ME").std()})
    nxt = m.rolling(3).mean().shift(-3)                     # the three months after each issue month
    for c in nxt:
        nxt[c] = nxt[c] - nxt[c].groupby(nxt.index.month).transform("mean")
    nxt["temp"] = nxt["temp"] / nxt["temp"].std()
    return nxt


def main():
    df = observed().join(seas5(), how="left")
    train, test = df["2018":"2022"].dropna(subset=["temp", "level", "vol"]), df["2023":].dropna()
    out = dict(train_months=len(train), test_months=len(test),
               forecast_temperature_correlation=float(df[["fc", "temp"]].dropna().corr().iloc[0, 1]), targets={})
    rows = []
    for target in ["level", "vol"]:
        X = np.column_stack([np.ones(len(train)), train.temp, train.temp ** 2])
        b = np.linalg.lstsq(X, train[target], rcond=None)[0]
        pred = b[0] + b[1] * test.fc + b[2] * test.fc ** 2
        clim = pd.Series(b[0] + b[2] * 1.0, index=test.index)        # a normal season: E[temp^2] = 1
        mse_f, mse_c = ((pred - test[target]) ** 2).mean(), ((clim - test[target]) ** 2).mean()
        hit = float((np.sign(pred - clim) == np.sign(test[target] - clim)).mean())
        out["targets"][target] = dict(coef=b.tolist(), skill_vs_climatology=float(1 - mse_f / mse_c),
                                      direction_right=hit, corr_forecast_actual=float(np.corrcoef(pred, test[target])[0, 1]))
        rows.append(pd.DataFrame({"target": target, "pred": pred, "clim": clim, "actual": test[target]}))
    pd.concat(rows).to_csv(os.path.join(RES, "seasonal_test.csv"))
    json.dump(out, open(os.path.join(RES, "seasonal_test.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharex=True)
    for ax, (target, title) in zip(axes, [("level", "Power relative to gas, next three months"),
                                          ("vol", "Volatility of power relative to gas, next three months")]):
        r = pd.concat(rows); r = r[r.target == target]
        ax.plot(r.index, r.actual, color=NAVY, lw=1.6, marker="o", ms=3, label="actual")
        ax.plot(r.index, r.pred, color=RED, lw=1.4, label="from the ECMWF forecast")
        ax.plot(r.index, r.clim, color=GREY, lw=1, ls="--", label="climatology")
        ax.set_title(title, loc="left"); ax.set_ylabel("anomaly for the time of year")
        sk = out["targets"][target]["skill_vs_climatology"]
        ax.text(0.02, 0.04, f"skill against climatology: {sk:+.2f}", transform=ax.transAxes, fontsize=9, color=NAVY)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    import matplotlib.dates as mdates
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.savefig(os.path.join(FIG, "fig11_seasonal.png"), dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
