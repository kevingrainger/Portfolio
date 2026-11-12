#!/usr/bin/env python3
#-------- daily_drivers.py ----------------------------------------------------------
#-----------------------------------------------------------------------------
# Day-to-day, how much of the Italian power price (relative to gas) do heat, renewables
# and demand move? The price relative to gas, log(price / gas), is taken as an anomaly
# against its own 30-day centred mean, so slow regimes such as the 2022 gas crisis drop out
# and only the day-to-day sway is left. The drivers are anomalies the same way:
#
#   heat      degrees above 24 C          cold     degrees below 12 C
#   renewables  wind and solar share of load    load   demand relative to normal
#   weekend
#
# A linear model is learned on 2018-2022 and scored on 2023-2026 (out-of-sample R^2), and
# the share of each driver is shown by the drop in R^2 when it is left out.
#
#   python daily_drivers.py
#
# Output: results/daily_drivers.json, figures/fig12_daily_drivers.png

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"


def anomaly(s, w=30):
    return s - s.rolling(w, center=True, min_periods=15).mean()


def main():
    d = pd.read_csv(os.path.join(HERE, "data", "real_daily.csv"), index_col=0, parse_dates=True)
    X = pd.DataFrame({"heat": anomaly((d.temp - 24).clip(lower=0)), "cold": anomaly((12 - d.temp).clip(lower=0)),
                      "renewables": anomaly(d.renewables), "load": anomaly(d.load), "weekend": anomaly(d.weekend.astype(float))})
    y = anomaly(np.log(d.price.clip(lower=1) / d.gas)).rename("y")
    df = X.join(y).join(anomaly(d.temp).rename("temp")).dropna()
    tr, te = df[:"2022"], df["2023":]
    cols = list(X.columns)

    def fit_score(c):
        A = np.column_stack([np.ones(len(tr))] + [tr[k] for k in c])
        b = np.linalg.lstsq(A, tr.y, rcond=None)[0]
        P = np.column_stack([np.ones(len(te))] + [te[k] for k in c]) @ b
        return 1 - ((te.y - P) ** 2).sum() / ((te.y - te.y.mean()) ** 2).sum(), b, P

    r2, b, pred = fit_score(cols)
    drop = {k: float(r2 - fit_score([c for c in cols if c != k])[0]) for k in cols}
    corr = {k: float(df[k].corr(df.y)) for k in cols + ["temp"]}
    out = dict(train_days=len(tr), test_days=len(te), r2_out_of_sample=float(r2),
               coef=dict(zip(["const"] + cols, map(float, b))), r2_drop_without=drop, correlation=corr)

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1], hspace=0.45, wspace=0.3)
    for k, (c, col, lab) in enumerate([("temp", RED, "temperature anomaly, June-September (°C)"), ("renewables", BLUE, "wind and solar share anomaly"),
                                       ("load", NAVY, "demand anomaly")]):
        ax = fig.add_subplot(gs[0, k])
        dd = df[df.index.month.isin([6, 7, 8, 9])] if c == "temp" else df          # heat acts in summer
        ax.scatter(dd[c], dd.y, s=4, color=col, alpha=0.25, lw=0)
        q = pd.qcut(dd[c].rank(method="first"), 20)
        m = dd.groupby(q, observed=True)[[c, "y"]].mean()
        if c == "temp":
            corr["temp_summer"] = float(dd[c].corr(dd.y))
        ax.plot(m[c], m.y, color="k", lw=1.6)
        r_show = corr.get("temp_summer") if c == "temp" else corr[c]
        ax.set_xlabel(lab); ax.set_title(f"{'Summer heat' if c == 'temp' else c.capitalize()}, r = {r_show:+.2f}", loc="left")
        if k == 0:
            ax.set_ylabel("power relative to gas, daily anomaly")
        ax.set_ylim(np.percentile(df.y, 0.5), np.percentile(df.y, 99.5))
    ax = fig.add_subplot(gs[1, :])
    t = te["2025-05":"2025-09"].index
    ax.plot(t, te.y.loc[t], color=NAVY, lw=1.3, label="actual")
    ax.plot(t, pd.Series(pred, index=te.index).loc[t], color=RED, lw=1.3, label="from heat, renewables, demand and weekends")
    ax.axhline(0, color="0.8", lw=0.8)
    ax.set_ylabel("daily anomaly"); ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title(f"Summer 2025, out of sample (R² over 2023-2026 = {r2:.2f})", loc="left")
    fig.savefig(os.path.join(HERE, "figures", "fig12_daily_drivers.png"), dpi=170, bbox_inches="tight"); plt.close(fig)
    out["correlation"] = corr
    json.dump(out, open(os.path.join(HERE, "results", "daily_drivers.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
