#!/usr/bin/env python3
#-------- analyse_smelters.py ------------------------------------------------------
#-----------------------------------------------------------------------------
# What the smelter heat series says (run smelter_series.py first).
#
# Activity index. For each site, the mean hot-pixel count over its clear passes (at least
# 80% of the box cloud-free) in each month, divided by the site's average over all months,
# so a big smelter and a small one count alike: 1 = a normal month for that site. The
# China index averages the six Chinese sites, the world index all of them; a month needs
# at least three sites observed.
#
# Some sites are strongly seasonal for reasons that are not production (Xiangguang, in
# Shandong, reads near zero every winter: haze or the low winter sun), so the index is
# compared year on year: the log ratio of its 3-month mean to the same months a year
# earlier, which cancels a site's seasonal cycle.
#
# Test. As for the ships: does that year-on-year change predict the next month's log
# change in COMEX copper? Expanding window from 2021, out-of-sample
# R^2 against a zero forecast.
#
#   python analyse_smelters.py
#
# Output: results/smelter_monthly.csv, results/smelter_test.json, figures/fig4_smelter_heat

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import figstyle                                   # noqa: E402
from analyse_ports import copper_price            # noqa: E402

CLEAR = 0.8
SHOW = ["Caletones (Codelco)", "Chuquicamata (Codelco)", "Yunnan Copper (Kunming)", "Xiangguang Copper (Yanggu)",
        "Mount Isa (Glencore)", "Kansanshi (First Quantum)"]


def monthly():
    d = pd.read_csv(os.path.join(HERE, "data", "smelter_passes.csv"), parse_dates=["date"])
    d = d[(d.hot >= 0) & (d.clear >= CLEAR)]
    sites = pd.read_csv(os.path.join(HERE, "smelters.csv")).set_index("name")
    m = d.groupby(["site", pd.Grouper(key="date", freq="MS")]).hot.mean().unstack(0)
    rel = m / m.mean()
    china = [s for s in rel.columns if sites.loc[s, "country"] == "China"]
    out = pd.DataFrame({"world": rel.mean(axis=1).where(rel.notna().sum(axis=1) >= 3),
                        "china": rel[china].mean(axis=1).where(rel[china].notna().sum(axis=1) >= 3)})
    return m, rel, out


def oos(x, y, start="2021-01-01"):
    pred, act = [], []
    for i in range(len(y)):
        if y.index[i] < pd.Timestamp(start) or i < 24:
            continue
        b, a = np.polyfit(x.iloc[:i - 1], y.iloc[:i - 1], 1)
        pred.append(a + b * x.iloc[i])
        act.append(y.iloc[i])
    pred, act = np.array(pred), np.array(act)
    return 1 - np.sum((act - pred) ** 2) / np.sum(act ** 2), len(act)


def main():
    m, rel, idx = monthly()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    idx.join(rel).to_csv(os.path.join(HERE, "results", "smelter_monthly.csv"))
    price = copper_price().resample("MS").last()
    nxt = (np.log(price).shift(-1) - np.log(price)).rename("next")
    out = {}
    for k in ["world", "china"]:
        sm = idx[k].rolling(3, min_periods=2).mean()
        x = np.log(sm / sm.shift(12)).replace([np.inf, -np.inf], np.nan).rename("x")
        df = pd.concat([x, nxt], axis=1).dropna()
        r2, n = oos(df.x, df["next"])
        out[k] = dict(oos_r2=r2, months=n, corr=df.x.corr(df["next"]))
    out["passes_used"] = int(m.notna().sum().sum())
    json.dump(out, open(os.path.join(HERE, "results", "smelter_test.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))

    figstyle.apply()
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 5.4), sharex=True, sharey=True)
    for ax, s in zip(axes.ravel(), SHOW):
        r = rel[s]
        ax.bar(r.index, r.values, width=25, color="#DDCC77")
        ax.plot(r.index, r.rolling(6, min_periods=3).mean(), color="#882255", lw=1.6)
        ax.axhline(1, color="0.6", lw=0.6)
        ax.set_title(s, fontsize=9, loc="left")
    for ax in axes[:, 0]:
        ax.set_ylabel("hot pixels, site average = 1")
    axes[0, 0].set_ylim(0, 3)
    import matplotlib.dates as mdates
    for ax in axes[1]:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Smelter heat seen by Sentinel-2, clear passes by month (bars) and 6-month mean (line)", x=0.01, ha="left")
    fig.tight_layout()
    figstyle.save(fig, os.path.join(HERE, "figures", "fig4_smelter_heat"))


if __name__ == "__main__":
    main()
