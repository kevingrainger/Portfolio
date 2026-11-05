#!/usr/bin/env python3
#-------- heat_lags.py --------------------------------------------------------------
#-----------------------------------------------------------------------------
# Heat intensity, duration and lags. Kevin's question: how does the intensity of a
# smelter's heat signal, and how long a drop in it lasts, relate to exports and to the
# copper price, and at what lag?
#
#   intensity  each smelter's monthly hot-pixel index from smelter_series.py, as a ratio to
#              its own median; a Chilean index averages the Chilean smelters
#   lags       correlation of the Chilean index's year-on-year change with Chilean
#              dry-bulk exports from the copper ports (PortWatch) and with COMEX copper
#              returns, from the heat leading by 6 months to lagging by 6, with 95% bands
#   duration   a low-heat episode is a run of months with a smelter below half its median;
#              for every episode in the world set, the copper price change over the
#              following 1-3 months against all other months
#
#   python heat_lags.py
#
# Output: results/heat_lags.json, figures/fig10_heat_lags.png

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"
CHILE = ["Altonorte (Glencore)", "Caletones (Codelco)", "Chagres (Anglo American)", "Chuquicamata (Codelco)"]
PORTS = {"port56", "port728", "port1031", "port1034", "port90", "port1058", "port2107"}    # Chilean copper ports
rng = np.random.default_rng(0)


def main():
    m = pd.read_csv(os.path.join(HERE, "results", "smelter_monthly.csv"), parse_dates=["date"]).set_index("date")
    rel = m / m.median()
    chile = rel[CHILE].mean(axis=1)
    hg = pd.read_csv(os.path.join(HERE, "data", "hg_front.csv"), parse_dates=["Date"]).set_index("Date").HG.resample("MS").last()
    ret = np.log(hg).diff()
    p = pd.read_csv(os.path.join(HERE, "data", "ports_daily.csv"), parse_dates=["date"])
    ex = p[p.portid.isin(PORTS)].groupby("date").export_dry_bulk.sum().resample("MS").sum()
    yoy = lambda s: np.log(s.replace(0, np.nan)).diff(12)
    heat_y, ex_y = yoy(chile), yoy(ex)
    lags = range(-6, 7)
    rows = []
    for name, target in [("Chilean copper-port exports", ex_y), ("COMEX copper return", ret)]:
        for L in lags:
            d = pd.concat([heat_y.shift(L), target], axis=1).dropna()
            r = d.corr().iloc[0, 1] if len(d) > 12 else np.nan
            rows.append(dict(target=name, lag=L, r=float(r), band=float(1.96 / np.sqrt(max(len(d) - 3, 1))), n=len(d)))
    lagdf = pd.DataFrame(rows)

    # low-heat episodes in every smelter, and the price after them
    starts = []
    durations = []
    for c in m.columns.drop(["world", "china"]):
        s = rel[c].dropna()
        low = s < 0.5
        run = 0
        for t, v in low.items():
            if v:
                run += 1
                if run == 1:
                    starts.append(t)
            else:
                if run:
                    durations.append(run)
                run = 0
    fwd = {h: np.log(hg).shift(-h) - np.log(hg) for h in (1, 2, 3)}
    ep = {}
    for h, f in fwd.items():
        after = f.reindex(starts).dropna()
        allm = f.dropna()
        boot = [rng.choice(allm.values, len(after)).mean() for _ in range(4000)]
        ep[h] = dict(n=len(after), mean_after=float(after.mean()), mean_all=float(allm.mean()),
                     p_value=float((np.array(boot) >= after.mean()).mean()))
    out = dict(lags=lagdf.to_dict(orient="records"), episodes=len(starts),
               duration_months=dict(median=float(np.median(durations)) if durations else None,
                                    share_one_month=float(np.mean(np.array(durations) == 1)) if durations else None),
               price_after_low_heat=ep)
    json.dump(out, open(os.path.join(HERE, "results", "heat_lags.json"), "w"), indent=1)
    print(lagdf.round(2).to_string(index=False)); print(json.dumps({k: v for k, v in out.items() if k != "lags"}, indent=1))

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True)
    for ax, (name, col) in zip(axes, [("Chilean copper-port exports", BLUE), ("COMEX copper return", RED)]):
        d = lagdf[lagdf.target == name]
        ax.axhspan(-d.band.mean(), d.band.mean(), color="0.92", lw=0)
        ax.vlines(d.lag, 0, d.r, color=col, lw=1, alpha=0.5)
        ax.plot(d.lag, d.r, "-o", color=col, lw=1.4, ms=6, mec="white", mew=1)
        out_band = d[abs(d.r) > d.band]
        ax.plot(out_band.lag, out_band.r, "o", color=col, ms=9, mfc="none", mew=1.4)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xlabel("months the heat signal leads (+) or lags (-)")
        ax.set_title(f"Chilean smelter heat vs {name.split(' ', 1)[1] if name.startswith('COMEX') else 'port exports'}", loc="left")
    axes[0].set_ylabel("correlation (grey: 95% band)")
    fig.savefig(os.path.join(HERE, "figures", "fig10_heat_lags.png"), dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
