#!/usr/bin/env python3
#-------- make_figures.py ----------------------------------------------------------
#-----------------------------------------------------------------------------
# The four figures of 015, from the saved results (run backtest.py and monte_carlo.py first).
#
#   fig1_frontier          error against cost, calm and shock trades, all six commodities
#   fig2_copper_2025       one trade through copper's tariff summer, day by day
#   fig3_variance_surprise why every hedge loses in a shock: the volatility it was sold at
#   fig4_monte_carlo       which market feature breaks which hedge
#
# Colours avoid pairing orange with blue (Paul Tol's muted scheme, without its blues
# where an orange-like sand is used).

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import figstyle                                               # noqa: E402
from hedging import hedge_trade, realised_vol                 # noqa: E402

RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
LABELS = {"none": "no hedge", "static": "static delta", "weekly": "weekly delta", "move": "on a 2% move",
          "daily": "daily delta", "garch": "daily, GARCH vol", "leland": "Leland", "band": "Whalley-Wilmott band"}
COLOURS = {"static": "0.55", "weekly": "#88CCEE", "move": "#44AA99", "daily": "#117733",
           "garch": "#882255", "leland": "#999933", "band": "#AA4499"}
COMMODITY_COLOURS = {"copper": "#882255", "gold": "#999933", "wheat": "#DDCC77", "corn": "#117733",
                     "coffee": "#332288", "cocoa": "#AA4499"}


def fig1_frontier():
    summ = pd.read_csv(os.path.join(RES, "summary.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    for ax, regime, title in zip(axes, ["calm", "shock"], ["Calm trades", "Shock trades"]):
        sub = summ[(summ.regime == regime) & (~summ.strategy.isin(["none", "static", "leland"]))]
        for s in ["weekly", "move", "band", "garch", "daily"]:
            r = sub[sub.strategy == s].iloc[0]
            ax.scatter(100 * r.mean_cost, 100 * r.sd_err, s=60, color=COLOURS[s], zorder=3, label=LABELS[s])
        ax.set_title(f"{title} ({int(sub.n.iloc[0])} per strategy)", loc="left")
        ax.set_xlabel("trading cost, % of premium")
        ax.margins(x=0.25, y=0.25)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    axes[0].set_ylabel("hedging error, standard deviation,\n% of premium")
    axes[0].legend(loc="upper right")
    fig.suptitle("Selling 3-month calls on six commodities, 2002-2026: tracking against cost", x=0.01, ha="left")
    fig.tight_layout()
    fig.text(0.01, -0.03, "Not shown: no hedge (error 175% calm, 418% shock), static delta (80%, 176%), and Leland, "
             "which at a 5 bp cost sits on daily delta.", fontsize=8, color="0.4")
    figstyle.save(fig, os.path.join(FIG, "fig1_frontier"))


def fig2_copper_2025():
    prices = pd.read_csv(os.path.join(HERE, "data", "front_months.csv"), index_col=0, parse_dates=True)["HG=F"].dropna()
    ret = np.diff(np.log(prices.values))
    vol = np.concatenate([[np.nan], realised_vol(ret)])
    i = np.searchsorted(prices.index, pd.Timestamp("2025-07-01"))
    n = 63
    F, sig, dates = prices.values[i:i + n + 1], vol[i:i + n], prices.index[i:i + n + 1]
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 7.0), sharex=True, gridspec_kw=dict(height_ratios=[1.1, 1, 1]))
    axes[0].plot(dates, F, color="0.2")
    axes[0].set_ylabel("COMEX copper, US$/lb")
    for d, txt, dy, ha in [("2025-07-08", "tariff announced", (-8, 6), "right"), ("2025-07-31", "refined copper exempted", (8, -18), "left")]:
        j = np.searchsorted(dates, pd.Timestamp(d))
        axes[0].annotate(txt, (dates[j], F[j]), xytext=dy, textcoords="offset points", fontsize=8.5, ha=ha,
                         arrowprops=dict(arrowstyle="-", color="0.5", lw=0.7))
    for s in ["daily", "move", "weekly", "band"]:
        out = hedge_trade(F, sig, n, s, k=5e-4, record=True)
        axes[1].step(dates[:-1], out["path"][:, 0], where="post", color=COLOURS[s], label=LABELS[s])
        axes[2].plot(dates[1:], out["path"][:, 1] / out["premium"], color=COLOURS[s])
    axes[1].set_ylabel("futures held per option")
    axes[1].legend(ncol=2, loc="upper right")
    axes[2].axhline(0, color="0.6", lw=0.7)
    axes[2].set_ylabel("running P&L,\nshare of premium")
    axes[0].set_title("Short a 3-month copper call sold 1 July 2025, hedged four ways", loc="left")
    fig.autofmt_xdate()
    fig.tight_layout()
    figstyle.save(fig, os.path.join(FIG, "fig2_copper_2025"))


def fig3_variance_surprise():
    t = pd.read_csv(os.path.join(RES, "trades.csv"))
    t = t[t.strategy == "daily"].copy()
    t["x"] = (t.sigma_realised ** 2 - t.sigma_sold ** 2) / t.sigma_sold ** 2
    t = t[t.x < 6]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for c, g in t.groupby("commodity"):
        ax.scatter(g.x, g.err, s=9, alpha=0.6, color=COMMODITY_COLOURS[c], label=c, lw=0)
    xs = np.linspace(-1, 6, 50)
    ax.plot(xs, -xs / 2, color="0.15", lw=1.4)
    ax.annotate("theory: error = -1/2 x variance surprise", (3.4, -2.75), fontsize=8.5, ha="center")
    ax.axvline(SHOCK := 1.5 ** 2 - 1, color="0.6", lw=0.8, ls=":")
    ax.annotate("shock trades", (SHOCK, 1.3), xytext=(5, 0), textcoords="offset points", fontsize=8.5, color="0.4")
    ax.set_xlabel("variance surprise (realised variance - variance sold) / variance sold")
    ax.set_ylabel("hedging error, daily delta, share of premium")
    ax.set_title("Hedged daily, a seller still loses what the volatility surprise costs", loc="left")
    ax.legend(loc="upper right", ncol=2, markerscale=2)
    ax.set_ylim(-3.6, 1.8)
    figstyle.save(fig, os.path.join(FIG, "fig3_variance_surprise"))


def fig4_monte_carlo():
    mc = pd.read_csv(os.path.join(RES, "monte_carlo.csv"))
    models = ["gbm", "jumps", "heston", "scheduled"]
    names = {"gbm": "constant volatility\n(Black's world)", "jumps": "random jumps\n(Merton)",
             "heston": "wandering volatility\n(Heston)", "scheduled": "one 15% jump\non a known day"}
    order = ["static", "weekly", "move", "band", "daily", "garch"]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for k, s in enumerate(order):
        sub = mc[mc.strategy == s].set_index("model").loc[models]
        y = np.arange(len(models)) + (k - 2.5) * 0.11
        ax.scatter(sub.sd_err, y, s=36, color=COLOURS[s], label=LABELS[s], zorder=3)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([names[m] for m in models])
    ax.invert_yaxis()
    ax.set_xlabel("hedging error, standard deviation, share of premium")
    ax.set_title("Simulated markets: what breaks which hedge (2,000 paths each)", loc="left")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0)
    figstyle.save(fig, os.path.join(FIG, "fig4_monte_carlo"))


if __name__ == "__main__":
    figstyle.apply()
    fig1_frontier()
    fig2_copper_2025()
    fig3_variance_surprise()
    if os.path.exists(os.path.join(RES, "monte_carlo.csv")):
        fig4_monte_carlo()
