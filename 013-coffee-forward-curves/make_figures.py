#!/usr/bin/env python3
#-------- make_figures.py ----------------------------------------------------------
#-----------------------------------------------------------------------------
# The three figures of 013 (run fetch_data.py and run_analysis.py first).
#
#   fig1_frost_2021   the July 2021 frost: what the forecasts said, what the price did
#   fig2_events       price paths around every frost night and every first warning
#   fig3_curve        how far along the curve a convenience-yield shock reaches

import json
import os
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import figstyle                                                  # noqa: E402
from coffee_data import cold_actual, front_month, warning        # noqa: E402
from schwartz import loading                                     # noqa: E402

RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
GREEN, WINE, GREY, SAND = "#117733", "#882255", "0.35", "#DDCC77"


def fig1_frost_2021():
    a, b = "2021-06-15", "2021-08-10"
    W, _ = warning("gfs")
    C = cold_actual()
    P = front_month()[a:b]
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.2]))
    ax = axes[0]
    ax.bar(W[a:b].index, W[a:b].values, width=0.8, color=SAND, label="frost warning in that day's forecast (next 7 days)")
    ax.plot(C[a:b].index, C[a:b].values, "o", ms=4, color=WINE, label="frost that night (ERA5)")
    ax.set_ylabel("cold excess below 5 C,\nproduction-weighted")
    ax.legend(loc="upper left")
    ax = axes[1]
    ax.plot(P.index, P.values, color=GREEN)
    ax.set_ylabel("arabica front month,\nUS cents/lb")
    for d, txt, xy in [("2021-07-16", "first warning", (-70, 18)), ("2021-07-20", "frost night", (-60, 30)),
                       ("2021-07-26", "harder frost forecast\nover the weekend", (-190, -8))]:
        j = P.index.searchsorted(pd.Timestamp(d))
        ax.annotate(txt, (P.index[j], P.iloc[j]), xytext=xy, textcoords="offset points", fontsize=8.5,
                    arrowprops=dict(arrowstyle="-", color="0.5", lw=0.7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes[0].set_title("July 2021: the forecasts warned, the price moved most once the damage was known", loc="left")
    fig.tight_layout()
    figstyle.save(fig, os.path.join(FIG, "fig1_frost_2021"))


def fig2_events():
    ev = pd.read_csv(os.path.join(RES, "events.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    for ax, kind, col, title in zip(axes, ["frost night (ERA5)", "first warning (GFS)"], [WINE, GREEN],
                                    ["Aligned on the frost nights of 2000, 2019 and 2021", "Aligned on the first forecast warning, 2021-2026"]):
        sub = ev[ev.kind == kind]
        for e, g in sub.groupby("event"):
            ax.plot(g.k, 100 * g.cum, color=col, alpha=0.35, lw=1)
            if not kind.startswith("frost"):
                ax.annotate(e[:7], (g.k.iloc[-1], 100 * g.cum.iloc[-1]),
                            xytext=(3, 0), textcoords="offset points", fontsize=7.5, color="0.35", va="center")
        m = sub.groupby("k").cum.mean()
        ax.plot(m.index, 100 * m.values, color=col, lw=2.4, label="average")
        ax.axvline(0, color="0.6", lw=0.8, ls=":")
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("trading days from the event")
        ax.legend(loc="upper left")
        ax.set_xlim(-10, 13)
    axes[0].set_ylabel("cumulative change in price, %")
    fig.tight_layout()
    figstyle.save(fig, os.path.join(FIG, "fig2_events"))


def fig3_curve():
    s = json.load(open(os.path.join(RES, "schwartz.json")))
    p = s["no_weather"]
    st = pd.read_csv(os.path.join(RES, "schwartz_states.csv"), index_col=0, parse_dates=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw=dict(width_ratios=[1, 1.3]))
    tau = np.linspace(0.01, 4, 200)
    ax = axes[0]
    ax.plot(tau, loading(p["kappa"], tau) / tau, color=GREEN)
    ax.set_xlabel("contract maturity, years")
    ax.set_ylabel("share of a convenience-yield shock\npassed to the contract's price")
    ax.set_title(f"How far along the curve (kappa = {p['kappa']:.2f})", loc="left", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax = axes[1]
    ax.plot(st.index, 100 * st.delta, color=WINE)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_ylabel("convenience yield, % a year")
    ax.set_title("The convenience yield the curve implies, 2024-2026", loc="left", fontsize=10)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.tight_layout()
    figstyle.save(fig, os.path.join(FIG, "fig3_curve"))


if __name__ == "__main__":
    figstyle.apply()
    fig1_frost_2021()
    fig2_events()
    fig3_curve()
