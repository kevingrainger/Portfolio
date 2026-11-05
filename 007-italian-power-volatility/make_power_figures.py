#!/usr/bin/env python3
#-------- make_power_figures.py -----------------------------------------------------
# Figures for the weather and Italian power price study, on the real daily series
# (data/real_daily.csv, 2018-2026):
#   fig8_stacked   TTF gas, the Italy North day-ahead price, power relative to gas (the
#                  price once gas is taken out), temperature and extreme-weather days, on one
#                  time axis, with the three worst heatwave summers shaded through every panel
#   fig9_rolling   rolling 90-day correlations: daily power against gas, and power relative to
#                  gas against temperature, so the gas link and the weather link can be seen
#                  changing through time
#
#   python make_power_figures.py

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"
EVENTS = [("2022-07-01", "2022-08-31", "heatwave 2022"), ("2023-07-01", "2023-08-31", "heatwave 2023"),
          ("2026-06-15", "2026-08-31", "heatwave 2026")]
plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})


def shade(ax, label=False):
    for k, (a, b, name) in enumerate(EVENTS):
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=RED, alpha=0.13, lw=0)
        if label:
            ax.text(pd.Timestamp(a), 0.97 - 0.09 * (k % 2), " " + name, transform=ax.get_xaxis_transform(),
                    fontsize=9, color=RED, va="top")


def main():
    d = pd.read_csv(os.path.join(HERE, "data", "real_daily.csv"), index_col=0, parse_dates=True)
    w = d.resample("W").mean()
    rel = np.log(d.price / d.gas)
    extreme_day = ((d.temp > 28) | (d.temp < 3)).astype(int)

    fig, axes = plt.subplots(4, 1, figsize=(12, 9.6), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.3, 1, 1, 1], hspace=0.38))
    ax = axes[0]
    ax.plot(w.index, w.price, color=NAVY, lw=1.4, label="Italy North power")
    ax.plot(w.index, w.gas, color=LIGHT, lw=1.4, label="TTF gas")
    ax.set_ylabel("€ per MWh"); ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 0.8)); shade(ax, label=True)
    ax.set_title("Power and gas prices (weekly mean)", loc="left")
    ax = axes[1]
    rw = rel.resample("W").mean()
    ax.plot(rw.index, rw, color=BLUE, lw=1.3)
    ax.axhline(rw.mean(), color="0.8", lw=0.8)
    ax.set_ylabel("log(power / gas)"); ax.set_title("Power relative to gas", loc="left"); shade(ax)
    ax = axes[2]
    ax.plot(w.index, w.temp, color=RED, lw=1.2)
    ax.set_ylabel("°C"); ax.set_title("Italian temperature (eight-city ERA5 mean)", loc="left"); shade(ax)
    ax = axes[3]
    m = extreme_day.resample("ME").sum()
    ax.bar(m.index, m.values, width=25, color=RED, alpha=0.8)
    ax.set_ylabel("days per month"); ax.set_title("Extreme-weather days (above 28 °C or below 3 °C)", loc="left"); shade(ax)
    fig.align_ylabels(axes)
    fig.savefig(os.path.join(FIG, "fig8_stacked.png"), dpi=170, bbox_inches="tight"); plt.close(fig)

    wk = pd.DataFrame({"power": np.log(w.price).diff(), "gas": np.log(w.gas).diff()}).dropna()
    c_gas = wk.power.rolling(26).corr(wk.gas)                   # weekly changes, half-year window
    dr = pd.DataFrame({"rel": rel, "temp_dev": (d.temp - 21).abs()}).dropna()
    c_wx = dr.rel.rolling(90).corr(dr.temp_dev)
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.8), sharex=True, gridspec_kw=dict(hspace=0.18))
    for ax, s, col, title in [(axes[0], c_gas, NAVY, "Weekly power vs gas price changes, rolling 26-week correlation"),
                              (axes[1], c_wx, RED, "Power relative to gas vs distance from 21 °C, rolling 90-day correlation")]:
        ax.plot(s.index, s, color=col, lw=1.3)
        ax.axhline(0, color="0.8", lw=0.8)
        ax.set_ylabel("correlation"); ax.set_title(title, loc="left"); shade(ax)
    fig.savefig(os.path.join(FIG, "fig9_rolling.png"), dpi=170, bbox_inches="tight"); plt.close(fig)
    print("gas corr mean", round(c_gas.mean(), 2), "weather corr mean", round(c_wx.mean(), 2),
          "weather corr in heatwaves", round(np.mean([c_wx[a:b].mean() for a, b, _ in EVENTS]), 2))


if __name__ == "__main__":
    main()
