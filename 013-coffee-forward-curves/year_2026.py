#!/usr/bin/env python3
#-------- year_2026.py --------------------------------------------------------------
#-----------------------------------------------------------------------------
# The tail rule in 2026, month by month. Each month's ECMWF SEAS5 forecast of Vietnam heat
# for the next three months (as standard deviations from the 1993-2016 hindcast normal),
# against the heat that then happened (ERA5) and the robusta and arabica prices (World Bank
# Pink Sheet). The rule flags a month when the forecast is above one standard deviation.
#
#   python year_2026.py
#
# Output: results/year_2026.csv, figures/fig18_2026.png

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from seasonal_backtest import build, pink_sheet

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"


def main():
    d = build()["2025-07":]
    px = pink_sheet()["2025-07":]
    cols = ["vietnam_heat_fc", "vietnam_heat_next3", "brazil_dry_fc", "robusta_next3_raw", "arabica_next3_raw"]
    d[cols].to_csv(os.path.join(RES, "year_2026.csv"))
    print(d[cols].round(2).to_string())

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True, gridspec_kw=dict(hspace=0.15, height_ratios=[1, 1]))
    ax = axes[0]
    ax.axhspan(1, max(6, d.vietnam_heat_fc.max() + 0.5), color=RED, alpha=0.07, lw=0)
    ax.text(d.index[0], 1.15, " flag: forecast above 1 standard deviation", color=RED, fontsize=9)
    ax.plot(d.index, d.vietnam_heat_fc, "-o", color=RED, lw=1.6, ms=5, label="SEAS5 forecast for the next three months, issued that month")
    ax.plot(d.index, d.vietnam_heat_next3, "-s", color=NAVY, lw=1.4, ms=4, label="what happened over those three months (ERA5)")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_ylabel("Vietnam heat, standard deviations"); ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Vietnam heat in the Central Highlands, forecast and outcome", loc="left")
    ax = axes[1]
    for bean, c, ls in [("robusta", BLUE, "-"), ("arabica", NAVY, "--")]:
        ax.plot(px.index, 100 * px[bean] / px[bean].iloc[0], ls, color=c, lw=1.8, label=bean)
    flag = d.index[d.vietnam_heat_fc > 1]
    for t in flag:
        ax.axvspan(t - pd.Timedelta(days=14), t + pd.Timedelta(days=14), color=RED, alpha=0.12, lw=0)
    ax.set_ylabel(f"price, {px.index[0]:%B %Y} = 100"); ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title("Arabica and robusta prices, with the months the rule flagged in red", loc="left")
    fig.savefig(os.path.join(FIG, "fig18_2026.png"), dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
