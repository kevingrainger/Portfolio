#!/usr/bin/env python3
#-------- make_wheat_figures.py -----------------------------------------------------
# Figures for the wheat 2010 producer hedge (wheat_2010.py):
#   fig5_wheat_2010     CBOT wheat through 2010, with the hedge date, sale date, the
#                       heatwave and the export ban, and the strikes set on 3 May
#   fig6_wheat_revenue  revenue per bushel for each hedge: what 2010 gave, against the range
#                       a producer could have expected on 3 May (5th to 95th percentile)
#   fig7_wheat_payoffs  revenue at sale against the sale price for each hedge
#
#   python make_wheat_figures.py

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wheat_2010 import EVENTS, LAYERS, SALE, START, revenue

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"
plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})


def main():
    o = json.load(open(os.path.join(RES, "wheat_2010.json")))
    F = pd.read_csv(os.path.join(RES, "wheat_2010_prices.csv"), index_col=0, parse_dates=True).squeeze()
    k = o["setup"]; F0 = o["F0"]

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(F.index, F, color=NAVY, lw=1.6)
    ax.axvspan(pd.Timestamp(START), pd.Timestamp(SALE), color=LIGHT, alpha=0.18, lw=0)
    for j, (d, name) in enumerate(EVENTS.items()):
        ax.axvline(pd.Timestamp(d), color=RED, lw=1, ls="--")
        ax.text(pd.Timestamp(d), 8.05 - 0.3 * j, name + " ", color=RED, fontsize=9, ha="right")
    for K, name, c in [(F0, "futures sold / put strike", BLUE), (0.95 * F0, "95% put", LIGHT), (0.85 * F0, "85% put", LIGHT),
                       (k["Kc_collar"], "collar call", RED), (k["Kc_fence"], "three-way call", RED)]:
        ax.hlines(K, pd.Timestamp(START), pd.Timestamp(SALE), color=c, lw=1, ls=":")
        ax.text(pd.Timestamp(SALE), K, f" {name}", color=c, fontsize=8.5, va="center")
    for d in LAYERS:
        ax.plot(pd.Timestamp(d), F[:d].iloc[-1], "o", color=BLUE, ms=5)
    ax.annotate("hedge set", (pd.Timestamp(START), F0), xytext=(-60, -30), textcoords="offset points",
                arrowprops=dict(arrowstyle="-", color=GREY), fontsize=9)
    ax.annotate("crop sold", (pd.Timestamp(SALE), o["FT"]), xytext=(15, 20), textcoords="offset points",
                arrowprops=dict(arrowstyle="-", color=GREY), fontsize=9)
    ax.set_ylabel("$ per bushel"); ax.set_ylim(4, 8.2)
    ax.set_title("CBOT wheat, 2010", loc="left")
    fig.savefig(os.path.join(FIG, "fig5_wheat_2010.png"), dpi=170, bbox_inches="tight"); plt.close(fig)

    names = list(o["realised"])
    fig, ax = plt.subplots(figsize=(11, 4.8))
    y = np.arange(len(names))
    lo = [o["ex_ante"][s]["p5"] for s in names]; hi = [o["ex_ante"][s]["p95"] for s in names]
    ax.hlines(y, lo, hi, color=LIGHT, lw=7, label="range expected on 3 May (5th-95th percentile)")
    ax.plot([o["realised"][s] for s in names], y, "D", color=NAVY, ms=7, label="2010")
    ax.axvline(F0, color=GREY, lw=0.8, ls=":")
    ax.set_yticks(y, names); ax.invert_yaxis()
    ax.set_xlabel("revenue at sale, $ per bushel")
    ax.set_title("Producer revenue by hedge", loc="left"); ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig.savefig(os.path.join(FIG, "fig6_wheat_revenue.png"), dpi=170, bbox_inches="tight"); plt.close(fig)

    FT = np.linspace(3.2, 7.2, 200)
    rev = revenue(F0, FT, np.full_like(FT, np.mean(o["layer_prices"])), k)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    groups = [["no hedge", "short futures", "50% futures", "long put"], ["zero-cost collar", "put spread", "three-way"]]
    cols = [GREY, NAVY, BLUE, RED, NAVY, BLUE, RED]
    for ax, g, cset in zip(axes, groups, [cols[:4], cols[4:]]):
        for s, c in zip(g, cset):
            ax.plot(FT, rev[s], color=c, lw=1.8, label=s)
        ax.axvline(o["FT"], color=RED, lw=0.8, ls="--"); ax.text(o["FT"], 3.3, " Aug 2010", color=RED, fontsize=8.5)
        ax.set_xlabel("wheat price at sale, $ per bushel"); ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("revenue, $ per bushel")
    axes[0].set_title("Futures and puts", loc="left"); axes[1].set_title("Option structures", loc="left")
    fig.savefig(os.path.join(FIG, "fig7_wheat_payoffs.png"), dpi=170, bbox_inches="tight"); plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
