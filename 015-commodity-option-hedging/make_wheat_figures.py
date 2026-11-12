#!/usr/bin/env python3
#-------- make_wheat_figures.py -----------------------------------------------------
# Figures for both sides of the 2010 wheat hedge (wheat_2010.py):
#   fig5_wheat_2010     wheat in euros per tonne through 2010, the hedge window, the heatwave
#                       and the export ban, and the dates the layered hedge trades
#   fig6_wheat_revenue  miller cost per tonne for each hedge: what 2010
#                       gave, against the range expected on 3 May (5th to 95th percentile)
#   fig7_wheat_payoffs  miller cost against the wheat price at the end
#
#   python make_wheat_figures.py

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wheat_2010 import EVENTS, LAYERS, SALE, START, cost, revenue

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"
plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})


def ranges(ax, real, ex, names, title, xlabel, legend):
    y = np.arange(len(names))
    ax.hlines(y, [ex[s]["p5"] for s in names], [ex[s]["p95"] for s in names], color=LIGHT, lw=7,
              label="range expected on 3 May (5th-95th percentile)")
    ax.plot([real[s] for s in names], y, "D", color=NAVY, ms=7, label="2010")
    ax.axvline(real["no hedge"], color=RED, lw=0.8, ls=":")
    ax.set_yticks(y, names); ax.invert_yaxis()
    ax.set_xlabel(xlabel); ax.set_title(title, loc="left")
    if legend:
        ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.4, -0.16), ncol=2)


def main():
    o = json.load(open(os.path.join(RES, "wheat_2010.json")))
    F = pd.read_csv(os.path.join(RES, "wheat_2010_prices.csv"), index_col=0, parse_dates=True).squeeze()
    F0, b = o["F0"], o["buyer"]

    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.plot(F.index, F, color=NAVY, lw=1.6)
    ax.axvspan(pd.Timestamp(START), pd.Timestamp(SALE), color=LIGHT, alpha=0.18, lw=0)
    for j, (d, name) in enumerate(EVENTS.items()):
        ax.axvline(pd.Timestamp(d), color=RED, lw=1, ls="--")
        ax.text(pd.Timestamp(d), 224 - 9 * j, name + " ", color=RED, fontsize=9, ha="right")
    for d in LAYERS:
        ax.plot(pd.Timestamp(d), F[:d].iloc[-1], "o", color=BLUE, ms=5)
    ax.annotate("hedge set, 3 May", (pd.Timestamp(START), F0), xytext=(-95, -28), textcoords="offset points",
                arrowprops=dict(arrowstyle="-", color=GREY), fontsize=9)
    ax.annotate("wheat bought, 31 Aug", (pd.Timestamp(SALE), o["FT"]), xytext=(15, 22), textcoords="offset points",
                arrowprops=dict(arrowstyle="-", color=GREY), fontsize=9)
    ax.set_ylabel("€ per tonne"); ax.set_ylim(110, 230)
    ax.set_title("Wheat, euros per tonne, 2010", loc="left")
    fig.savefig(os.path.join(FIG, "fig5_wheat_2010.png"), dpi=170, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ranges(ax, b["realised"], b["ex_ante"], list(b["realised"]), "Miller cost by hedge",
           "cost, € per tonne (lower is better)", True)
    fig.savefig(os.path.join(FIG, "fig6_wheat_revenue.png"), dpi=170, bbox_inches="tight"); plt.close(fig)

    FT = np.linspace(90, 200, 200)
    lay = np.full_like(FT, np.mean(o["layer_prices"]))
    cst = cost(F0, FT, lay, b["setup"])
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for d, keys, title, ylab in [
            (cst, ["no hedge", "long futures", "long call", "zero-cost collar", "call spread"], "Miller cost", "cost, € per tonne")]:
        for s, c, ls in zip(keys, [GREY, NAVY, RED, BLUE, LIGHT], ["-", "-", "-", "-", "--"]):
            ax.plot(FT, d[s], color=c, lw=1.8, ls=ls, label=s)
        ax.axvline(o["FT"], color=RED, lw=0.8, ls=":"); ax.text(o["FT"], 92, " Aug 2010", color=RED, fontsize=8.5)
        ax.set_xlabel("wheat price at the end, € per tonne"); ax.set_ylabel(ylab)
        ax.set_title(title, loc="left"); ax.legend(frameon=False, fontsize=9)
    fig.savefig(os.path.join(FIG, "fig7_wheat_payoffs.png"), dpi=170, bbox_inches="tight"); plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
