#!/usr/bin/env python3
#-------- make_extra_figures.py ------------------------------------------------------
#-----------------------------------------------------------------------------
# Two figures drawn from saved results (run month_end_check.py and recent_normal.py first).
#
#   fig19_month_end_lags     arabica correlation with Vietnam heat by lag, monthly average
#                            prices against month-end futures prices
#   fig20_normals            price change after a Vietnam flag, against the 1993-2016 normal
#                            and against the trailing 15-year normal

import json
import os
import sys

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import figstyle                                                  # noqa: E402

RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
NAVY, LIGHT, RED, GREY = "#0b2545", "#8cc0e8", "#c0392b", "0.5"


def fig19_month_end_lags():
    d = json.load(open(os.path.join(RES, "month_end_check.json")))["lags"]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    for name, c, lab in [("arabica Pink Sheet", LIGHT, "monthly average price (Pink Sheet)"),
                         ("arabica KC=F month-end", NAVY, "month-end futures price (ICE)")]:
        g = d[name]["vietnam_heat"]
        lags = [x["lag"] for x in g]
        ax.plot(lags, [x["r"] for x in g], "-o", color=c, lw=1.6, ms=6, label=lab)
        ax.plot(lags, [x["band"] for x in g], ":", color=c, lw=1.2)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xlabel("months Vietnam heat leads the price")
    ax.set_ylabel("Gaussian-rank correlation")
    ax.set_title("Arabica correlation with Vietnam heat by lag", loc="left")
    ax.text(0.99, 0.02, "dotted: 95% band for no relationship", transform=ax.transAxes, ha="right", fontsize=8, color=GREY)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    figstyle.save(fig, os.path.join(FIG, "fig19_month_end_lags"))


def fig20_normals():
    d = json.load(open(os.path.join(RES, "recent_normal.json")))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
    rows = [("1993-2016 normal", "against the 1993-2016 normal"), ("trailing 15-year normal", "against the 15 years before")]
    for ax, bean in zip(axes, ["arabica", "robusta"]):
        for j, (k, lab) in enumerate(rows):
            r = d[k][bean]
            m, lo, hi = 100 * r["mean_change"], 100 * r["ci90"][0], 100 * r["ci90"][1]
            ax.plot([lo, hi], [j, j], color=RED, lw=2)
            ax.plot(m, j, "o", color=RED, ms=8, mec="white", mew=1.2)
            ax.text(hi + 0.4, j, f"{r['months']} months", va="center", fontsize=8.5, color=GREY)
        ax.axvline(100 * d[rows[0][0]][bean]["all_months"], color=GREY, lw=1, ls="--")
        ax.axvline(0, color="0.75", lw=0.8)
        ax.set_yticks(range(len(rows)), [r[1] for r in rows])
        ax.set_ylim(-0.6, len(rows) - 0.4); ax.invert_yaxis()
        ax.set_xlabel("price change over the next three months (%)")
        ax.set_title(f"{bean.capitalize()}, months the forecast flagged Vietnam", loc="left", fontsize=10)
    figstyle.save(fig, os.path.join(FIG, "fig20_normals"))


if __name__ == "__main__":
    figstyle.apply()
    fig19_month_end_lags()
    fig20_normals()
